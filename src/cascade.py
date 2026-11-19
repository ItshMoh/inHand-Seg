"""Phase-5 inference and composed 3-class scoring. Existing RGB evaluate() is unchanged."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import DATA_ROOT, IGNORE_INDEX, EgoHOSDataset, EvalTransform, remap_label
from src.dataset_stats import EXPECTED_COUNTS
from src.evaluate import (
    boundary_width, confusion_matrix, display_source, eval_checks, iou_from_confusion,
    mask_boundary, pick_vis_names, vis_file_stem,
)
from src.model import load_model
from src.sanity import _display_size, _figure, _to_png, overlay_classes

CLASS_NAMES = ["background", "hand", "interacting_object"]
MODES = ("predicted", "oracle")


def load_stage1(path, device="cuda"):
    model, ckpt = load_model(path)
    cfg = ckpt["config"]
    if cfg["model"]["class_names"] != ["background", "hand"] or cfg["data"].get("label_map") != {2: 0}:
        raise ValueError("Stage 1 must be the hands-only model with label_map {2: 0}")
    if model.config.num_channels != 3:
        raise ValueError("Stage 1 must take RGB")
    return model.to(device).requires_grad_(False).eval()


def compose_mask(objects: torch.Tensor, hands: torch.Tensor) -> torch.Tensor:
    """At original resolution, paint objects first and hands last."""
    if objects.shape != hands.shape:
        raise ValueError("Compose masks only after resizing both logits to original resolution")
    pred = (objects == 1).long() * 2
    pred[hands.bool()] = 1
    return pred


@torch.no_grad()
def cascade_predictions(stage1, stage2, image, original_size, gt=None, modes=MODES, bf16=True):
    """One RGB image. Returns each composed mask and the exact binary hand input (at model scale).

    Stage-1 logits are independently upsampled to RGB-input and original sizes before argmax.
    Never resize an input-scale argmax for the pasted hands: that would change stage-1 IoU.
    Predicted-only inference needs no GT; oracle uses GT both as input and pasted hands.
    """
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError(f"modes must be drawn from {MODES}")
    device_type = image.device.type
    if stage1 is not None:
        stage1.eval()
    stage2.eval()
    hands = {}
    if "predicted" in modes:
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16):
            logits = stage1(pixel_values=image).logits
        at_input = F.interpolate(logits.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
        at_original = F.interpolate(logits.float(), size=original_size, mode="bilinear", align_corners=False)
        hands["predicted"] = ((at_input.argmax(1, keepdim=True) == 1).float(), at_original.argmax(1)[0] == 1)
    if "oracle" in modes:
        if gt is None or tuple(gt.shape) != tuple(original_size):
            raise ValueError("Oracle requires the clean original-resolution GT")
        original = gt == 1  # 255 and 2nd-order objects can never become hands
        at_input = F.interpolate(original[None, None].float(), size=image.shape[-2:], mode="nearest-exact")
        hands["oracle"] = (at_input, original)
    outputs = {}
    for mode in modes:
        hand_input, original_hands = hands[mode]
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16):
            logits = stage2(pixel_values=torch.cat([image, hand_input], dim=1)).logits
        logits = F.interpolate(logits.float(), size=original_size, mode="bilinear", align_corners=False)
        outputs[mode] = {
            "prediction": compose_mask(logits.argmax(1)[0], original_hands),
            "hand_input": hand_input[0, 0],
        }
    return outputs


class Scores:
    """Same confusion-matrix and Boundary-IoU definitions as src.evaluate.evaluate()."""

    def __init__(self, device, ratio):
        self.ratio = ratio
        self.hist = torch.zeros(3, 3, dtype=torch.int64, device=device)
        self.b_inter = torch.zeros(3, dtype=torch.int64, device=device)
        self.b_union = torch.zeros(3, dtype=torch.int64, device=device)
        self.per_image = {}

    def update(self, name, pred, target):
        hist = confusion_matrix(pred, target, 3)
        self.hist += hist
        d = boundary_width(*target.shape, self.ratio)
        valid = target != IGNORE_INDEX
        for c in (1, 2):
            gt_b = mask_boundary(target == c, d)
            pred_b = mask_boundary(pred == c, d) & valid
            self.b_inter[c] += (gt_b & pred_b).sum()
            self.b_union[c] += (gt_b | pred_b).sum()
        self.per_image[name] = {
            "size": [int(target.shape[1]), int(target.shape[0])],
            "iou": dict(zip(CLASS_NAMES, iou_from_confusion(hist.cpu().numpy()))),
        }

    def result(self, split):
        hist = self.hist.cpu().numpy()
        iou = iou_from_confusion(hist)
        inter, union = self.b_inter.cpu().numpy(), self.b_union.cpu().numpy()
        biou = {CLASS_NAMES[c]: float(inter[c] / union[c]) if union[c] else None for c in (1, 2)}

        def mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None

        return {
            "split": split, "n_images": len(self.per_image), "class_names": CLASS_NAMES,
            "iou": dict(zip(CLASS_NAMES, iou)), "miou": mean(iou), "miou_fg": mean(iou[1:]),
            "boundary_iou": biou, "boundary_miou_fg": mean(biou.values()),
            "boundary_dilation_ratio": self.ratio,
            "pixel_accuracy": float(np.trace(hist) / max(hist.sum(), 1)),
            "confusion_matrix": hist.tolist(), "per_image": self.per_image,
        }


@torch.no_grad()
def evaluate_cascade(stage1, stage2, dataset, cfg, device="cuda", keep=(), modes=MODES):
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"],
                        pin_memory=device.type == "cuda")
    scores = {mode: Scores(device, cfg["eval"]["boundary_dilation_ratio"]) for mode in modes}
    kept = {mode: {} for mode in modes}
    t0 = time.perf_counter()
    for i, (image, target) in enumerate(loader):
        name = dataset.names[i]
        image, target = image.to(device), target[0].to(device)
        outputs = cascade_predictions(stage1, stage2, image, target.shape, target, modes, cfg["train"]["bf16"])
        for mode, output in outputs.items():
            scores[mode].update(name, output["prediction"], target)
            if name in keep:
                kept[mode][name] = {key: value.cpu().numpy().astype(np.uint8) for key, value in output.items()}
    seconds = round(time.perf_counter() - t0, 1)
    metrics = {}
    for mode, score in scores.items():
        metrics[mode] = {
            **score.result(dataset.split), "eval_seconds": seconds, "hand_mode": mode,
            "eval_seconds_scope": "all requested hand modes together",
            "run_name": cfg["run_name"], "stage1_checkpoint": cfg["cascade"]["stage1_checkpoint"],
            "stage2_label_map": cfg["data"]["label_map"], "hand_input_encoding": "binary_0_1",
            "oracle_pastes_gt_hands": mode == "oracle",
        }
    return metrics, kept


def render_cascade_panel(name, image, gt, kept, metrics):
    disp_image, disp_gt = _display_size(image, gt)
    _, disp_pred = _display_size(image, kept["prediction"])
    # Overlay the exact received channel; display resizing only, never used for inference.
    hand = cv2.resize(kept["hand_input"], (disp_gt.shape[1], disp_gt.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
    fig, axes = _figure(5, *disp_gt.shape, col_width=4.5)
    scores = metrics["per_image"][name]["iou"]
    title_scores = "  ".join(f"{c} {v:.3f}" for c, v in scores.items() if c != "background" and v is not None)
    fig.suptitle(f"{name}  {display_source(name)}  {metrics['hand_mode']}  {title_scores}", fontsize=10)
    panels = [disp_image, overlay_classes(disp_image, disp_gt), overlay_classes(disp_image, hand),
              overlay_classes(disp_image, disp_pred)]
    errors = overlay_classes(disp_image, np.where(disp_gt == IGNORE_INDEX, IGNORE_INDEX, 0).astype(np.uint8))
    errors[(disp_pred != disp_gt) & (disp_gt != IGNORE_INDEX)] = (230, 25, 25)
    panels.append(errors)
    for ax, panel, title in zip(axes, panels, ("image", "ground truth", "hand input (display resized)", "composed prediction", "errors")):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)
    return _to_png(fig)


def stage1_identity_checks(metrics, cfg, split):
    """Compare with existing seed-0 scores, without reevaluating or writing the stage-1 run."""
    ref_path = Path(cfg["cascade"]["stage1_checkpoint"]).parent / "eval" / f"{split}.json"
    checks = {}
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        if ref["n_images"] != metrics["n_images"]:
            raise ValueError("Stage-1 reference and cascade must score the same full split")
        for key in ("iou", "boundary_iou"):
            checks[f"{split}_predicted_hand_{key}_matches_stage1"] = metrics[key]["hand"] == ref[key]["hand"]
    return checks


def evaluate_cascade_split(stage1, stage2, cfg, split, run_dir, train_names=None, device="cuda", data_root=DATA_ROOT):
    dataset = EgoHOSDataset(split, data_root, EvalTransform(), file_list=train_names if split == "train" else None)
    vis_names = pick_vis_names(dataset, cfg["eval"]["n_vis"]) if cfg["eval"]["n_vis"] else []
    metrics, kept = evaluate_cascade(stage1, stage2, dataset, cfg, device, vis_names)
    files, summaries, checks = {}, {}, {}
    for mode, result in metrics.items():
        result["vis_names"] = vis_names
        files[f"eval/{split}_{mode}.json"] = json.dumps(result, indent=2).encode()
        expected_n = len(train_names) if split == "train" else EXPECTED_COUNTS.get(split)
        checks.update({f"{mode}_{k}": v for k, v in eval_checks(result, expected_n).items()})
        summaries[mode] = {k: v for k, v in result.items() if k != "per_image"}
    checks.update(stage1_identity_checks(metrics["predicted"], cfg, split))
    checks[f"{split}_oracle_hand_iou_is_1"] = metrics["oracle"]["iou"]["hand"] in (1.0, None)
    index = {name: i for i, name in enumerate(dataset.names)}
    for j, name in enumerate(vis_names):
        image, label = dataset.load(index[name])
        for mode in MODES:
            rel = f"eval/{split}_{mode}_vis/{j}_{vis_file_stem(name)}.png"
            files[rel] = render_cascade_panel(name, image, remap_label(label), kept[mode][name], metrics[mode])
    for rel, data in files.items():
        destination = Path(run_dir) / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return {"metrics": summaries, "checks": checks, "files": files}


def evaluate_cascade_run(run_dir, split, device="cuda", data_root=DATA_ROOT):
    from src.data import read_file_list

    run_dir = Path(run_dir)
    stage2, ckpt = load_model(run_dir / "model_final.pt")
    cfg = ckpt["config"]
    stage1 = load_stage1(cfg["cascade"]["stage1_checkpoint"], device)
    names = read_file_list(run_dir / "train_names.txt") if split == "train" else None
    return evaluate_cascade_split(stage1, stage2.to(device), cfg, split, run_dir, names, device, data_root)
