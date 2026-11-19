"""Full-resolution evaluation (Phase 2 protocol): predict at a 512 px shorter side, upsample the logits bilinearly
to the original size, argmax, score against the original-resolution label (255 = ignore, excluded).

Metrics: per-class IoU, mIoU and foreground mIoU from a dataset-level confusion matrix, and Boundary IoU
(Cheng et al. 2021) for the foreground classes, accumulated over the dataset the same way.
"""

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.data import DATA_ROOT, IGNORE_INDEX, EgoHOSDataset, EvalTransform, read_file_list, remap_label, source_of
from src.dataset_stats import EXPECTED_COUNTS
from src.labels import apply_label_map, label_lut, with_label_map
from src.model import load_model
from src.sanity import _class_legend, _display_size, _figure, _to_png, overlay_classes

ERROR_COLOUR = (230, 25, 25)
VIS_SEED = 0  # the same picks for every run, whatever the run's seed


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Rows = ground truth, columns = prediction; ignore pixels are excluded."""
    valid = target != IGNORE_INDEX
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_confusion(hist: np.ndarray) -> list[float | None]:
    """Per-class IoU; None for a class absent from both ground truth and prediction."""
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(0) + hist.sum(1) - inter
    return [float(i / u) if u > 0 else None for i, u in zip(inter, union)]


def boundary_width(height: int, width: int, dilation_ratio: float) -> int:
    """d = dilation_ratio x image diagonal, at least 1 px (as in the official boundary-iou-api)."""
    return max(1, int(round(dilation_ratio * (height**2 + width**2) ** 0.5)))


def mask_boundary(mask: torch.Tensor, d: int) -> torch.Tensor:
    """Pixels of a bool HW mask within d px (chessboard distance) of its contour: mask minus its erosion by a
    (2d+1)^2 square. The image border counts as outside the mask, matching the official implementation
    (zero pad + 3x3 erosion iterated d times). Separable max-pool on the inverted mask, so 4K images stay fast."""
    k = 2 * d + 1
    outside = F.pad((~mask).float()[None, None], (d, d, d, d), value=1.0)
    outside = F.max_pool2d(outside, (1, k), stride=1)
    outside = F.max_pool2d(outside, (k, 1), stride=1)
    return mask & (outside[0, 0] > 0)


def display_source(name: str) -> str:
    """source_of() for titles; out-of-domain frames are named youtube_<video>_<frame> (source_of says "own")."""
    return "youtube" if name.startswith("youtube_") else source_of(name)


def video_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def vis_file_stem(name: str, max_len: int = 80) -> str:
    """File-system-safe stem; out-of-domain names have spaces, brackets and ~140 characters (keeps the frame suffix)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe if len(safe) <= max_len else f"{safe[:50]}__{safe[-28:]}"


def pick_vis_names(dataset: EgoHOSDataset, n: int, seed: int = VIS_SEED) -> list[str]:
    """Fixed picks so every run shows the same images. test_outdomain (all 1920x1080, all YouTube): one frame from
    each of n different videos. Other splits: smallest and largest image, one per source, then random."""
    names = list(dataset.names)
    if n >= len(names):
        return names
    rng = random.Random(seed)
    if dataset.split == "test_outdomain":
        by_video: dict[str, list[str]] = {}
        for name in sorted(names):
            by_video.setdefault(video_of(name), []).append(name)
        picked = [rng.choice(by_video[v]) for v in rng.sample(sorted(by_video), min(n, len(by_video)))]
    else:
        area = {}
        for name in names:
            with Image.open(dataset.images[name]) as im:
                area[name] = im.size[0] * im.size[1]
        by_area = sorted(names, key=lambda x: (area[x], x))
        picked = [by_area[0], by_area[-1]][:n]
        for source in ("ego4d", "epic", "thu", "own"):
            pool = sorted(x for x in names if source_of(x) == source and x not in picked)
            if pool and len(picked) < n:
                picked.append(rng.choice(pool))
    rest = sorted(set(names) - set(picked))
    return picked + rng.sample(rest, n - len(picked))


@torch.no_grad()
def evaluate(
    model,
    dataset: EgoHOSDataset,
    class_names: list[str],
    boundary_dilation_ratio: float = 0.02,
    num_workers: int = 8,
    keep: tuple[str, ...] | list[str] = (),
    bf16: bool = True,
    device: str | torch.device = "cuda",
) -> tuple[dict, dict[str, np.ndarray]]:
    """Returns (metrics, predictions for the names in `keep` as uint8 HW arrays at original resolution).
    `dataset` must use EvalTransform (mask at original resolution), so batch size is 1."""
    device = torch.device(device)
    k = len(class_names)
    fg = range(1, k)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")

    model.eval()
    hist = torch.zeros(k, k, dtype=torch.int64, device=device)
    b_inter = torch.zeros(k, dtype=torch.int64, device=device)
    b_union = torch.zeros(k, dtype=torch.int64, device=device)
    per_image, kept = {}, {}
    t0 = time.perf_counter()
    for i, (image, mask) in enumerate(loader):
        name = dataset.names[i]
        image = image.to(device, non_blocking=True)
        mask = mask[0].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(pixel_values=image).logits
        logits = F.interpolate(logits.float(), size=mask.shape, mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0]

        h = confusion_matrix(pred, mask, k)
        hist += h
        d = boundary_width(*mask.shape, boundary_dilation_ratio)
        valid = mask != IGNORE_INDEX
        for c in fg:
            gt_b = mask_boundary(mask == c, d)
            pred_b = mask_boundary(pred == c, d) & valid
            b_inter[c] += (gt_b & pred_b).sum()
            b_union[c] += (gt_b | pred_b).sum()

        per_image[name] = {
            "size": [int(mask.shape[1]), int(mask.shape[0])],
            "iou": dict(zip(class_names, iou_from_confusion(h.cpu().numpy()))),
        }
        if name in keep:
            kept[name] = pred.to(torch.uint8).cpu().numpy()

    hist_np = hist.cpu().numpy()
    iou = iou_from_confusion(hist_np)
    b_inter, b_union = b_inter.cpu().numpy(), b_union.cpu().numpy()
    biou = {class_names[c]: float(b_inter[c] / b_union[c]) if b_union[c] > 0 else None for c in fg}
    valid_iou = [v for v in iou if v is not None]
    valid_fg = [iou[c] for c in fg if iou[c] is not None]
    valid_biou = [v for v in biou.values() if v is not None]
    metrics = {
        "split": dataset.split,
        "n_images": len(dataset),
        "class_names": class_names,
        "iou": dict(zip(class_names, iou)),
        "miou": float(np.mean(valid_iou)) if valid_iou else None,
        "miou_fg": float(np.mean(valid_fg)) if valid_fg else None,
        "boundary_iou": biou,
        "boundary_miou_fg": float(np.mean(valid_biou)) if valid_biou else None,
        "boundary_dilation_ratio": boundary_dilation_ratio,
        "pixel_accuracy": float(np.trace(hist_np) / max(hist_np.sum(), 1)),
        "confusion_matrix": hist_np.tolist(),
        "eval_seconds": round(time.perf_counter() - t0, 1),
        "per_image": per_image,
    }
    return metrics, kept


def render_prediction_panel(
    name: str, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, iou: dict, num_classes: int = 3
) -> bytes:
    """image | ground truth | prediction | errors (wrong pixels red, ignore hatched grey)."""
    h, w = gt.shape
    disp_image, disp_gt = _display_size(image, gt)
    _, disp_pred = _display_size(image, pred)
    scores = "   ".join(f"IoU {c} {v:.3f}" for c, v in iou.items() if c != "background" and v is not None)
    fig, axes = _figure(4, *disp_gt.shape, col_width=4.5)
    fig.suptitle(f"{name}   source={display_source(name)}   {w}x{h}   {scores}", fontsize=10)

    axes[0].imshow(disp_image)
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(overlay_classes(disp_image, disp_gt))
    axes[1].set_title("ground truth", fontsize=9)
    axes[2].imshow(overlay_classes(disp_image, disp_pred))
    axes[2].set_title("prediction", fontsize=9)
    ignore_only = np.where(disp_gt == IGNORE_INDEX, IGNORE_INDEX, 0).astype(np.uint8)
    errors = overlay_classes(disp_image, ignore_only)
    wrong = (disp_pred != disp_gt) & (disp_gt != IGNORE_INDEX)
    errors[wrong] = ERROR_COLOUR
    axes[3].imshow(errors)
    axes[3].set_title(f"errors (red): {wrong.mean():.1%} of display pixels", fontsize=9)
    axes[1].legend(handles=_class_legend(num_classes), loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=8, frameon=False)
    return _to_png(fig)


def eval_checks(metrics: dict, expected_n: int | None, min_fg_iou: float | None = None) -> dict[str, bool]:
    split = metrics["split"]
    values = [v for v in metrics["iou"].values() if v is not None]
    values += [v for v in metrics["boundary_iou"].values() if v is not None]
    checks = {
        f"{split}_n_images_{expected_n}": expected_n is None or metrics["n_images"] == expected_n,
        f"{split}_ious_in_0_1": bool(values) and all(0.0 <= v <= 1.0 for v in values),
    }
    if min_fg_iou is not None:
        fg = [metrics["iou"][c] for c in metrics["class_names"][1:]]
        checks[f"{split}_fg_iou_ge_{min_fg_iou}"] = all(v is not None and v >= min_fg_iou for v in fg)
    return checks


def evaluate_split(
    model, cfg: dict, split: str, run_dir: str | Path, train_names: list[str] | None = None,
    device="cuda", data_root: str = DATA_ROOT,
) -> dict:
    """Evaluate one split at original resolution and write eval/<split>.json + eval/<split>_vis/*.png in the run dir.
    split "train" means the images this run trained on (`train_names`). Returns metrics, checks and the files written."""
    run_dir = Path(run_dir)
    ev = cfg["eval"]
    class_names = cfg["model"]["class_names"]
    label_map = cfg["data"].get("label_map")  # absent in Phase 3 checkpoints
    lut = label_lut(label_map, len(class_names))
    file_list = train_names if split == "train" else None
    dataset = EgoHOSDataset(split, data_root, with_label_map(EvalTransform(), lut), file_list=file_list)
    vis_names = pick_vis_names(dataset, ev["n_vis"]) if ev["n_vis"] else []

    metrics, preds = evaluate(
        model, dataset, class_names, ev["boundary_dilation_ratio"], cfg["data"]["num_workers"],
        keep=vis_names, bf16=cfg["train"]["bf16"], device=device,
    )
    metrics["run_name"] = cfg["run_name"]
    metrics["label_map"] = label_map
    metrics["vis_names"] = vis_names

    files = {f"eval/{split}.json": json.dumps(metrics, indent=2).encode()}
    index = {n: i for i, n in enumerate(dataset.names)}
    for j, name in enumerate(vis_names):
        image, label = dataset.load(index[name])
        gt = apply_label_map(remap_label(label), lut)
        panel = render_prediction_panel(name, image, gt, preds[name], metrics["per_image"][name]["iou"], len(class_names))
        files[f"eval/{split}_vis/{j}_{vis_file_stem(name)}.png"] = panel

    for rel, data in files.items():
        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / rel).write_bytes(data)

    expected_n = len(train_names) if split == "train" else EXPECTED_COUNTS.get(split)
    summary = {k: v for k, v in metrics.items() if k != "per_image"}
    return {"metrics": summary, "checks": eval_checks(metrics, expected_n, ev["min_fg_iou"]), "files": files}


def evaluate_run(run_dir: str | Path, split: str, device="cuda", data_root: str = DATA_ROOT) -> dict:
    """Evaluate a finished run's model_final.pt on one split (writes into the run dir like evaluate_split)."""
    run_dir = Path(run_dir)
    model, ckpt = load_model(run_dir / "model_final.pt")
    if ckpt["config"].get("cascade", {}).get("enabled"):
        from src.cascade import evaluate_cascade_run

        return evaluate_cascade_run(run_dir, split, device, data_root)
    train_names = read_file_list(run_dir / "train_names.txt") if split == "train" else None
    return evaluate_split(model.to(device), ckpt["config"], split, run_dir, train_names, device, data_root)
