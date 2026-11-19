"""CPU-only Phase-5 checks on synthetic data. No Modal or real training data.

    .venv/bin/python scripts/check_phase5_local.py [--init-only]

The user runs this to inspect the init report. The full check also covers clean-target
isolation, worker seeding, original-resolution composition, resume, saved 4-channel
reload, both eval modes/curves/panels, frozen stage 1 and the Phase-5 table.
"""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from check_phase3_local import TRAIN_SIZES, VAL_SIZES, make_split
from phase5_table import build as build_table
from src.cascade import cascade_predictions, compose_mask, load_stage1
from src.cascade_data import CascadeTrainTransform, corrupt_hand
from src.config import DEFAULTS, _merge, load_config
from src.data import EgoHOSDataset, TrainTransform, remap_label
from src.evaluate import evaluate_run, evaluate_split
from src.model import add_hand_channel, build_model, first_patch_projection, load_model, save_model
from src.train import run, seed_worker

RESULTS = {}


def check(name, ok):
    RESULTS[name] = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def init_check():
    torch.manual_seed(0)
    rgb_model = build_model(["background", "interacting_object"]).eval()
    expanded = copy.deepcopy(rgb_model)
    _, old = first_patch_projection(rgb_model)
    add_hand_channel(expanded)
    _, new = first_patch_projection(expanded)
    print(f"Loaded first projection: {old}; expanded: {new}", flush=True)
    check("projection uses loaded shape, preserves RGB weights and bias, fourth channel is exactly zero",
          new.weight.shape == (old.out_channels, 4, *old.kernel_size)
          and torch.equal(old.weight, new.weight[:, :3]) and torch.equal(old.bias, new.bias)
          and torch.count_nonzero(new.weight[:, 3]).item() == 0 and expanded.config.num_channels == 4)
    before, after = rgb_model.state_dict(), expanded.state_dict()
    check("every other parameter/buffer, including the decoder, is identical",
          all(torch.equal(before[k], after[k]) for k in before if before[k].shape == after[k].shape))
    rgb = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        reference = rgb_model(pixel_values=rgb).logits
        for label, hand in (("zeros", torch.zeros(2, 1, 64, 96)), ("ones", torch.ones(2, 1, 64, 96)),
                            ("arbitrary nonbinary values", torch.randn(2, 1, 64, 96) * 7 - 2)):
            actual = expanded(pixel_values=torch.cat([rgb, hand], 1)).logits
            error = (reference - actual).abs().max().item()
            check(f"init logits match for {label}, max absolute error={error:.3g}",
                  torch.allclose(actual, reference, atol=2e-6, rtol=1e-5))
    return expanded


class IdentityRGB:
    """Fixed geometry for isolating worker corruption RNGs from RGB augmentation."""
    @property
    def aug(self):
        return self

    def set_random_seed(self, seed):
        pass

    def __call__(self, image, mask):
        return image.astype(np.float32) / 255, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    expanded = init_check()
    if not args.init_only:
        integration_checks(expanded)
    print(f"\n{sum(RESULTS.values())}/{len(RESULTS)} checks passed", flush=True)
    if not all(RESULTS.values()):
        raise SystemExit(1)


def integration_checks(expanded):
    cfg0 = load_config(REPO / "configs/b0_cascade_stage2.yaml")
    cfg1 = load_config(REPO / "configs/b0_cascade_stage2_s1.yaml")
    check("cascade seed configs differ only in seed/run name",
          {k: v for k, v in cfg0.items() if k not in ("run_name", "seed")} ==
          {k: v for k, v in cfg1.items() if k not in ("run_name", "seed")} and cfg1["seed"] == 1)
    tmp = Path(tempfile.mkdtemp(prefix="phase5_check_"))
    root, runs = tmp / "data", tmp / "runs"
    rng = np.random.default_rng(0)
    make_split(root, "train", TRAIN_SIZES, rng)
    for split in ("val", "test_indomain", "test_outdomain"):
        make_split(root, split, VAL_SIZES, rng)
    dataset = EgoHOSDataset("train", root)
    image, original = dataset.load(0)
    remapped = remap_label(original)
    transform = CascadeTrainTransform(TrainTransform(), cfg0["cascade"]["corruption"], seed=17)
    plain = TrainTransform(seed=17)
    inputs, target, clean, _ = transform.draw(image, remapped)
    rgb, plain_clean = plain(image, remapped)
    check("RGB geometry/colour/normalisation exactly reuse Phase 2; target is the clean mapped label",
          np.array_equal(inputs[..., :3], rgb) and np.array_equal(clean, plain_clean)
          and np.array_equal(target, np.where(clean == 1, 0, np.where(clean == 2, 1, clean))))
    check("hand input is binary float32; clean target retains 255 and never contains class 2",
          inputs.dtype == np.float32 and set(np.unique(inputs[..., 3])) <= {0, 1}
          and set(np.unique(target)) <= {0, 1, 255} and np.array_equal(target == 255, clean == 255))
    before = remapped.copy()
    for _ in range(20):
        transform(image, remapped)
    check("repeated corruption never mutates source labels", np.array_equal(before, remapped))

    # Every recipe is applied on identical geometry; only the hand input may change.
    targets, rgbs, hands = [], [], []
    for morph in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        recipe = {**cfg0["cascade"]["corruption"], "morph_probs": morph, "drop_prob": 0, "blob_prob": 0}
        x, y = CascadeTrainTransform(TrainTransform(), recipe, seed=3)(image, remapped)
        targets.append(y)
        rgbs.append(x[..., :3])
        hands.append(x[..., 3])
    check("changing corruption changes only the hand input, never RGB or loss target",
          all(np.array_equal(targets[0], t) for t in targets) and all(np.array_equal(rgbs[0], x) for x in rgbs)
          and not np.array_equal(hands[0], hands[1]) and not np.array_equal(hands[0], hands[2]))
    mask = np.zeros((128, 128), np.uint8)
    mask[10:30, 10:30], mask[70:100, 70:100] = 1, 1
    drop = {**cfg0["cascade"]["corruption"], "morph_probs": [1, 0, 0], "drop_prob": 1, "blob_prob": 0}
    dropped, _ = corrupt_hand(mask, drop, np.random.default_rng(0))
    check("drop removes one whole component, retains the other, never mutates input",
          dropped.sum() in (400, 900) and mask.sum() == 1300)

    recipe = {**drop, "drop_prob": 0, "blob_prob": 1}
    ds = EgoHOSDataset("train", root, CascadeTrainTransform(IdentityRGB(), recipe), file_list=[dataset.names[0]] * 4)
    def worker_draws():
        torch.manual_seed(123)
        return list(DataLoader(ds, batch_size=1, num_workers=2, worker_init_fn=seed_worker))
    a, b = worker_draws(), worker_draws()
    check("worker seeds independently reseed corruption, reproducibly for the same launch seed",
          not torch.equal(a[0][0][:, 3], a[1][0][:, 3])
          and all(torch.equal(x[0], y[0]) for x, y in zip(a, b)))

    # Synthetic stage-1 fixture: save random weights; do not train or modify it.
    stage1_dir = runs / "local_hands"
    stage1_dir.mkdir(parents=True)
    hands_cfg = _merge(DEFAULTS, {"run_name": "local_hands", "data": {"label_map": {2: 0}, "num_workers": 0},
                                  "model": {"class_names": ["background", "hand"]},
                                  "train": {"bf16": False}, "eval": {"n_vis": 0}})
    stage1_model = build_model(["background", "hand"])
    stage1_path = stage1_dir / "model_final.pt"
    save_model(stage1_path, stage1_model, config=hands_cfg)
    digest = hashlib.sha256(stage1_path.read_bytes()).hexdigest()
    stage1 = load_stage1(stage1_path, "cpu")
    check("stage 1 is frozen and in eval mode", not stage1.training and not any(p.requires_grad for p in stage1.parameters()))
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 96)
        size = (101, 137)
        gt = torch.zeros(size, dtype=torch.long)
        gt[5:70, 10:80] = 1
        gt[:5] = 255
        expected = F.interpolate(stage1(pixel_values=x).logits.float(), size=size, mode="bilinear", align_corners=False).argmax(1)[0] == 1
        outputs = cascade_predictions(stage1, expanded, x, size, gt, bf16=False)
        check("predicted hands equal stage-1 original-resolution argmax exactly; oracle pastes GT hands",
              torch.equal(outputs["predicted"]["prediction"] == 1, expected)
              and torch.equal(outputs["oracle"]["prediction"] == 1, gt == 1))
        wanted = F.interpolate((gt == 1)[None, None].float(), size=x.shape[-2:], mode="nearest-exact")[0, 0]
        check("oracle hand input uses clean GT, nearest-exact resize and binary 0/1",
              torch.equal(outputs["oracle"]["hand_input"], wanted))
        predicted_only = cascade_predictions(stage1, expanded, x, size, modes=("predicted",), bf16=False)
        check("predicted inference needs no GT", torch.equal(predicted_only["predicted"]["prediction"], outputs["predicted"]["prediction"]))
        check("hand class takes precedence over objects", compose_mask(torch.ones(2, 2), torch.tensor([[1, 0], [0, 1]])).tolist() == [[1, 2], [2, 1]])

    base_cfg = _merge(DEFAULTS, {"run_name": "local_full", "data": {"num_workers": 0}, "train": {"bf16": False}, "eval": {"n_vis": 0}})
    baseline = build_model(DEFAULTS["model"]["class_names"])
    for split in ("val", "test_indomain", "test_outdomain"):
        evaluate_split(stage1, hands_cfg, split, stage1_dir, device="cpu", data_root=str(root))
        evaluate_split(baseline, base_cfg, split, runs / "local_full", device="cpu", data_root=str(root))

    cfg = _merge(cfg0, {
        "run_name": "local_cascade", "data": {"train_root": str(root), "num_workers": 0},
        "train": {"max_iters": 4, "batch_size": 2, "warmup_iters": 1, "log_every": 2, "ckpt_every": 2, "bf16": False},
        "eval": {"val_every": 2, "n_vis": 1}, "cascade": {"stage1_checkpoint": str(stage1_path)},
    })
    class Interrupt(Exception):
        pass
    def interrupt():
        if (runs / cfg["run_name"] / "checkpoint_last.pt").exists():
            raise Interrupt()
    try:
        run(cfg, runs, REPO, device="cpu", commit=interrupt, data_root=str(root))
        check("synthetic run interrupted at checkpoint", False)
    except Interrupt:
        check("synthetic run interrupted at checkpoint", True)
    result = run(cfg, runs, REPO, resume=True, device="cpu", data_root=str(root))
    run_dir = runs / cfg["run_name"]
    curve = [json.loads(line) for line in (run_dir / "val_log.jsonl").read_text().splitlines()]
    check("resumed curve has steps 2,4, with predicted and oracle metrics at each", [r["step"] for r in curve] == [2, 4]
          and all("predicted" in r and "oracle" in r for r in curve))
    model, ckpt = load_model(run_dir / "model_final.pt")
    check("final checkpoint rebuilds offline with four channels and saved cascade config",
          first_patch_projection(model)[1].in_channels == 4 and ckpt["config"]["cascade"] == cfg["cascade"])
    check("fourth-channel weights learned", torch.count_nonzero(first_patch_projection(model)[1].weight[:, 3]).item() > 0)
    check("stage-1 checkpoint bytes unchanged", hashlib.sha256(stage1_path.read_bytes()).hexdigest() == digest)
    evals = result["summary"]["eval"]
    check("six composed 3-class evals and six panels returned", len(evals) == 6
          and all(m["class_names"] == ["background", "hand", "interacting_object"] for m in evals.values())
          and len([name for name in result["files"] if "_vis/" in name]) == 6)
    check("predicted hand IoU and Boundary IoU match saved stage 1 on every split",
          all(ok for key, ok in result["checks"].items() if "matches_stage1" in key)
          and len([key for key in result["checks"] if "matches_stage1" in key]) == 6)
    again = evaluate_run(run_dir, "val", device="cpu", data_root=str(root))["metrics"]
    check("evaluate_run dispatch reloads four channels and reproduces both modes",
          all(again[mode]["confusion_matrix"] == evals[f"val_{mode}"]["confusion_matrix"] for mode in ("predicted", "oracle")))
    cfg_s1 = copy.deepcopy(cfg)
    cfg_s1.update(run_name="local_cascade_s1", seed=1)
    second = run(cfg_s1, runs, REPO, device="cpu", data_root=str(root))
    check("second stage-2 seed has identical visualisation picks", all(
        second["summary"]["eval"][key]["vis_names"] == m["vis_names"] for key, m in evals.items()))
    text = build_table(runs, ["local_full"], ["local_cascade", "local_cascade_s1"])
    (tmp / "phase5_table.md").write_text(text)
    check("table contains full metrics, three object numbers, means, spreads and oracle diagnostic",
          all(s in text for s in ("BIoU object", "Three object-IoU", "predicted mean", "Observed seed spreads", "Diagnostic:")))
    print(f"Synthetic data and outputs: {tmp}", flush=True)


if __name__ == "__main__":
    main()
