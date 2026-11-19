"""Local CPU check of the Phase 3 code on synthetic data (no Modal, no real dataset; downloads nvidia/mit-b0).

    .venv/bin/python scripts/check_phase3_local.py

Checks: per-worker augmentation seeding, FixedTransform output size, Boundary IoU against the official cv2
implementation, IoU arithmetic, and a tiny end-to-end run (train -> checkpoint -> interrupted -> resume -> eval
-> reload model_final.pt gives the same metrics).
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DEFAULTS, _merge  # noqa: E402
from src.data import EgoHOSDataset, TrainTransform  # noqa: E402
from src.evaluate import boundary_width, confusion_matrix, evaluate_run, iou_from_confusion, mask_boundary  # noqa: E402
from src.train import FIXED_PAD, FixedTransform, run, seed_worker  # noqa: E402

TRAIN_SIZES = [(683, 512), (456, 256), (910, 512), (640, 480), (342, 256), (683, 512)]
VAL_SIZES = [(1440, 1080), (456, 256), (1920, 1080), (640, 480)]
NAMES = ["ego4d_a", "epic_b", "thu_c", "escape_d", "ego4d_e", "epic_f"]
results: dict[str, bool] = {}


def check(name: str, ok: bool) -> None:
    results[name] = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def make_split(root: Path, split: str, sizes, rng, names: list[str] | None = None) -> None:
    (root / split / "image").mkdir(parents=True)
    (root / split / "label").mkdir(parents=True)
    for i, (w, h) in enumerate(sizes):
        label = np.zeros((h, w), np.uint8)
        cv2.ellipse(label, (w // 3, h // 2), (w // 8, h // 5), 0, 0, 360, 1, -1)
        cv2.ellipse(label, (2 * w // 3, h // 2), (w // 10, h // 6), 0, 0, 360, 3, -1)
        cv2.rectangle(label, (0, 0), (w // 10, h // 10), 7, -1)
        colours = np.array([[60, 60, 60], [200, 120, 100], [60, 200, 60]] + [[0, 0, 0]] * 4 + [[90, 90, 200]], np.float32)
        image = colours[np.minimum(label, 7)] + rng.normal(0, 20, (h, w, 3))
        name = names[i] if names else NAMES[i % len(NAMES)] + f"_{split}{i}"
        cv2.imwrite(str(root / split / "image" / f"{name}.jpg"), np.clip(image, 0, 255).astype(np.uint8)[..., ::-1])
        cv2.imwrite(str(root / split / "label" / f"{name}.png"), label)


def official_boundary(mask: np.ndarray, d: int) -> np.ndarray:
    """boundary-iou-api mask_to_boundary with a fixed dilation d."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=d)[1 : h + 1, 1 : w + 1]
    return (mask - eroded).astype(bool)


def main() -> None:
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="phase3_check_"))
    root = tmp / "egohos"
    make_split(root, "train", TRAIN_SIZES, rng)
    make_split(root, "val", VAL_SIZES, rng)

    # 1. Worker seeding: the same image in two workers must get different augmentations.
    name = EgoHOSDataset("train", root).names[0]
    for fn, label in ((None, "without"), (seed_worker, "with")):
        ds = EgoHOSDataset("train", root, TrainTransform(), file_list=[name] * 4)
        batches = list(DataLoader(ds, batch_size=1, num_workers=2, worker_init_fn=fn))
        same = torch.equal(batches[0][0], batches[1][0])
        if fn is None:
            check("bug_reproduced: without seed_worker, two workers draw identical augmentations", same)
        else:
            check("with seed_worker, two workers draw different augmentations", not same)

    # 2. FixedTransform: every stored size ends up at the common pad size, padding is ignore.
    ds = EgoHOSDataset("train", root, FixedTransform())
    shapes = {tuple(ds[i][1].shape) for i in range(len(ds))}
    check(f"FixedTransform output is {FIXED_PAD} for sizes {TRAIN_SIZES}", shapes == {FIXED_PAD})

    # 3. Boundary band matches the official implementation exactly (including the image-border rule).
    ok = True
    for d in (1, 3, 11, 36):
        m = np.zeros((300, 400), np.uint8)
        cv2.circle(m, (200, 150), 90, 1, -1)
        cv2.rectangle(m, (0, 0), (60, 300), 1, -1)
        m |= (rng.random((300, 400)) < 0.01).astype(np.uint8)
        ours = mask_boundary(torch.from_numpy(m.astype(bool)), d).numpy()
        ok &= np.array_equal(ours, official_boundary(m, d))
    check("mask_boundary equals official boundary-iou-api for d in 1, 3, 11, 36", ok)
    check("boundary width: 2% of diagonal = 10 px (456x256), 36 px (1440x1080), 88 px (3840x2160)",
          [boundary_width(h, w, 0.02) for w, h in ((456, 256), (1440, 1080), (3840, 2160))] == [10, 36, 88])

    # 4. IoU arithmetic: ignore excluded, known values.
    gt = torch.tensor([[0, 1, 1, 255], [2, 2, 0, 255]])
    pred = torch.tensor([[0, 1, 2, 1], [2, 2, 0, 2]])
    iou = iou_from_confusion(confusion_matrix(pred, gt, 3).numpy())
    check(f"IoU on a known 2x4 example = [1, 0.5, 0.667]: got {[round(v, 3) for v in iou]}", np.allclose(iou, [1, 0.5, 2 / 3]))

    # 5. Tiny end-to-end run on CPU, interrupted after the first checkpoint, then resumed.
    cfg = _merge(DEFAULTS, {
        "run_name": "local_check",
        "data": {"train_root": str(root), "num_workers": 2},
        "train": {"max_iters": 6, "batch_size": 2, "warmup_iters": 2, "log_every": 2, "ckpt_every": 4, "bf16": False},
        "eval": {"splits": ["val", "train"], "n_vis": 2},
    })
    runs = tmp / "runs"

    class Interrupt(Exception):
        pass

    def commit_then_crash():
        if (runs / "local_check" / "checkpoint_last.pt").exists():
            raise Interrupt

    try:
        run(cfg, runs, REPO, device="cpu", commit=commit_then_crash, data_root=str(root))
        check("interrupted at the first checkpoint", False)
    except Interrupt:
        check("interrupted at the first checkpoint (checkpoint_last.pt written)", True)
    try:
        run(cfg, runs, REPO, device="cpu", data_root=str(root))
        check("rerun without --overwrite refused", False)
    except FileExistsError:
        check("rerun without --overwrite refused", True)

    out = run(cfg, runs, REPO, resume=True, device="cpu", data_root=str(root))
    run_dir = runs / "local_check"
    steps = [json.loads(line)["step"] for line in (run_dir / "train_log.jsonl").read_text().splitlines()]
    check(f"resumed run logged steps {steps} == [2, 4, 6]", steps == [2, 4, 6])
    check("checkpoint_last.pt deleted, model_final.pt written",
          not (run_dir / "checkpoint_last.pt").exists() and (run_dir / "model_final.pt").exists())
    expected = {"eval/val.json", "eval/train.json", "loss_curve.png", "train_log.jsonl", "summary.json", "config.yaml"}
    check("output files present", expected <= set(out["files"]) and len([f for f in out["files"] if "_vis/" in f]) == 4)
    check(f"checks returned by run(): {out['checks']}", "val_ious_in_0_1" in out["checks"])

    again = evaluate_run(run_dir, "val", device="cpu", data_root=str(root))["metrics"]
    check(f"reloaded model_final.pt reproduces val mIoU ({again['miou']:.4f})",
          abs(again["miou"] - out["summary"]["eval"]["val"]["miou"]) < 1e-6)

    print(f"\n{sum(results.values())}/{len(results)} checks passed. Synthetic data and outputs in {tmp}")


if __name__ == "__main__":
    main()
