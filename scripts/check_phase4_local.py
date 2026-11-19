"""Local CPU check of the Phase 4 additions on synthetic data (no Modal, no real dataset; downloads nvidia/mit-b0).

    .venv/bin/python scripts/check_phase4_local.py

Checks: the label map (validation; applied before the transform, so objects become background while 2nd-order
pixels and padding stay 255; worker seeding still works through the wrapper), the hand-IoU identity behind the
hands-only vs 3-class comparison, seeded head initialisation, visualisation picks (independent of the run seed,
distinct videos out-of-domain, safe file names), tiny end-to-end runs (hands-only interrupted + resumed with a val
curve, a second seed, head-only keeps the encoder fixed, full fine-tune), and scripts/phase4_table.py on them.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from check_phase3_local import TRAIN_SIZES, VAL_SIZES, make_split  # noqa: E402
from src.config import DEFAULTS, _merge, load_config  # noqa: E402
from src.data import EgoHOSDataset, TrainTransform  # noqa: E402
from src.evaluate import (  # noqa: E402
    confusion_matrix, evaluate_run, iou_from_confusion, pick_vis_names, video_of, vis_file_stem,
)
from src.labels import LabelMapTransform, label_lut, with_label_map  # noqa: E402
from src.model import build_model, load_model  # noqa: E402
from src.train import FixedTransform, run, seed_worker  # noqa: E402

OUT_NAMES = [
    "youtube_how to cook PASTA RISOTTATA  UNDER 30 MINUTE PASTA MADE LIKE RISOTTO  JONAH IN THE KITCHEN-QHjXmwMg3MY-1080p-1644528336886_009780",
    "youtube_how to cook PASTA RISOTTATA  UNDER 30 MINUTE PASTA MADE LIKE RISOTTO  JONAH IN THE KITCHEN-QHjXmwMg3MY-1080p-1644528336886_008700",
    "youtube_Making a Wooden Spoon (1)_000120",
    "youtube_Making a Wooden Spoon (1)_000480",
    "youtube_Fixing a bike chain-abc123-1080p_001500",
]
HANDS_MAP = {2: 0}
results: dict[str, bool] = {}


def check(name: str, ok: bool) -> None:
    results[name] = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def tiny_cfg(root: Path, run_name: str, seed: int = 0, **sections) -> dict:
    base = {
        "run_name": run_name,
        "seed": seed,
        "data": {"train_root": str(root), "num_workers": 2},
        "train": {"max_iters": 4, "batch_size": 2, "warmup_iters": 1, "log_every": 2, "ckpt_every": 2, "bf16": False},
        "eval": {"splits": ["val", "test_indomain", "test_outdomain"], "n_vis": 2, "val_every": 2},
    }
    for key, value in sections.items():
        base[key] = {**base.get(key, {}), **value}
    return _merge(DEFAULTS, base)


def main() -> None:
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="phase4_check_"))
    root = tmp / "egohos"
    make_split(root, "train", TRAIN_SIZES, rng)
    make_split(root, "val", VAL_SIZES, rng)
    make_split(root, "test_indomain", VAL_SIZES, rng)
    make_split(root, "test_outdomain", [(960, 540)] * len(OUT_NAMES), rng, names=OUT_NAMES)
    runs = tmp / "runs"

    # 1. Label map validation and the configs.
    lut = label_lut(HANDS_MAP, 2)
    check("hands-only map sends remapped [0, 1, 2, 255] to [0, 1, 0, 255]", lut[[0, 1, 2, 255]].tolist() == [0, 1, 0, 255])
    check("no map with 3 classes -> no lookup table", label_lut(None, 3) is None)
    check("2 class_names without a label_map is rejected", raises(lambda: label_lut(None, 2)))
    check("map to an id outside the classes ({2: 5}) is rejected", raises(lambda: label_lut({2: 5}, 2)))
    check("map from a non-remapped id ({7: 0}) is rejected", raises(lambda: label_lut({7: 0}, 3)))
    cfgs = {p.stem: load_config(p) for p in sorted((REPO / "configs").glob("*.yaml"))}
    check(f"all configs load: {sorted(cfgs)}", {"b0_head_only", "b0_full_ft", "b0_hands_only", "overfit8", "smoke_dev500"} <= set(cfgs))
    check("b0_hands_only: class_names [background, hand], label_map {2: 0}",
          cfgs["b0_hands_only"]["model"]["class_names"] == ["background", "hand"] and cfgs["b0_hands_only"]["data"]["label_map"] == HANDS_MAP)
    same = ["data", "train", "eval"]
    check("b0_*_s1 configs differ from their seed-0 config only in the seed",
          all({k: cfgs[f"{g}_s1"][k] for k in same + ["model"]} == {k: cfgs[g][k] for k in same + ["model"]}
              and cfgs[f"{g}_s1"]["seed"] == 1 for g in ("b0_full_ft", "b0_hands_only")))

    # 2. The map acts on the remapped mask before the transform.
    plain = EgoHOSDataset("train", root, FixedTransform())
    mapped = EgoHOSDataset("train", root, with_label_map(FixedTransform(), lut))
    ok = True
    for i in range(len(plain)):
        m0, m1 = plain[i][1], mapped[i][1]
        ok &= 2 in m0 and set(np.unique(m1).tolist()) <= {0, 1, 255} and (m1 == 1).any()
        ok &= np.array_equal(m1 == 255, m0 == 255) and np.array_equal(m1 == 0, (m0 == 0) | (m0 == 2))
    check("FixedTransform + map: object -> background, 2nd-order and padding stay 255, hand unchanged", ok)
    aug = EgoHOSDataset("train", root, with_label_map(TrainTransform(seed=0), lut))
    values = set()
    for _ in range(3):
        for i in range(len(aug)):
            values |= set(np.unique(aug[i][1]).tolist())
    check(f"TrainTransform + map: training masks only contain {{0, 1, 255}} (got {sorted(values)})", values <= {0, 1, 255})

    name = plain.names[0]
    ds = EgoHOSDataset("train", root, LabelMapTransform(TrainTransform(), lut), file_list=[name] * 4)
    batches = list(DataLoader(ds, batch_size=1, num_workers=2, worker_init_fn=seed_worker))
    check("seed_worker reaches the augmentation through the label-map wrapper (two workers differ)",
          not torch.equal(batches[0][0], batches[1][0]))

    # 3. Hand IoU of a 3-class model equals hand IoU after merging object into background (2nd-order ignored),
    #    but not if 2nd-order pixels became background.
    g = torch.Generator().manual_seed(0)
    gt = torch.tensor([0, 1, 2, 255])[torch.randint(0, 4, (200, 300), generator=g)]
    pred = torch.randint(0, 3, (200, 300), generator=g)
    lut_t = torch.from_numpy(lut.astype(np.int64))
    hand3 = iou_from_confusion(confusion_matrix(pred, gt, 3).numpy())[1]
    hand2 = iou_from_confusion(confusion_matrix(lut_t[pred], lut_t[gt], 2).numpy())[1]
    lut_bg = torch.from_numpy(label_lut({2: 0, 255: 0}, 2).astype(np.int64))
    hand_bg = iou_from_confusion(confusion_matrix(lut_t[pred], lut_bg[gt], 2).numpy())[1]
    check(f"hand IoU identity: 3-class {hand3:.4f} == hands-only {hand2:.4f} (2nd-order as background: {hand_bg:.4f})",
          abs(hand3 - hand2) < 1e-12 and abs(hand_bg - hand3) > 1e-3)

    # 4. Seeded head initialisation.
    heads = []
    for s in (0, 0, 1):
        torch.manual_seed(s)
        heads.append(build_model(["background", "hand"]).decode_head.classifier.weight.detach().clone())
    check("head init is reproducible from the seed (same seed equal, other seed different)",
          torch.equal(heads[0], heads[1]) and not torch.equal(heads[0], heads[2]))

    # 5. Visualisation picks and file names.
    out_ds = EgoHOSDataset("test_outdomain", root)
    picks = pick_vis_names(out_ds, 3)
    check(f"test_outdomain picks come from 3 different videos: {[video_of(p)[-20:] for p in picks]}",
          len({video_of(p) for p in picks}) == 3 and picks == pick_vis_names(out_ds, 3))
    stems = [vis_file_stem(n) for n in OUT_NAMES]
    check("out-of-domain vis file names: no spaces/brackets, <= 80 chars, frame number kept",
          all(" " not in s and "(" not in s and len(s) <= 80 and s.endswith(n[-6:]) for s, n in zip(stems, OUT_NAMES)))
    phase3_name = "ego4d_6f93978d-a3d0-467d-b401-e23b3c92f3bf_40800"
    check("Phase 3 style names keep their file name", vis_file_stem(phase3_name) == phase3_name)

    # 6. Hands-only end to end, interrupted at the first checkpoint, then resumed.
    cfg = tiny_cfg(root, "local_hands_only", data={"label_map": HANDS_MAP}, model={"class_names": ["background", "hand"]})

    class Interrupt(Exception):
        pass

    def commit_then_crash():
        if (runs / "local_hands_only" / "checkpoint_last.pt").exists():
            raise Interrupt

    check("hands-only run interrupted at the first checkpoint",
          raises(lambda: run(cfg, runs, REPO, device="cpu", commit=commit_then_crash, data_root=str(root)), Interrupt))
    out = run(cfg, runs, REPO, resume=True, device="cpu", data_root=str(root))
    run_dir = runs / "local_hands_only"
    val_steps = [json.loads(line)["step"] for line in (run_dir / "val_log.jsonl").read_text().splitlines()]
    check(f"resumed val curve steps {val_steps} == [2, 4] (step 4 = the final val eval)", val_steps == [2, 4])
    check("val_log.jsonl and val_curve.png returned", {"val_log.jsonl", "val_curve.png"} <= set(out["files"]))
    val = json.loads((run_dir / "eval" / "val.json").read_text())
    check("hands-only eval: classes [background, hand], 2x2 confusion, Boundary IoU for hand only, label_map saved",
          val["class_names"] == ["background", "hand"] and np.array(val["confusion_matrix"]).shape == (2, 2)
          and list(val["boundary_iou"]) == ["hand"] and val["label_map"] == {"2": 0})
    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    _, ckpt = load_model(run_dir / "model_final.pt")
    check("label_map stored in config.yaml and in model_final.pt",
          saved["data"]["label_map"] == HANDS_MAP and ckpt["config"]["data"]["label_map"] == HANDS_MAP)
    again = evaluate_run(run_dir, "test_outdomain", device="cpu", data_root=str(root))["metrics"]
    first = out["summary"]["eval"]["test_outdomain"]
    # Only JSON artifacts stringify label-map keys; the Python return value keeps integer keys.
    check(f"evaluate_run on model_final.pt reproduces test_outdomain mIoU ({again['miou']:.4f}) with the saved map",
          abs(again["miou"] - first["miou"]) < 1e-6 and again["label_map"] == HANDS_MAP)
    vis = [f for f in out["files"] if "_vis/" in f]
    check(f"{len(vis)} panels (2 per split), file names without spaces",
          len(vis) == 6 and all(" " not in f for f in vis))
    check("test_outdomain panels show 2 different videos", len({video_of(n) for n in first["vis_names"]}) == 2)

    # 7. Second seed: different training, same visualisation picks.
    cfg_s1 = tiny_cfg(root, "local_hands_only_s1", seed=1, data={"label_map": HANDS_MAP}, model={"class_names": ["background", "hand"]})
    out_s1 = run(cfg_s1, runs, REPO, device="cpu", data_root=str(root))
    check("seed 1 shows the same images as seed 0 on every split",
          all(out_s1["summary"]["eval"][s]["vis_names"] == m["vis_names"] for s, m in out["summary"]["eval"].items()))
    check("seed 1 trains differently (different final loss)",
          out_s1["summary"]["train"]["last_loss"] != out["summary"]["train"]["last_loss"])

    # 8. Head-only: the encoder never changes, only head parameters are trainable.
    out_a = run(tiny_cfg(root, "local_head_only", model={"freeze_encoder": True}), runs, REPO, device="cpu", data_root=str(root))
    trained, _ = load_model(runs / "local_head_only" / "model_final.pt")
    torch.manual_seed(0)
    fresh = build_model(DEFAULTS["model"]["class_names"])
    enc_same = all(torch.equal(a, b) for a, b in zip(trained.segformer.state_dict().values(), fresh.segformer.state_dict().values()))
    head_moved = not torch.equal(trained.decode_head.classifier.weight, fresh.decode_head.classifier.weight)
    t = out_a["summary"]["train"]
    log = [json.loads(line) for line in (runs / "local_head_only" / "train_log.jsonl").read_text().splitlines()]
    check(f"head-only: encoder weights unchanged, head trained, {t['n_trainable_M']}M of {t['n_params_M']}M trainable, no encoder lr",
          enc_same and head_moved and t["n_trainable_M"] < t["n_params_M"] and all(r["lr_encoder"] is None for r in log))

    # 9. Full fine-tune (two seeds), then the table over all five runs.
    for name, seed in (("local_full_ft", 0), ("local_full_ft_s1", 1)):
        run(tiny_cfg(root, name, seed=seed), runs, REPO, device="cpu", data_root=str(root))
    table = tmp / "phase4_table.md"
    names = "local_head_only,local_full_ft,local_full_ft_s1,local_hands_only,local_hands_only_s1"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "phase4_table.py"), "--runs", names, "--runs-dir", str(runs), "--out", str(table)],
        capture_output=True, text=True,
    )
    text = table.read_text() if table.exists() else ""
    sections = ["## Runs", "## Results per split", "## Seed noise", "## Stage-1 question", "## Domain gap",
                "## Reference: EgoHOS paper", "## Automatic consistency checks"]
    check(f"phase4_table.py runs on the tiny runs and writes every section ({table})",
          proc.returncode == 0 and all(s in text for s in sections) and "local_full_ft mean" in text)
    if proc.returncode != 0:
        print(proc.stderr)

    print(f"\n{sum(results.values())}/{len(results)} checks passed. Synthetic data and outputs in {tmp}")


if __name__ == "__main__":
    main()
