"""Phase 4 results table, built only from local run outputs (no GPU, no torch):
results/runs/<run>/{config.yaml, summary.json, eval/<split>.json} -> results/phase4_table.md.

    .venv/bin/python scripts/phase4_table.py [--runs a,b,...] [--runs-dir results/runs] [--out results/phase4_table.md]

Runs missing locally (or splits not evaluated yet) show as "-", so it can be rebuilt at any time.
All numbers are IoU x 100. Runs named <group>_s<k> are extra seeds of <group>; groups get a mean row.
"""

import argparse
import json
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNS = ["b0_head_only", "b0_full_ft", "b0_full_ft_s1", "b0_hands_only", "b0_hands_only_s1"]
SPLITS = {"val": "val", "test_indomain": "test in-domain", "test_outdomain": "test out-of-domain (YouTube)"}
SHORT = {"background": "bg", "hand": "hand", "interacting_object": "object"}
FULL, HEAD, HANDS = "full fine-tune", "head-only", "hands-only"

# EgoHOS paper (arXiv 2208.03826), evaluated on the YouTube test set (= our test_outdomain), best checkpoint on val.
PAPER_T5_PARALLEL = {  # Table 5 row "Para. Decode": one network, 5 classes decoded in parallel (per-class IoU)
    "left hand": 69.08, "right hand": 73.50,
    "left-hand object": 48.67, "right-hand object": 36.21, "two-hand object": 37.46,
}
PAPER_T3_BINARY_HAND = 83.18  # Table 3 "Ours", ResNet-18 + HRNet, binary hand "mIoU" (whether bg is averaged in: not stated)
PAPER_HAND = (PAPER_T5_PARALLEL["left hand"] + PAPER_T5_PARALLEL["right hand"]) / 2
PAPER_OBJECT = sum(v for k, v in PAPER_T5_PARALLEL.items() if "object" in k) / 3
BALLPARK = 15.0  # "same ballpark" = within this many IoU points


def load_run(runs_dir: Path, name: str) -> dict | None:
    d = runs_dir / name
    if not (d / "config.yaml").exists():
        return None
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
    evals = {}
    for split in SPLITS:
        path = d / "eval" / f"{split}.json"
        if path.exists():
            m = json.loads(path.read_text())
            m.pop("per_image", None)
            evals[split] = m
    if cfg["model"]["freeze_encoder"]:
        role = HEAD
    elif len(cfg["model"]["class_names"]) == 2:
        role = HANDS
    else:
        role = FULL
    return {"name": name, "cfg": cfg, "summary": summary, "eval": evals, "role": role,
            "group": re.sub(r"_s\d+$", "", name), "seed": cfg["seed"]}


def metric(run: dict, split: str, key: str) -> float | None:
    """key: miou | miou_fg | iou:<class> | biou:<class>."""
    m = run["eval"].get(split)
    if m is None:
        return None
    if ":" in key:
        kind, cls = key.split(":")
        return m["iou" if kind == "iou" else "boundary_iou"].get(cls)
    return m[key]


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals and len(vals) == len(values) else None


def group_metric(runs: list[dict], split: str, key: str) -> float | None:
    return mean([metric(r, split, key) for r in runs])


def pts(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def signed(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:+.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def build(runs: list[dict]) -> tuple[str, dict[str, bool]]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    by_role = {}
    for g, rs in groups.items():
        by_role.setdefault(rs[0]["role"], (g, rs))

    out = ["# Phase 4: single-stage SegFormer-B0 on the full EgoHOS training set", ""]
    out += [
        "Built by `scripts/phase4_table.py` from `results/runs/<run>/eval/<split>.json`. All numbers are IoU x 100.",
        "Protocol (Phase 2): input shorter side 512 px, logits upsampled bilinearly to the original resolution, scored",
        "against the original label; 255 (2nd-order objects) excluded. Dataset-level confusion matrix. Boundary IoU",
        "(BIoU) with d = 2% of the image diagonal. The final model of each run is reported (no checkpoint selection).",
        "",
    ]

    # Runs
    out += ["## Runs", ""]
    rows = []
    for r in runs:
        t = (r["summary"] or {}).get("train", {})
        c, m = r["cfg"], r["cfg"]["model"]
        rows.append([
            f"`{r['name']}`", r["role"], ", ".join(SHORT.get(x, x) for x in m["class_names"]),
            "frozen" if m["freeze_encoder"] else "trained", str(r["seed"]), str(c["train"]["max_iters"]),
            f"{t['train_seconds'] / 60:.1f}" if t else "-", str(t.get("imgs_per_s", "-")),
            f"{t['data_wait_frac']:.1%}" if t else "-", f"{t['n_trainable_M']}M" if t else "-",
        ])
    out += table(["run", "model", "classes", "encoder", "seed", "iterations", "train min", "img/s", "data wait", "trainable"], rows)
    out += ["", "Hands-only maps interacting object -> background (`label_map: {2: 0}`); 2nd-order objects stay ignored,",
            "so its hand IoU is scored on exactly the same pixels as the 3-class runs.", ""]

    # Per split
    cols = ["iou:background", "iou:hand", "iou:interacting_object", "miou", "miou_fg", "biou:hand", "biou:interacting_object"]
    header = ["run", "IoU bg", "IoU hand", "IoU object", "mIoU", "mIoU_fg", "BIoU hand", "BIoU object"]
    out += ["## Results per split", ""]
    for split, title in SPLITS.items():
        n = next((r["eval"][split]["n_images"] for r in runs if split in r["eval"]), None)
        out += [f"### {title}" + (f" ({n:,} images)" if n else ""), ""]
        rows = []
        for g, rs in groups.items():
            rows += [[f"`{r['name']}`"] + [pts(metric(r, split, k)) for k in cols] for r in rs]
            if len(rs) > 1:
                seeds = ", ".join(str(r["seed"]) for r in rs)
                rows.append([f"**{g} mean (seeds {seeds})**"] + [f"**{pts(group_metric(rs, split, k))}**" for k in cols])
        out += table(header, rows) + [""]
    out += ["mIoU and mIoU_fg average over the run's own classes: for hands-only that is (bg, hand) and hand only,",
            "so compare hands-only through its hand columns, not its mIoU.", ""]

    # Seed noise
    multi = {g: rs for g, rs in groups.items() if len(rs) > 1}
    if multi:
        out += ["## Seed noise", ""]
        rows, worst = [], 0.0
        for g, rs in multi.items():
            for split in SPLITS:
                for key, label in (("iou:hand", "IoU hand"), ("iou:interacting_object", "IoU object"), ("miou_fg", "mIoU_fg"),
                                   ("biou:hand", "BIoU hand"), ("biou:interacting_object", "BIoU object")):
                    vals = [metric(r, split, key) for r in rs]
                    if any(v is None for v in vals) or (key == "miou_fg" and rs[0]["role"] == HANDS):
                        continue
                    spread = max(vals) - min(vals)
                    worst = max(worst, spread)
                    rows.append([g, split, label] + [pts(v) for v in vals] + [pts(spread)])
        n_seeds = max(len(rs) for rs in multi.values())
        out += table(["group", "split", "metric"] + [f"seed {i}" for i in range(n_seeds)] + ["max - min"], rows)
        out += ["", f"Largest seed spread: {100 * worst:.1f} points. Differences between models smaller than this are noise.", ""]

    # Hands-only vs full fine-tune
    checks: dict[str, bool] = {}
    if FULL in by_role and HANDS in by_role:
        (gb, rb), (gc, rc) = by_role[FULL], by_role[HANDS]
        out += [f"## Stage-1 question: hand from the hands-only model ({gc}) vs the 3-class model ({gb})", ""]

        def cell(rs, split, key):
            vals = [metric(r, split, key) for r in rs]
            per_seed = " / ".join(pts(v) for v in vals)
            return pts(mean(vals)) + (f" ({per_seed})" if len(rs) > 1 else "")

        rows = []
        for split in SPLITS:
            row = [split]
            for key in ("iou:hand", "biou:hand"):
                b, c = group_metric(rb, split, key), group_metric(rc, split, key)
                row += [cell(rb, split, key), cell(rc, split, key), signed(None if b is None or c is None else c - b)]
            rows.append(row)
        out += table(["split", f"IoU hand {gb}", f"IoU hand {gc}", "diff (c - b)",
                      f"BIoU hand {gb}", f"BIoU hand {gc}", "diff (c - b)"], rows)
        out += ["", "Means over seeds, per-seed values in brackets. Compare the difference with the seed spread above.", ""]

    # Domain gap
    out += ["## Domain gap: test in-domain -> test out-of-domain", ""]
    gap_keys = [("iou:hand", "hand"), ("iou:interacting_object", "object"), ("miou_fg", "mIoU_fg"),
                ("biou:hand", "BIoU hand"), ("biou:interacting_object", "BIoU object")]
    header = ["run"] + [f"{label} in / out / drop" for _, label in gap_keys]
    rows = []
    for g, rs in groups.items():
        entries = [(f"`{r['name']}`", [r]) for r in rs] + ([(f"**{g} mean**", rs)] if len(rs) > 1 else [])
        for label, sel in entries:
            row = [label]
            for key, _ in gap_keys:
                a, b = group_metric(sel, "test_indomain", key), group_metric(sel, "test_outdomain", key)
                if a is None or b is None or (key == "miou_fg" and sel[0]["role"] == HANDS):
                    row.append("-")
                else:
                    row.append(f"{pts(a)} / {pts(b)} / {100 * (a - b):.1f} ({(a - b) / a:.0%})" if a else "-")
            rows.append(row)
    out += table(header, rows)
    out += ["", "drop = in-domain - out-of-domain in IoU points (relative drop in brackets).", ""]

    # EgoHOS paper reference
    out += ["## Reference: EgoHOS paper (arXiv 2208.03826)", ""]
    out += [
        "The paper reports on the YouTube test set (our test_outdomain) unless stated otherwise, with the best",
        "checkpoint chosen on val. Its labels keep left/right hands and left/right/two-hand objects, and its models",
        "are much larger (ResNet-18 + HRNet, Swin-L + UperNet), so the numbers are a ballpark, not a comparison.",
        "",
    ]
    rows = [[f"Table 5, Para. Decode: {k}", f"{v:.1f}"] for k, v in PAPER_T5_PARALLEL.items()]
    rows += [["Table 5, Para. Decode: mean of the two hands", f"{PAPER_HAND:.1f}"],
             ["Table 5, Para. Decode: mean of the three objects", f"{PAPER_OBJECT:.1f}"],
             ["Table 3, binary hand, ResNet-18 + HRNet (\"mIoU\"; unclear if bg is averaged in)", f"{PAPER_T3_BINARY_HAND:.1f}"]]
    for role in (FULL, HANDS, HEAD):
        if role in by_role:
            g, rs = by_role[role]
            rows.append([f"ours, `{g}` ({role}), test_outdomain IoU hand", pts(group_metric(rs, "test_outdomain", "iou:hand"))])
            if role != HANDS:
                rows.append([f"ours, `{g}` ({role}), test_outdomain IoU object",
                             pts(group_metric(rs, "test_outdomain", "iou:interacting_object"))])
    out += table(["source", "IoU"], rows)
    out += ["", "Our classes merge left and right (easier: no left/right confusion) and our model is B0 with 3.7M",
            "parameters (harder). Ours is one merged-class IoU; the paper means above average per-class IoUs.", ""]

    # Consistency checks
    def check(name: str, ok: bool | None) -> None:
        if ok is not None:
            checks[name] = bool(ok)

    def gm(role, split, key):
        return group_metric(by_role[role][1], split, key) if role in by_role else None

    for split in SPLITS:
        for key, label in (("iou:hand", "hand IoU"), ("iou:interacting_object", "object IoU"), ("miou_fg", "mIoU_fg")):
            a, b = gm(HEAD, split, key), gm(FULL, split, key)
            check(f"{split}: full fine-tune > head-only on {label}", None if a is None or b is None else b > a)
    for role in (FULL, HEAD):
        for split in SPLITS:
            h, o = gm(role, split, "iou:hand"), gm(role, split, "iou:interacting_object")
            check(f"{split}: {role} hand IoU > object IoU", None if h is None or o is None else h > o)
    for role in (FULL, HEAD, HANDS):
        for key, label in (("iou:hand", "hand IoU"), ("iou:interacting_object", "object IoU")):
            if role == HANDS and "object" in label:
                continue
            a, b = gm(role, "test_indomain", key), gm(role, "test_outdomain", key)
            check(f"{role}: out-of-domain {label} < in-domain", None if a is None or b is None else b < a)
    h, o = gm(FULL, "test_outdomain", "iou:hand"), gm(FULL, "test_outdomain", "iou:interacting_object")
    check(f"full fine-tune out-of-domain hand IoU within {BALLPARK:.0f} points of the paper's {PAPER_HAND:.1f}",
          None if h is None else abs(100 * h - PAPER_HAND) <= BALLPARK)
    check(f"full fine-tune out-of-domain object IoU within {BALLPARK:.0f} points of the paper's {PAPER_OBJECT:.1f}",
          None if o is None else abs(100 * o - PAPER_OBJECT) <= BALLPARK)

    out += ["## Automatic consistency checks", "",
            "Group means. These are sanity checks, not the verification (look at the numbers and the overlays).", ""]
    out += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()] + [""]
    return "\n".join(out), checks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=",".join(RUNS))
    p.add_argument("--runs-dir", default=str(REPO / "results" / "runs"))
    p.add_argument("--out", default=str(REPO / "results" / "phase4_table.md"))
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    runs, missing = [], []
    for name in [n.strip() for n in args.runs.split(",") if n.strip()]:
        run = load_run(runs_dir, name)
        (runs if run else missing).append(run or name)
    if not runs:
        raise SystemExit(f"No runs found in {runs_dir} (looked for {args.runs})")
    text, checks = build(runs)
    Path(args.out).write_text(text)

    for r in runs:
        print(f"{r['name']:<20} {r['role']:<15} splits: {', '.join(r['eval']) or 'none'}")
    for name in missing:
        print(f"{name:<20} not found in {runs_dir}")
    print()
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
