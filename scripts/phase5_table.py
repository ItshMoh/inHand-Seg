"""Build the Phase-5 comparison from existing local JSONs; no inference, torch or prediction cache."""

import argparse
import json
from pathlib import Path

from phase4_table import SPLITS, mean, pts, signed, table

REPO = Path(__file__).resolve().parents[1]
KEYS = [("iou", "background"), ("iou", "hand"), ("iou", "interacting_object"),
        (None, "miou"), (None, "miou_fg"), ("boundary_iou", "hand"), ("boundary_iou", "interacting_object")]


def read_metrics(runs_dir, run, split, mode=None):
    suffix = f"_{mode}" if mode else ""
    path = runs_dir / run / "eval" / f"{split}{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; fetch the completed run first")
    return json.loads(path.read_text())


def value(metrics, key):
    section, name = key
    return metrics[section][name] if section else metrics[name]


def build(runs_dir, baseline_runs, cascade_runs):
    out = ["# Phase 5: hand-guided two-stage cascade", "",
           "IoU × 100. Same Phase-2 original-resolution scoring, 255 excluded, Boundary IoU d=2% of diagonal.",
           "Final weights, no checkpoint selection. Baseline JSONs are reused without inference.",
           "Stage 2: ImageNet MiT-B0, RGB + binary hand input, fresh binary object head; hands are background in its loss.",
           "Oracle uses GT for both the input channel and pasted hands. All cascade seeds share frozen seed-0 stage 1.",
           "This compares complete systems; task/head and compute also differ from the single-stage baseline.", ""]
    loaded = {}
    for split, title in SPLITS.items():
        groups = {
            "single-stage": [(run, read_metrics(runs_dir, run, split)) for run in baseline_runs],
            "predicted": [(run, read_metrics(runs_dir, run, split, "predicted")) for run in cascade_runs],
            "oracle": [(run, read_metrics(runs_dir, run, split, "oracle")) for run in cascade_runs],
        }
        loaded[split] = groups
        rows = []
        for label, entries in groups.items():
            for run, metrics in entries:
                rows.append([f"{run} ({label})"] + [pts(value(metrics, key)) for key in KEYS])
            if len(entries) > 1:
                rows.append([f"**{label} mean**"] + [f"**{pts(mean([value(m, key) for _, m in entries]))}**" for key in KEYS])
        out += [f"## {title}", ""] + table(
            ["run", "IoU bg", "IoU hand", "IoU object", "mIoU", "mIoU_fg", "BIoU hand", "BIoU object"], rows
        ) + [""]
    out += ["## Three object-IoU numbers and differences", ""]
    rows = []
    for split, groups in loaded.items():
        averages = {label: mean([m["iou"]["interacting_object"] for _, m in entries]) for label, entries in groups.items()}
        b, p, o = (averages[label] for label in ("single-stage", "predicted", "oracle"))
        rows.append([split, pts(b), pts(p), pts(o), signed(p - b), signed(o - p)])
    out += table(["split", "single-stage", "predicted cascade", "oracle cascade", "predicted − baseline", "oracle − predicted"], rows)
    out += ["", "## Observed seed spreads", ""]
    rows = []
    for split, groups in loaded.items():
        for label, entries in groups.items():
            vals = [m["iou"]["interacting_object"] for _, m in entries]
            rows.append([split, label, str(len(vals)), pts(min(vals)), pts(max(vals)), pts(max(vals) - min(vals))])
    out += table(["split", "system", "seeds", "min object IoU", "max", "spread"], rows)
    out += ["", "Two seeds show observed variability; their spread is not a bound on future seed noise.",
            "Compare systems within each split. Higher YouTube scores are not evidence of a domain-gap improvement.", "",
            "## Correctness checks and oracle diagnostic", ""]
    for split, groups in loaded.items():
        for run, predicted in groups["predicted"]:
            cfg_path = runs_dir / run / "config.yaml"
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text())
            stage1_name = Path(cfg["cascade"]["stage1_checkpoint"]).parent.name
            stage1 = read_metrics(runs_dir, stage1_name, split)
            for section in ("iou", "boundary_iou"):
                ok = predicted[section]["hand"] == stage1[section]["hand"]
                out.append(f"- {'PASS' if ok else 'FAIL'}: {split}, {run}: hand {section} equals {stage1_name} exactly.")
        for run, oracle in groups["oracle"]:
            ok = oracle["iou"]["hand"] == 1.0
            out.append(f"- {'PASS' if ok else 'FAIL'}: {split}, {run}: oracle hand IoU is 100%.")
            predicted = dict(groups["predicted"])[run]
            delta = oracle["iou"]["interacting_object"] - predicted["iou"]["interacting_object"]
            out.append(f"- Diagnostic: {split}, {run}: oracle − predicted object IoU {signed(delta)} points.")
    out += ["", "Oracle ≥ predicted object IoU is an expectation, not a mathematical guarantee for a fixed model.",
            "A reversal needs investigation; it is not an automatic correctness failure.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=REPO / "results/runs")
    parser.add_argument("--baseline-runs", default="b0_full_ft,b0_full_ft_s1")
    parser.add_argument("--cascade-runs", default="b0_cascade_stage2,b0_cascade_stage2_s1")
    parser.add_argument("--out", type=Path, default=REPO / "results/phase5_table.md")
    args = parser.parse_args()
    text = build(args.runs_dir, args.baseline_runs.split(","), args.cascade_runs.split(","))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
