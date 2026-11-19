"""Rebuild Phase-6 curves/counts from fetched compact measurements; no SAM or torch.

    .venv/bin/python scripts/phase6_report.py
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.clutter_sam import aggregate, atomic_bytes, file_sha256, fingerprint, read_record, validate_coverage


def load_measurements(root):
    manifest, summary = read_record(root / "manifest.json"), read_record(root / "summary.json")
    if (manifest["analysis_fingerprint"] != summary["analysis_fingerprint"]
            or file_sha256(root / "measurements.jsonl.gz") != summary["measurements_sha256"]):
        raise ValueError("Manifest, summary or measurements identity mismatch")
    with gzip.open(root / "measurements.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    for row in records:
        payload = {k: v for k, v in row.items() if k != "payload_sha256"}
        if row.get("payload_sha256") != fingerprint(payload) or row.get("analysis_fingerprint") != manifest["analysis_fingerprint"]:
            raise ValueError("Invalid/incompatible per-image measurement")
    validate_coverage(manifest["inventory"], records)
    return manifest, summary, records


def draw_plot(splits, cfg, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1.2]}, squeeze=False)
    colours = {"baseline": "#3266b2", "cascade": "#d36320"}
    labels = {"baseline": "Single-stage", "cascade": "Predicted cascade"}
    observed = [m["rate"] for data in splits.values() for b in data["bins"]
                for m in b["models"].values() if m["rate"] is not None]
    ymax = min(1.0, max(0.01, max(observed, default=0) * 1.2))
    for col, (split, data) in enumerate(splits.items()):
        ax, denominator = axes[0, col], axes[1, col]
        bins = data["bins"]
        xx = np.arange(len(bins))
        for kind in ("baseline", "cascade"):
            for model in (m for m in cfg["models"] if m["kind"] == kind):
                yy = [b["models"][model["name"]]["rate"] for b in bins]
                ax.plot(xx, [np.nan if y is None else y for y in yy], color=colours[kind], alpha=0.45,
                        linewidth=1, linestyle=":" if model["seed"] else "--", label=f"{labels[kind]} seed {model['seed']}")
            yy = [b["systems"][kind]["mean_rate"] for b in bins]
            ax.plot(xx, [np.nan if y is None else y for y in yy], color=colours[kind], marker="o", linewidth=2.5,
                    label=f"{labels[kind]} mean (2 seeds)")
        ax.set_title(f"{split}\n{data['n_images']:,} images; {data['n_candidates']:,} retained regions; {data['n_no_hand_candidates']:,} without hand distance")
        ax.set_ylim(-0.02 * ymax, 1.02 * ymax)
        ax.set_ylabel(f"Region false-positive rate (object fraction > {cfg['fp_threshold']:g})")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks(xx)
        ax.set_xticklabels([])
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="best")
        denominator.bar(xx, [b["n_candidates"] for b in bins], color="#b7c2d1", width=0.7)
        for x, b in zip(xx, bins):
            denominator.annotate(f"{b['n_candidates']:,}\n({b['n_images']:,} img)", (x, b["n_candidates"]),
                                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        denominator.set_ylim(0, max([b["n_candidates"] for b in bins] + [1]) * 1.55)
        denominator.set_ylabel("Regions\n(shared denominator)")
        intervals = [f"[{b['lower']:g}, {b['upper']:g}{']' if i == len(bins)-1 else ')'}" for i, b in enumerate(bins)]
        denominator.set_xticks(xx, intervals, rotation=20, ha="right", fontsize=9)
        denominator.set_xlabel("Minimum GT-hand distance / original image diagonal (fixed bins)")
    fig.suptitle("Clutter candidate false positives versus hand distance", fontsize=15)
    fig.text(0.5, 0.012, "Candidates are SAM proposals after GT filtering, clipping and mask NMS. Empty bins are undefined (gaps). Seed lines are descriptive; no confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _percent(value):
    return "undefined" if value is None else f"{100 * value:.2f}%"


def observed_findings(splits, cfg):
    lines = ["## Observed comparison", "",
             "Two-seed means at the agreed fraction threshold, over all retained regions:", "",
             "| split | regions shared by all models | baseline | predicted cascade | cascade − baseline (pp) |",
             "|---|---:|---:|---:|---:|"]
    deltas, matched = [], []
    for split, data in splits.items():
        rates = data["overall_including_no_hands"]
        means = {kind: np.mean([rates[m["name"]]["rate"] for m in cfg["models"] if m["kind"] == kind])
                 if data["n_candidates"] else None for kind in ("baseline", "cascade")}
        delta = means["cascade"] - means["baseline"] if means["baseline"] is not None else None
        delta_text = "undefined" if delta is None else f"{100 * delta:+.2f}"
        lines.append(f"| {split} | {data['n_candidates']:,} | {_percent(means['baseline'])} | {_percent(means['cascade'])} | {delta_text} |")
        deltas.append(delta)
        for seed in sorted({m["seed"] for m in cfg["models"]}):
            pair = {m["kind"]: rates[m["name"]]["rate"] for m in cfg["models"] if m["seed"] == seed}
            if all(v is not None for v in pair.values()):
                matched.append((split, seed, pair["cascade"] - pair["baseline"]))
    lines += [""]
    if deltas and all(d is not None and d > 0 for d in deltas):
        lines += ["The cascade has a higher mean false-positive rate on both splits. The observed result does not support reduced clutter false positives under this fixed SAM background-region proxy.", ""]
    elif deltas and all(d is not None and d < 0 for d in deltas):
        lines += ["The cascade has a lower mean false-positive rate on both splits under this fixed proxy. This is a descriptive comparison; two seeds and correlated proposals do not establish statistical significance.", ""]
    else:
        lines += ["The fixed-proxy comparison does not show a lower cascade rate across every split; interpret the split-specific differences above.", ""]
    if matched:
        lines += ["Same-seed cascade minus baseline overall differences: " + "; ".join(
            f"{split} seed {seed}: {100 * delta:+.3f} pp" for split, seed, delta in matched) + ".", ""]
    lines += ["Inspect the distance-bin results below as well as the overall rates. Region false positives and segmentation IoU measure different behaviour; a higher object IoU can coexist with a higher region false-positive rate.", ""]
    return lines


def report_text(splits, cfg, manifest):
    lines = ["# Phase 6: clutter candidate analysis", "", f"Analysis: `{cfg['analysis_name']}`; fingerprint `{manifest['analysis_fingerprint']}`.",
             "", "This compares the complete single-stage and predicted-hand cascade systems at both existing seeds. Both cascades share frozen seed-0 stage 1. No training, oracle inference or segmentation re-evaluation is performed.",
             "", "Candidates are original SAM ViT-H proposals, not independent labelled object instances. Filters use original GT, never model predictions. Raw and eligible area bounds are inclusive; rejection IoU/inside limits, mask NMS and the FP event use strict greater-than comparisons.",
             "", "Protocol: original RGB to fp32 SAM; SegFormer input shorter side 512; saved bf16 settings; fp32 bilinear logit upsampling before argmax; cascade predicted-only composition scored as class 2. GT hands/objects are removed from retained regions, and every raw proposal touching second-order/255 pixels is rejected.", ""]
    lines += observed_findings(splits, cfg)
    for split, data in splits.items():
        lines += [f"## {split}", "", f"Processed {data['n_images']:,} images; retained {data['n_candidates']:,} regions. {data['n_empty_images']:,} images have no retained regions. {data['n_no_hand_images']:,} images lack labelled hands; their {data['n_no_hand_candidates']:,} regions are excluded from distance curves.",
                  "", "| model | FP / all retained regions (including no-hand images) | rate |", "|---|---:|---:|"]
        for model in cfg["models"]:
            m = data["overall_including_no_hands"][model["name"]]
            lines.append(f"| {model['name']} | {m['fp_count']} / {data['n_candidates']} | {_percent(m['rate'])} |")
        means = {}
        for kind in ("baseline", "cascade"):
            vals = [data["overall_including_no_hands"][m["name"]]["rate"] for m in cfg["models"] if m["kind"] == kind]
            means[kind] = float(np.mean(vals)) if vals[0] is not None else None
        delta = means["cascade"] - means["baseline"] if means["cascade"] is not None else None
        change = "undefined because no regions were retained" if delta is None else f"{100 * delta:+.2f} percentage points ({'lower' if delta < 0 else 'higher' if delta > 0 else 'equal'} cascade rate)"
        lines += ["", f"Two-seed mean overall rates: baseline {_percent(means['baseline'])}; cascade {_percent(means['cascade'])}; cascade minus baseline {change}. This overall value includes no-hand regions; the plot is restricted to defined hand distances.",
                  "", "| distance interval | regions | images | baseline mean | cascade mean | cascade − baseline (pp) |", "|---|---:|---:|---:|---:|---:|"]
        for i, b in enumerate(data["bins"]):
            a, c = b["systems"]["baseline"]["mean_rate"], b["systems"]["cascade"]["mean_rate"]
            d = "undefined" if a is None else f"{100 * (c-a):+.2f}"
            lines.append(f"| [{b['lower']:g}, {b['upper']:g}{']' if i == len(data['bins'])-1 else ')'} | {b['n_candidates']} | {b['n_images']} | {_percent(a)} | {_percent(c)} | {d} |")
        lines += ["", "Per-seed FP counts/rates, descriptive seed ranges and the secondary mean of per-image bin rates are in the JSON. Images with no candidate in a bin do not contribute to that secondary average.",
                  "", "| source | images | raw proposals | retained regions | regions with distance |", "|---|---:|---:|---:|---:|"]
        for source, s in sorted(data["sources"].items()):
            lines.append(f"| {source} | {s['n_images']} | {s['n_proposals']} | {s['n_candidates']} | {s['n_distance_candidates']} |")
        lines += ["", "First-rejection counts: " + "; ".join(f"{k}={v}" for k, v in data["filter_counts"].items()) + ".", ""]
    lines += ["## Interpretation limits", "", "SAM can omit clutter and propose object parts, nested regions and surfaces. Mask NMS removes near-duplicates, not all dependence. Images with more proposals have greater weight in the primary rate. Strict ignore rejection and GT clipping can reduce coverage near hands and fragment regions; retained fragments remain one proposal. Original-resolution pixel distances describe nearest mask pixels, not object centres or physical distance.",
              "", "Two seeds describe observed variation; they do not establish statistical significance. No proposal-independent confidence interval is reported. The compared systems also differ in heads and compute, so differences cannot be attributed solely to the hand channel. Compare systems within each split; easier YouTube results do not establish a closed domain gap.",
              "", "A lower cascade curve and higher near-hand FP rates are hypotheses, not correctness conditions. User review of the pilot masks and final curves/counts is required before Phase 6 is complete.",
              "", "The plot JSON records exact config, package/code/model provenance and the SHA-256 of the compact per-image measurements. Continuous object fractions and distances allow threshold/bin changes without SAM or inference; changing filtering/eligible masks requires remeasurement using the proposal cache.", ""]
    return "\n".join(lines)


def build(root, out_dir, fp_threshold=None, bin_edges=None, prefix="clutter_fp_vs_distance"):
    manifest, saved, records = load_measurements(root)
    cfg = json.loads(json.dumps(manifest["config"]))
    if fp_threshold is not None:
        if not np.isfinite(fp_threshold) or not 0 <= fp_threshold <= 1:
            raise ValueError("FP threshold must be in [0,1]")
        cfg["fp_threshold"] = fp_threshold
    if bin_edges is not None:
        if (len(bin_edges) < 2 or not np.isfinite(bin_edges).all() or bin_edges[0] != 0 or bin_edges[-1] != 1
                or any(a >= b for a, b in zip(bin_edges, bin_edges[1:]))):
            raise ValueError("Bin edges must increase from 0 to 1")
        cfg["distance_bin_edges"] = bin_edges
    if (fp_threshold is not None or bin_edges is not None) and prefix == "clutter_fp_vs_distance":
        raise ValueError("Use --prefix for an alternative threshold/bin analysis; preserve the agreed main plot")
    if Path(prefix).name != prefix or not prefix:
        raise ValueError("prefix must be a file stem")
    splits = aggregate(records, cfg)
    if fp_threshold is None and bin_edges is None and splits != saved["splits"]:
        raise ValueError("Local aggregation does not reproduce the merged summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_plot(splits, cfg, out_dir / f"{prefix}.png")
    info = {"analysis_fingerprint": manifest["analysis_fingerprint"], "config": cfg,
            "splits": splits, "models": manifest["models"], "versions": manifest["versions"],
            "source_hashes": manifest["source_hashes"], "proposal_recipe": manifest["proposal_recipe"],
            "measurements": {"path": str((root / 'measurements.jsonl.gz').resolve()),
                             "sha256": saved["measurements_sha256"]},
            "report_script_sha256": file_sha256(__file__),
            "uncertainty": "Two descriptive seeds, no proposal-independent confidence intervals"}
    atomic_bytes(out_dir / f"{prefix}.json", json.dumps(info, indent=2, allow_nan=False).encode())
    report_name = "phase6_report.md" if prefix == "clutter_fp_vs_distance" else f"{prefix}_report.md"
    atomic_bytes(out_dir / report_name, report_text(splits, cfg, manifest).encode())
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=REPO / "results/phase6/clutter_sam_v1")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results")
    parser.add_argument("--fp-threshold", type=float)
    parser.add_argument("--bin-edges", help="Comma-separated edges, starting at 0 and ending at 1")
    parser.add_argument("--prefix", default="clutter_fp_vs_distance")
    args = parser.parse_args()
    edges = [float(x) for x in args.bin_edges.split(",")] if args.bin_edges else None
    build(args.analysis_dir, args.out_dir, args.fp_threshold, edges, args.prefix)
    print(f"Plot/JSON/report -> {args.out_dir}; final user verification remains required")


if __name__ == "__main__":
    main()
