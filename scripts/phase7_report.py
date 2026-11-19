"""Local saved-data Phase-7 report, manual case selection and preview acceptance.

Commands: build, select, approve-preview. No remote work or model inference.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.clutter_sam import atomic_bytes, file_sha256, read_record, write_record
from src.phase7_analysis import load_inputs, source_summaries, strict_json
from src.phase7_config import FAILURE_TAGS, load_config
from src.phase7_selection import build_selection, choose_shortlist, make_request, review_template, validate_selection
from src.phase7_workflow import bare, gt_lookup, preview_content_identity, preview_token


def pts(v, signed=False):
    return "undefined" if v is None else format(100*v, "+.2f" if signed else ".2f")


def table(header, rows):
    def cell(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(header) + " |", "|" + "---|"*len(header)] + ["| " + " | ".join(map(cell, r)) + " |" for r in rows]


def source_plot(sources, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), squeeze=False)
    for ax, cls in zip(axes[:,0], ("hand", "interacting_object")):
        for i, src in enumerate(sources):
            p = src["classes"][cls]["paired_four_run_defined"]
            for offset, system, colour in ((-.10, "baseline", "#3266b2"), (.10, "cascade", "#d36320")):
                s = p[system]
                if s["mean"] is not None:
                    ax.plot([i+offset]*2, [100*s["q25"], 100*s["q75"]], color=colour, lw=4, alpha=.55)
                    ax.plot(i+offset, 100*s["mean"], "o", color=colour, label=system if i == 0 else None)
        ax.set_xticks(np.arange(len(sources)), [f"{s['source']}\nn={s['classes'][cls]['paired_four_run_defined']['n_eligible']}" for s in sources])
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"{cls}: per-image IoU × 100")
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("Source comparison on the four-run defined-score intersection\nDot = mean; bar = image IQR; seed-averaged per image; no confidence intervals")
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_text(manifest, sources, tables, phase6, root, gt_ready, selection=None, gallery=None):
    pooled, signed = tables
    cfg = manifest["config"]
    out = ["# Phase 7: source analysis and selected failure gallery", "",
           f"Analysis `{cfg['analysis_name']}`; fingerprint `{manifest['analysis_fingerprint']}`.", "",
           "Status: saved-result analysis computed. " + ("CPU label scan complete." if gt_ready else "CPU label scan pending.") +
           " Real panels, category captions and final source comparisons require user verification. Phase 7 is not complete.", "",
           "The primary quantitative population is the complete 1,126-image in-domain and 500-image YouTube test splits. "
           "Both baseline seeds and both predicted-cascade stage-2 seeds are included. Both cascades share frozen seed-0 stage 1. "
           "Existing val, head-only, hands-only and oracle metrics are context; they add no inference.", "",
           "Input shorter side 512; saved bf16; fp32 bilinear logit upsampling to original resolution, align_corners=False, then argmax. "
           "Original GT is background/hand/interacting object; second-order IDs 6–8 become 255 and are excluded. "
           "Split IoU is pooled from confusion counts; Boundary IoU pools boundary intersections/unions with width 2% of the original diagonal.", "",
           "## Saved pooled split metrics", ""]
    rows = []
    for split in ("val", *cfg["splits"]):
        for p in pooled:
            if p["split"] != split:
                continue
            rows.append([split, p["run"], p["mode"] or "single", p["n_images"],
                         *(pts(p["iou"].get(c)) if c in p["iou"] else "N/A" for c in ("background", "hand", "interacting_object")),
                         *(pts(p["boundary_iou"].get(c)) if c in p["boundary_iou"] else "N/A" for c in ("hand", "interacting_object"))])
        for runs, mode, label in ((cfg["baseline_runs"], None, "baseline mean"),
                                  (cfg["cascade_runs"], "predicted", "predicted cascade mean"),
                                  (cfg["cascade_runs"], "oracle", "oracle mean")):
            pp = [p for p in pooled if p["run"] in runs and p["split"] == split and p["mode"] == mode]
            rows.append([split, label, "seeds 0,1", pp[0]["n_images"],
                         *(pts(sum(p["iou"][c] for p in pp)/2) for c in ("background", "hand", "interacting_object")),
                         *(pts(sum(p["boundary_iou"][c] for p in pp)/2) for c in ("hand", "interacting_object"))])
    out += table(["split", "run/system", "mode", "images", "IoU bg", "IoU hand", "IoU object", "BIoU hand", "BIoU object"], rows)
    out += ["", "All scores are ×100. Hands-only background includes objects; compare its hand scores only. "
            "Oracle changes both hand input and pasted hands; perfect hand scores are by construction. Means average two final-model metrics, not an ensemble.", "",
            "## Signed split differences", ""]
    out += table(["system", "metric", "class", "in-domain", "YouTube", "YouTube − in-domain (points)"],
                 [[r["system"], r["metric"], r["class"], pts(r["test_indomain"]), pts(r["test_outdomain"]), pts(r["youtube_minus_indomain"], True)] for r in signed])
    out += ["", "Object IoU is higher on YouTube for every object-predicting system; oracle hands remain perfect by construction. These different source/content mixtures do not isolate a causal domain effect or demonstrate a closed domain gap. "
            "The small in-domain object-IoU gain is smaller than the observed baseline seed spread (about 0.7 points). Two seeds do not establish significance.", "",
            "## Source-level per-image summaries", "",
            "These are unweighted image summaries, not pooled source IoU. Image dimensions cannot reconstruct the missing intersections/unions. "
            "No per-source/per-image Boundary IoU is invented. Each per-run row uses that run's defined scores; its valid count can differ. "
            "A null IoU means absent in both GT and prediction; zero remains zero and can include predictions on GT-absent images.", ""]
    out += table(["split", "source", "images", "raw filename groups", "conservative diversity groups"],
                 [[s["split"], s["source"], s["n_images"], s["n_filename_video_groups"], s["n_diversity_video_groups"]] for s in sources])
    out += ["", "Grouping removes explicit download-copy suffixes only when a matching embedded YouTube ID and original filename group exist. "
            "Raw groups remain in provenance. Neither grouping proves video independence or duplicate-content removal.", ""]
    runs = cfg["baseline_runs"] + cfg["cascade_runs"]
    for cls in ("hand", "interacting_object"):
        out += [f"### {cls}", ""]
        out += table(["source/split", "run", "valid / total", "undefined", "mean", "median", "Q25", "Q75"],
            [[f"{s['source']} / {s['split']}", run, f"{v['n_valid']}/{v['n_total']}", v["n_undefined"],
              *(pts(v[k]) for k in ("mean", "median", "q25", "q75"))]
             for s in sources for run in runs for v in [s["classes"][cls]["per_run_all_defined"][run]]])
        out += ["", "Paired comparisons use the intersection where all four runs have a defined score. "
                "For each image, average the two same-seed cascade-minus-baseline differences, then average images equally. "
                "Seed-specific deltas below use exactly this shared cohort; same seed numbers do not imply matched training randomness.", ""]
        out += table(["source", "eligible", "excluded", "baseline mean", "cascade mean", "mean Δ (points)", "seed-0 Δ", "seed-1 Δ"],
            [[s["source"], p["n_eligible"], p["n_excluded"], pts(p["baseline"]["mean"]), pts(p["cascade"]["mean"]),
              pts(p["delta"]["mean"], True), *(pts(v["mean"], True) for v in p["per_seed_delta"])]
             for s in sources for p in [s["classes"][cls]["paired_four_run_defined"]]])
        out += [""]
        if gt_ready:
            out += ["GT presence comes only from the original labels. This model-independent GT-present cohort is separate from the shared-defined cohort. "
                    "GT-absent prediction counts indicate any class prediction on valid pixels, not SAM-region events.", ""]
            out += table(["source", "GT-present images", "GT-absent images", *(f"{r} present mean / absent prediction count" for r in runs), "GT-present mean Δ"],
                [[s["source"], c["gt_present"]["n_images"], c["gt_absent"]["n_images"],
                  *(f"{pts(c['gt_present']['per_run'][r]['mean'])} / {c['gt_absent']['prediction_present_count'][r]}" for r in runs),
                  pts(c["gt_present"]["paired_delta"]["mean"], True)] for s in sources for c in [s["classes"][cls]]])
            out += [""]
    if gt_ready:
        out += ["### Original GT pixel support", "", "Counts below include the ignored population explicitly. Pixel shares use all original frame pixels in the same split.", ""]
        totals = {split: sum(sum(s["gt_pixel_counts"].values()) for s in sources if s["split"] == split) for split in cfg["splits"]}
        out += table(["source", "split pixel share", "background", "hand", "object", "ignore"],
            [[s["source"], f"{100*sum(s['gt_pixel_counts'].values())/totals[s['split']]:.2f}%",
              *(s["gt_pixel_counts"][c] for c in ("background", "hand", "interacting_object", "ignore"))] for s in sources]) + [""]
    out += ["## Pixel precision/recall diagnostic", "",
            "Computed from the saved three-class split confusion matrices (GT rows, prediction columns), with 255 excluded. "
            "These pixel metrics are distinct from the fixed SAM-region majority FP measure and do not establish a causal explanation.", ""]
    out += table(["split", "run", "object pixel precision", "object pixel recall"],
        [[p["split"], p["run"], pts(p["object_pixel_precision"]), pts(p["object_pixel_recall"])]
         for p in pooled if p["run"] in runs and p["split"] in cfg["splits"] and p["mode"] in (None, "predicted")])
    out += ["", "## Frozen Phase-6 context", "",
            "FP means strictly more than half of a retained eligible background proposal is predicted as composed object class 2. "
            "The candidate denominator is shared by all models. Proposals include correlated/nested regions, surfaces, clothing and object parts, not independently labelled untouched objects.", ""]
    rows = []
    for split, s in phase6["splits"].items():
        rates = s["overall_including_no_hands"]
        b, p = [sum(rates[r]["rate"] for r in cfg[group])/2 for group in ("baseline_runs", "cascade_runs")]
        rows.append([split, s["n_images"], s["n_candidates"], f"{100*b:.2f}%", f"{100*p:.2f}%", f"{100*(p-b):+.2f}"])
    out += table(["split", "images", "regions", "baseline mean", "cascade mean", "cascade − baseline (pp)"], rows)
    out += ["", "This fixed proxy does not support reduced cascade clutter FP. Its higher FP rate can coexist with higher object IoU. "
            "Thresholds, filters, seeds and completed results are unchanged.", "", "## Selected qualitative gallery", "",
            "The shortlist is deterministic exploratory diagnosis after the frozen evaluation: source support, low object IoU, paired improvements/regressions, "
            "GT small components, original ID 5 and both-seed-consistent SAM disagreements. At most two frames per conservative video group. "
            "Small means an 8-connected GT object component at most 1% of original frame area; components are not independent object instances. "
            "ID 5 suggests two-hand interaction, with visual review still required. Appearance categories are manually reviewed and may be unsupported.", ""]
    if selection:
        out += table(["category", "review status", "selected examples", "note"],
            [[tag, v["status"], sum(tag in c["tags"] for c in selection["cases"]), v["note"]] for tag, v in selection["categories"].items()])
        out += [""] + table(["case ID", "split/source", "tags", "caption"],
            [[c["case_id"], f"{c['split']}/{c['source']}", ", ".join(c["tags"]), c["caption"]] for c in selection["cases"]]) + [""]
    else:
        out += ["Case review/selection is pending. No failure category has been confirmed.", ""]
    out += ["Gallery computation: " + (f"{gallery['n_cases']} cases rendered; final user review pending." if gallery else "pending."), "",
            "Panels: original RGB / baseline / predicted cascade over original GT / baseline object errors / cascade object errors. "
            "Blue=hand, yellow=object, grey hatch=ignore; red=object FP, cyan=object FN. Object FP includes hand/object confusion; "
            "SAM outlines identify specifically eligible GT-background regions. Full-frame and matched close-up views share coordinates. "
            "Display uses seed 0 for both systems; both-seed scores/consistency are recorded. Captions describe evidence; mechanisms remain hypotheses.", "",
            "## Provenance and limits", "",
            f"Machine-readable files: `phase7/{cfg['analysis_name']}/manifest.json`, `source_metrics.json`, `request.json`, "
            "and, after the CPU scan, `gt_stats.json`, `shortlist.json` and the editable `review.json`. "
            "Selected cases are in `case_selection.json`; panels and per-case records are fetched separately.", "",
            "Manifest hashes bind saved configs/evaluations, all four final checkpoints and shared stage 1, original image/label identities, "
            "Phase-6 measurements and both analysis/proposal fingerprints through its sealed manifest, and new Phase-7 code. "
            "Dense predictions and SAM bitmaps are not reconstructed from summary JSONs. Only selected predictions are recomputed; completed run directories remain untouched.", "",
            "Compared systems differ in task/head and compute. No binary RGB ablation isolates the hand channel. "
            "No confidence intervals, significance, category prevalence or video independence is claimed. Final user verification is pending.", ""]
    return "\n".join(out)


def build(config, runs_dir, phase6_dir, root, report):
    cfg = load_config(config)
    root = Path(root)
    if root.resolve().is_relative_to(REPO) and not root.resolve().is_relative_to(REPO / "results/phase7"):
        raise ValueError("Repository-local Phase-7 outputs must stay under results/phase7")
    report_path = Path(report).resolve()
    if (report_path.suffix != ".md" or (report_path.is_relative_to(REPO)
            and report_path != REPO / "results/phase7_report.md"
            and not report_path.is_relative_to(REPO / "results/phase7"))):
        raise ValueError("Use a dedicated Phase-7 Markdown report path")
    for protected in (Path(runs_dir).resolve(), Path(phase6_dir).resolve(), REPO / "data", REPO / "src", REPO / "configs"):
        if root.resolve() == protected or protected in root.resolve().parents or root.resolve() in protected.parents:
            raise ValueError("Local output directory overlaps protected inputs")
        if Path(report).resolve() == protected or protected in Path(report).resolve().parents:
            raise ValueError("Report path overlaps protected inputs")
    if Path(report).resolve() in (REPO / "phase.md", REPO / "skild_project.md"):
        raise ValueError("Report cannot overwrite the project phase plan/brief")
    manifest, rows, measurements, tables, summary6 = load_inputs(cfg, runs_dir, phase6_dir)
    if (root / "manifest.json").exists() and bare(read_record(root / "manifest.json")) != manifest:
        raise ValueError("Existing analysis identity changed; use a new analysis_name/output directory")
    root.mkdir(parents=True, exist_ok=True)
    write_record(root / "manifest.json", manifest)
    request = write_record(root / "request.json", make_request(manifest, rows, measurements))
    gt = gt_lookup(read_record(root / "gt_stats.json"), request) if (root / "gt_stats.json").exists() else None
    if gt:
        shortlist = choose_shortlist(request, gt)
        if (root / "shortlist.json").exists() and read_record(root / "shortlist.json")["images"] != shortlist:
            raise ValueError("Fetched shortlist does not reproduce deterministic local selection")
        template = review_template(request, shortlist)
        atomic_bytes(root / "review.template.json", json.dumps(template, indent=2).encode())
        if not (root / "review.json").exists():
            atomic_bytes(root / "review.json", json.dumps(template, indent=2).encode())
    sources = source_summaries(rows, cfg["baseline_runs"]+cfg["cascade_runs"], gt)
    write_record(root / "source_metrics.json", {"analysis_fingerprint": manifest["analysis_fingerprint"],
                 "definitions": manifest["metric_definitions"], "cpu_gt_scan_complete": gt is not None,
                 "sources": sources, "pooled_split_metrics": tables[0], "signed_split_differences": tables[1], "per_image": rows})
    source_plot(sources, root / "source_comparison.png")
    selection = read_record(root / "case_selection.json") if (root / "case_selection.json").exists() else None
    if selection:
        validate_selection(selection, request)
    gallery = read_record(root / "gallery_summary.json") if (root / "gallery_summary.json").exists() else None
    if gallery and (not selection or gallery["analysis_fingerprint"] != manifest["analysis_fingerprint"]
                    or gallery["case_fingerprints"] != {c["case_id"]: c["case_fingerprint"] for c in selection["cases"]}):
        gallery = None
    atomic_bytes(report, report_text(manifest, sources, tables, summary6, root, gt is not None, selection, gallery).encode())
    print(f"Source analysis: {len(rows)} test images, {len(sources)} source/split groups; CPU GT scan {'complete' if gt else 'pending'}")
    print(f"Report: {report}\nOutputs: {root}")
    return request


def select(root, final=False):
    request = read_record(root / "request.json")
    selection = build_selection(request, read_record(root / "shortlist.json")["images"], strict_json(root / "review.json"))
    validate_selection(selection, request, final=final)
    write_record(root / "case_selection.json", selection)
    print(f"Selected {len(selection['cases'])} cases; exactly four previews. Category counts: {dict(Counter(t for c in selection['cases'] for t in c['tags']))}")


def approve_preview(root):
    selection = read_record(root / "case_selection.json")
    preview = read_record(root / "preview_summary.json")
    if preview["preview_token"] != preview_token(selection) or preview["n_cases"] != 4:
        raise ValueError("Current selection differs from the four rendered previews")
    for rel, sha in preview["files_sha256"].items():
        if file_sha256(root / rel) != sha:
            raise ValueError("Preview files are missing or changed; fetch and inspect them first")
    write_record(root / "preview_approval.json", {"analysis_fingerprint": selection["analysis_fingerprint"],
                 "preview_summary_sha256": preview["payload_sha256"], "preview_token": preview_token(selection),
                 "preview_content_sha256": preview_content_identity(preview),
                 "reviewed_by_user": True, "statement": "User explicitly ran approve-preview after inspecting the four real panels/layout"})
    print("Recorded user preview acceptance. Final gallery and source-report verification remain pending.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("build", "select", "approve-preview"), nargs="?", default="build")
    p.add_argument("--config", type=Path, default=REPO / "configs/phase7/phase7_v1.yaml")
    p.add_argument("--runs-dir", type=Path, default=REPO / "results/runs")
    p.add_argument("--phase6-dir", type=Path, default=REPO / "results/phase6/clutter_sam_v1")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--report", type=Path, default=REPO / "results/phase7_report.md")
    p.add_argument("--final", action="store_true", help="Require every category status resolved when selecting")
    args = p.parse_args()
    cfg = load_config(args.config)
    root = args.out_dir or REPO / "results/phase7" / cfg["analysis_name"]
    if args.command == "build":
        build(args.config, args.runs_dir, args.phase6_dir, root, args.report)
    elif args.command == "select":
        select(root, args.final)
    else:
        approve_preview(root)


if __name__ == "__main__":
    main()
