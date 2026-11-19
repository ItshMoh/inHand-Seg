"""Saved-result analysis and label geometry, independent of torch and SAM."""

import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.clutter_sam import aggregate, file_sha256, fingerprint, read_record, validate_coverage
from src.clutter_workflow import source_hashes
from src.phase7_config import validate_config

CLASSES = ["background", "hand", "interacting_object"]
SOURCES = ("ego4d", "epic", "thu", "own", "youtube")
REPO = Path(__file__).resolve().parents[1]


def source(name):
    # Match display_source without importing evaluate.py (which imports torch).
    prefix = name.split("_", 1)[0]
    return prefix if prefix in ("ego4d", "epic", "thu", "youtube") else "own"


def raw_video(name):
    return name.rsplit("_", 1)[0]


def video_aliases(names):
    """Conservatively join explicit download-copy suffixes with matching embedded IDs.

    This is a diversity key, not a claim of independent or deduplicated videos.
    Original filename groups are always retained separately.
    """
    videos = {raw_video(n) for n in names}
    aliases = {v: v for v in videos}
    for v in sorted(videos):
        base = re.sub(r" \([1-9][0-9]*\)$", "", v)
        if (v != base and base in videos and v.startswith("youtube_")
                and re.search(r"-[A-Za-z0-9_-]{11}-1080p-[0-9]+$", base)):
            aliases[v] = base
    return aliases


def code_hashes():
    paths = [*(Path("src") / f"phase7_{n}.py" for n in ("config", "analysis", "selection", "render", "workflow")),
             Path("modal_phase7.py"), Path("scripts/phase7_report.py")]
    return {str(p): file_sha256(REPO / p) for p in paths}


def strict_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def normal(value):
    return json.loads(json.dumps(value))


def sealed(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a checksummed completed record")
    checksum = value.get("payload_sha256")
    payload = {k: v for k, v in value.items() if k != "payload_sha256"}
    if value.get("schema") != 1 or value.get("complete") is not True or checksum != fingerprint(payload):
        raise ValueError("Invalid/incomplete Phase-7 record")
    return value


def load_phase6(root, cfg):
    root = Path(root)
    manifest, summary = read_record(root / "manifest.json"), read_record(root / "summary.json")
    identity = {k: manifest[k] for k in ("config", "inventory", "models", "proposal_fingerprint", "versions", "source_hashes")}
    if (manifest["analysis_fingerprint"] != cfg["phase6_fingerprint"]
            or manifest["analysis_fingerprint"] != fingerprint(identity)
            or manifest["proposal_fingerprint"] != fingerprint(manifest["proposal_recipe"])
            or manifest["config"]["analysis_name"] != cfg["phase6_name"]
            or manifest["source_hashes"] != source_hashes()
            or summary["analysis_fingerprint"] != manifest["analysis_fingerprint"]
            or summary["measurements_sha256"] != file_sha256(root / "measurements.jsonl.gz")):
        raise ValueError("Frozen Phase-6 code/config/measurements identity mismatch")
    with gzip.open(root / "measurements.jsonl.gz", "rt") as stream:
        rows = [sealed(json.loads(line)) for line in stream if line.strip()]
    if any(r["analysis_fingerprint"] != manifest["analysis_fingerprint"] for r in rows):
        raise ValueError("Mixed Phase-6 measurements")
    validate_coverage(manifest["inventory"], rows)
    inventory = {(e["split"], e["name"]): e for e in manifest["inventory"]}
    for row in rows:
        e = inventory[(row["split"], row["name"])]
        if (row["image"] != {k: e[k] for k in ("split", "name", "width", "height", "image_sha256")}
                or row["label_sha256"] != e["label_sha256"]
                or any(row[k] != e[k] for k in ("width", "height", "video", "display_source"))):
            raise ValueError("Phase-6 measurement metadata differs from its frozen inventory")
    if aggregate(rows, manifest["config"]) != summary["splits"]:
        raise ValueError("Phase-6 compact measurements do not reproduce the summary")
    return manifest, summary, rows


def ious(hist):
    hist = np.asarray(hist)
    inter = np.diag(hist)
    union = hist.sum(0) + hist.sum(1) - inter
    return [float(a / b) if b else None for a, b in zip(inter, union)]


def validate_eval(m, run, split, classes, count, mode=None):
    if (m["run_name"] != run or m["split"] != split or m["class_names"] != classes
            or m["n_images"] != count or len(m["per_image"]) != count
            or m["boundary_dilation_ratio"] != 0.02):
        raise ValueError(f"Evaluation identity/protocol mismatch: {run}/{split}/{mode}")
    if mode:
        if m.get("hand_mode") != mode or m.get("stage2_label_map") != {"1": 0, "2": 1}:
            raise ValueError("Expected composed three-class cascade evaluation")
    elif m.get("label_map") != ({"2": 0} if len(classes) == 2 else None):
        raise ValueError("Wrong original/evaluation label map")
    h = np.asarray(m["confusion_matrix"])
    if h.shape != (len(classes), len(classes)) or h.dtype.kind not in "iu" or (h < 0).any():
        raise ValueError("Invalid pooled confusion matrix")
    if list(m["iou"]) != classes or any(a != b for a, b in zip(ious(h), m["iou"].values())):
        raise ValueError("Pooled IoU does not match the confusion matrix")
    for row in m["per_image"].values():
        if (len(row["size"]) != 2 or any(type(v) is not int or v <= 0 for v in row["size"])
                or set(row["iou"]) != set(classes)):
            raise ValueError("Invalid per-image schema")
        for value in row["iou"].values():
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Invalid per-image IoU")
    if any(v is not None and (not math.isfinite(v) or not 0 <= v <= 1) for v in m["boundary_iou"].values()):
        raise ValueError("Invalid Boundary IoU")


def describe(values):
    valid = [v for v in values if v is not None]
    return {"n_total": len(values), "n_valid": len(valid), "n_undefined": len(values) - len(valid),
            "mean": float(np.mean(valid)) if valid else None,
            "median": float(np.median(valid)) if valid else None,
            "q25": float(np.quantile(valid, .25)) if valid else None,
            "q75": float(np.quantile(valid, .75)) if valid else None}


def image_rows(evals, cfg):
    """Join by split and original name; never zip model listing orders."""
    runs = cfg["baseline_runs"] + cfg["cascade_runs"]
    rows = []
    for split, count in cfg["splits"].items():
        sets = [set(evals[(run, split, "predicted" if run in cfg["cascade_runs"] else None)]["per_image"]) for run in runs]
        if len(sets[0]) != count or any(s != sets[0] for s in sets):
            raise ValueError("Main-model image inventories differ")
        aliases = video_aliases(sets[0])
        for name in sorted(sets[0]):
            records = {run: evals[(run, split, "predicted" if run in cfg["cascade_runs"] else None)]["per_image"][name] for run in runs}
            if len({tuple(r["size"]) for r in records.values()}) != 1:
                raise ValueError("Original-resolution coordinates differ across models")
            row = {"split": split, "name": name, "source": source(name), "video": raw_video(name),
                   "diversity_video": aliases[raw_video(name)], "size": records[runs[0]]["size"],
                   "scores": {run: r["iou"] for run, r in records.items()}, "paired": {}}
            for cls in ("hand", "interacting_object"):
                values = [records[r]["iou"][cls] for r in runs]
                valid = all(v is not None for v in values)
                delta = [values[i + 2] - values[i] for i in (0, 1)] if valid else None
                row["paired"][cls] = {"eligible": valid, "per_seed_delta": delta,
                    "mean_delta": float(np.mean(delta)) if valid else None,
                    "same_direction": bool(delta[0] * delta[1] > 0) if valid else None,
                    "baseline_mean": float(np.mean(values[:2])) if valid else None,
                    "cascade_mean": float(np.mean(values[2:])) if valid else None}
            rows.append(row)
    return rows


def label_statistics(original, small_fraction):
    """Geometry is a connected-component proxy, not instance annotation."""
    if original.ndim != 2 or not original.size or not np.isin(original, np.arange(9)).all():
        raise ValueError("Expected original 0–8 HW labels")
    n = np.bincount(original.ravel(), minlength=9)
    obj = np.isin(original, [3, 4, 5]).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    components = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        components.append({"component_id": i, "area": area, "frame_fraction": area / original.size,
                           "bbox_xyxy": [x, y, x+w, y+h], "small": area / original.size <= small_fraction})
    return {"size": [original.shape[1], original.shape[0]],
            "original_counts": {str(i): int(v) for i, v in enumerate(n)},
            "class_counts": {"background": int(n[0]), "hand": int(n[1]+n[2]),
                             "interacting_object": int(n[3:6].sum()), "ignore": int(n[6:].sum())},
            "two_hand_pixels": int(n[5]), "object_components": components}


def source_summaries(rows, runs, gt=None):
    gt = gt or {}
    summaries = []
    for split, src in sorted({(r["split"], r["source"]) for r in rows}):
        group = [r for r in rows if (r["split"], r["source"]) == (split, src)]
        has_gt = all((split, r["name"]) in gt for r in group)
        entry = {"split": split, "source": src, "n_images": len(group),
                 "n_filename_video_groups": len({r["video"] for r in group}),
                 "n_diversity_video_groups": len({r["diversity_video"] for r in group}), "classes": {}}
        for cls in ("hand", "interacting_object"):
            scores = {run: describe([r["scores"][run][cls] for r in group]) for run in runs}
            paired = [r for r in group if r["paired"][cls]["eligible"]]
            info = {"per_run_all_defined": scores,
                    "paired_four_run_defined": {"n_eligible": len(paired), "n_excluded": len(group)-len(paired),
                        "delta": describe([r["paired"][cls]["mean_delta"] for r in paired]),
                        "per_seed_delta": [describe([r["paired"][cls]["per_seed_delta"][s] for r in paired]) for s in (0, 1)],
                        "baseline": describe([r["paired"][cls]["baseline_mean"] for r in paired]),
                        "cascade": describe([r["paired"][cls]["cascade_mean"] for r in paired])}}
            if has_gt:
                present = [r for r in group if gt[(split, r["name"])]["class_counts"][cls] > 0]
                absent = [r for r in group if gt[(split, r["name"])]["class_counts"][cls] == 0]
                if any(r["scores"][run][cls] is None for r in present for run in runs):
                    raise ValueError("GT-present class has undefined saved IoU")
                if any(r["scores"][run][cls] not in (0, None) for r in absent for run in runs):
                    raise ValueError("GT-absent class has nonzero saved IoU")
                info["gt_present"] = {"n_images": len(present), "per_run": {
                    run: describe([r["scores"][run][cls] for r in present]) for run in runs},
                    "paired_delta": describe([r["paired"][cls]["mean_delta"] for r in present])}
                info["gt_absent"] = {"n_images": len(absent), "prediction_present_count": {
                    run: sum(r["scores"][run][cls] == 0 for r in absent) for run in runs}}
            entry["classes"][cls] = info
        if has_gt:
            entry["gt_pixel_counts"] = {c: sum(gt[(split, r["name"])]["class_counts"][c] for r in group)
                                        for c in ("background", "hand", "interacting_object", "ignore")}
        summaries.append(entry)
    return summaries


def split_tables(evals, cfg):
    result = []
    for (run, split, mode), m in evals.items():
        h = np.asarray(m["confusion_matrix"])
        k = m["class_names"].index("interacting_object") if "interacting_object" in m["class_names"] else None
        result.append({"run": run, "split": split, "mode": mode, "n_images": m["n_images"],
                       "iou": m["iou"], "boundary_iou": m["boundary_iou"],
                       "object_pixel_precision": float(h[k,k] / h[:,k].sum()) if k is not None and h[:,k].sum() else None,
                       "object_pixel_recall": float(h[k,k] / h[k].sum()) if k is not None and h[k].sum() else None})
    comparisons = []
    groups = [([run], mode, run + (f" ({mode})" if mode else ""))
              for run, mode in sorted({(r, m) for r, _, m in evals}, key=lambda x: (x[0], x[1] or ""))]
    groups += [(cfg["baseline_runs"], None, "baseline mean"),
               (cfg["cascade_runs"], "predicted", "predicted cascade mean"),
               (cfg["cascade_runs"], "oracle", "oracle mean")]
    for runs, mode, system in groups:
        for metric in ("iou", "boundary_iou"):
            for cls in ("hand", "interacting_object"):
                if any(cls not in evals[(r, split, mode)][metric] for r in runs for split in cfg["splits"]):
                    continue
                means = {split: float(np.mean([evals[(r, split, mode)][metric][cls] for r in runs])) for split in cfg["splits"]}
                comparisons.append({"system": system, "metric": metric, "class": cls, **means,
                                    "youtube_minus_indomain": means["test_outdomain"] - means["test_indomain"]})
    return result, comparisons


def load_inputs(cfg, runs_dir, phase6_dir):
    validate_config(cfg)
    runs_dir = Path(runs_dir)
    manifest6, summary6, measurements = load_phase6(phase6_dir, cfg)
    evals, files, configs = {}, {}, {}
    runs = cfg["baseline_runs"] + cfg["cascade_runs"] + ["b0_head_only", "b0_hands_only", "b0_hands_only_s1"]
    for run in runs:
        root = runs_dir / run
        c = yaml.safe_load((root / "config.yaml").read_text())
        configs[run] = c
        if c["run_name"] != run or c["seed"] != int(run.endswith("_s1")) or c["train"]["bf16"] is not True:
            raise ValueError("Run name/seed/precision mismatch")
        info = manifest6["models"].get(run) or (manifest6["models"]["stage1"] if run == "b0_hands_only" else None)
        if info and normal(c) != info["run_config"]:
            raise ValueError("Local saved config differs from the measured final checkpoint config")
        for name in ("config.yaml", "summary.json"):
            files[f"runs/{run}/{name}"] = file_sha256(root / name)
        for split, count in {"val": 1124, **cfg["splits"]}.items():
            for mode in (("predicted", "oracle") if run in cfg["cascade_runs"] else (None,)):
                rel = f"eval/{split}{'_'+mode if mode else ''}.json"
                m = strict_json(root / rel)
                classes = ["background", "hand"] if "hands_only" in run else CLASSES
                validate_eval(m, run, split, classes, count, mode)
                if mode and m["stage1_checkpoint"] != manifest6["config"]["stage1_checkpoint"]:
                    raise ValueError("Cascade stage-1 dependency changed")
                evals[(run, split, mode)] = m
                files[f"runs/{run}/{rel}"] = file_sha256(root / rel)
    for split in ("val", *cfg["splits"]):
        ref = evals[(cfg["baseline_runs"][0], split, None)]["per_image"]
        for (_, s, _), m in evals.items():
            if s == split and (set(m["per_image"]) != set(ref) or any(m["per_image"][n]["size"] != ref[n]["size"] for n in ref)):
                raise ValueError("Context models/splits are not aligned")
        hand = evals[("b0_hands_only", split, None)]
        for run in cfg["cascade_runs"]:
            for metric in ("iou", "boundary_iou"):
                if evals[(run, split, "predicted")][metric]["hand"] != hand[metric]["hand"]:
                    raise ValueError("Composed hands differ from shared seed-0 stage 1")
    rows = image_rows(evals, cfg)
    inventory = {(e["split"], e["name"]): e for e in manifest6["inventory"]}
    if {(r["split"], r["name"]) for r in rows} != set(inventory):
        raise ValueError("Saved evaluations differ from frozen original-image inventory")
    for row in rows:
        e = inventory[(row["split"], row["name"])]
        if row["size"] != [e["width"], e["height"]] or row["source"] != e["display_source"]:
            raise ValueError("Saved image dimensions/source differ from Phase 6")
    identity = {"config": cfg, "input_files_sha256": files, "code_sha256": code_hashes(),
                "phase6_manifest_sha256": manifest6["payload_sha256"], "phase6_measurements_sha256": summary6["measurements_sha256"],
                "models": {k: {q: v[q] for q in ("checkpoint", "sha256")} for k, v in manifest6["models"].items()}}
    manifest = {**identity, "analysis_fingerprint": fingerprint(identity), "inventory": manifest6["inventory"],
                "metric_definitions": {"source": "unweighted per-image IoU summaries, not pooled IoU",
                    "paired": "four-run defined intersection; mean same-seed deltas per image; equal image weights",
                    "undefined": "absent in both GT and prediction; retained as null",
                    "quantiles": "NumPy linear interpolation at 0.25, 0.50, 0.75; descriptive image distributions",
                    "split_delta": "YouTube minus in-domain; positive means higher YouTube score",
                    "uncertainty": "descriptive seeds; no confidence intervals or significance claim"}}
    return manifest, rows, measurements, split_tables(evals, cfg), summary6
