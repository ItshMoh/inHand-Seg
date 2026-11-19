"""Bounded Phase-7 survey/render operations. Never generates SAM proposals."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.clutter_sam import (atomic_bytes, decode_rle, file_sha256, filter_proposals,
                            fingerprint, image_identity, image_key, read_record, write_record)
from src.clutter_workflow import cached_proposals, load_manifest, load_original
from src.phase7_analysis import (CLASSES, code_hashes, ious, label_statistics, load_inputs,
                                 normal, sealed)
from src.phase7_config import validate_config
from src.phase7_selection import (build_selection, choose_shortlist, key, make_request,
                                  review_template, validate_selection)


def bare(record):
    return {k: v for k, v in record.items() if k not in ("schema", "complete", "payload_sha256")}


def output_root(cfg):
    validate_config(cfg)
    return Path(cfg["analysis_root"]) / cfg["analysis_name"]


def verify_request(request):
    """Rebuild from Volume evaluations and compare the local request byte identities."""
    sealed(request)
    cfg = validate_config(request["manifest"]["config"])
    if request["manifest"]["code_sha256"] != code_hashes():
        raise ValueError("Phase-7 code changed: rebuild locally before remote work")
    phase6_root = Path(cfg["analysis_root"]) / cfg["phase6_name"]
    manifest, rows, measurements, _, _ = load_inputs(cfg, "/data/runs", phase6_root)
    if bare(request) != make_request(manifest, rows, measurements):
        raise ValueError("Local request differs from the saved Volume evaluations/provenance")
    old = read_record(phase6_root / "manifest.json")
    cfg6 = {**old["config"], "execution": old["prepared_execution"]}
    manifest6 = load_manifest(cfg6)  # Retain every frozen source AND runtime compatibility check.
    return cfg, cfg6, manifest6, measurements


def ensure_analysis(root, request):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Analysis root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    manifest = request["manifest"]
    p = root / "manifest.json"
    if p.exists() and bare(read_record(p)) != manifest:
        raise ValueError("Analysis directory already belongs to another immutable identity")
    if not p.exists():
        write_record(p, manifest)
    p = root / "request.json"
    if p.exists() and read_record(p) != request:
        raise ValueError("Existing Phase-7 request differs; use a new analysis name")
    if not p.exists():
        write_record(p, bare(request))


def scan_label(entry, cfg6, cfg):
    path = Path(cfg6["data_root"]) / entry["split"] / "label" / f"{entry['name']}.png"
    if file_sha256(path) != entry["label_sha256"]:
        raise ValueError(f"Original label changed: {entry['name']}")
    with Image.open(path) as im:
        original = np.array(im)
    stats = label_statistics(original, cfg["gallery"]["small_component_fraction"])
    if stats["size"] != [entry["width"], entry["height"]]:
        raise ValueError("Original label/image coordinates differ")
    return stats


def gt_lookup(record, request):
    sealed(record)
    if record["analysis_fingerprint"] != request["analysis_fingerprint"]:
        raise ValueError("CPU statistics belong to another analysis")
    expected = {key(e): e for e in request["manifest"]["inventory"]}
    rows = {key(r): r for r in record["rows"]}
    if len(rows) != len(record["rows"]) or set(rows) != set(expected):
        raise ValueError("CPU label scan has duplicate/missing/unexpected images")
    for k, r in rows.items():
        e = expected[k]
        n = r["original_counts"]
        if set(n) != {str(i) for i in range(9)} or any(type(v) is not int or v < 0 for v in n.values()):
            raise ValueError("Invalid original-label pixel counts")
        counts = {"background": n["0"], "hand": n["1"]+n["2"], "interacting_object": n["3"]+n["4"]+n["5"],
                  "ignore": n["6"]+n["7"]+n["8"]}
        if (r["analysis_fingerprint"] != request["analysis_fingerprint"]
                or r["image"] != image_identity(e) or r["label_sha256"] != e["label_sha256"]
                or r["size"] != [e["width"], e["height"]]
                or r["class_counts"] != counts or r["two_hand_pixels"] != n["5"]
                or sum(r["class_counts"].values()) != e["width"]*e["height"]
                or sum(r["original_counts"].values()) != e["width"]*e["height"]):
            raise ValueError("CPU label statistics have invalid provenance/counts")
        components = r["object_components"]
        if (sum(c["area"] for c in components) != counts["interacting_object"]
                or [c["component_id"] for c in components] != list(range(1, len(components)+1))):
            raise ValueError("GT component areas do not reconcile with object pixels")
        for c in components:
            x0, y0, x1, y1 = c["bbox_xyxy"]
            fraction = c["area"]/(e["width"]*e["height"])
            if (type(c["area"]) is not int or c["area"] <= 0 or not 0 <= x0 < x1 <= e["width"]
                    or not 0 <= y0 < y1 <= e["height"] or c["area"] > (x1-x0)*(y1-y0)
                    or c["frame_fraction"] != fraction
                    or type(c["small"]) is not bool
                    or c["small"] != (fraction <= request["manifest"]["config"]["gallery"]["small_component_fraction"])):
                raise ValueError("Invalid original-coordinate object-component geometry")
    return rows


def _file_intact(root, rel, digest):
    p = root / rel
    return p.is_file() and file_sha256(p) == digest


def survey(request, commit=lambda: None):
    from src.phase7_render import survey_index, survey_thumbnail
    cfg, cfg6, _, measurements = verify_request(request)
    root = output_root(cfg)
    ensure_analysis(root, request)
    inventory = request["manifest"]["inventory"]
    rows = []
    measurements = {key(r): r for r in measurements}
    for i, e in enumerate(inventory):
        dest = root / "survey_records" / f"{image_key(e)}.json"
        expected = {"analysis_fingerprint": request["analysis_fingerprint"],
                    "image": image_identity(e), "label_sha256": e["label_sha256"]}
        if dest.exists():
            row = read_record(dest)
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError("Incompatible completed label-scan record")
        else:
            row = write_record(dest, {**expected, "split": e["split"], "name": e["name"], **scan_label(e, cfg6, cfg)})
        if bool(row["class_counts"]["hand"]) != measurements[key(e)]["has_hands"]:
            raise ValueError("GT hand presence differs from frozen Phase-6 measurement")
        rows.append(bare(row))
        if (i+1) % cfg["execution"]["scan_commit_every"] == 0 or i+1 == len(inventory):
            write_record(root / "survey_progress.json", {"analysis_fingerprint": request["analysis_fingerprint"],
                         "stage": "label_scan", "completed": i+1, "total": len(inventory)})
            commit()
            print(f"[phase7 labels] {i+1}/{len(inventory)}", flush=True)
    expected_paths = {root / "survey_records" / f"{image_key(e)}.json" for e in inventory}
    if set((root / "survey_records").glob("*.json")) != expected_paths:
        raise ValueError("Unexpected survey records; refusing an ambiguous merge")
    record = write_record(root / "gt_stats.json", {"analysis_fingerprint": request["analysis_fingerprint"], "rows": rows})
    gt = gt_lookup(record, request)
    shortlist = choose_shortlist(request, gt)
    write_record(root / "shortlist.json", {"analysis_fingerprint": request["analysis_fingerprint"], "images": shortlist})
    atomic_bytes(root / "review.template.json", json.dumps(review_template(request, shortlist), indent=2).encode())
    files = {}
    entries = {key(e): e for e in inventory}
    for i, row in enumerate(shortlist):
        rel = f"survey/{row['image_id']}.png"
        done = root / "thumbnail_records" / f"{row['image_id']}.json"
        previous = read_record(done) if done.exists() else None
        identity = fingerprint(row)
        if previous and previous["shortlist_row_sha256"] == identity and _file_intact(root, rel, previous["png_sha256"]):
            digest = previous["png_sha256"]
        else:
            rgb, truth = load_original(entries[key(row)], cfg6)
            content = survey_thumbnail(rgb, truth)
            atomic_bytes(root / rel, content)
            digest = hashlib.sha256(content).hexdigest()
            write_record(done, {"shortlist_row_sha256": identity, "png_sha256": digest})
        files[rel] = digest
        if (i+1) % 10 == 0 or i+1 == len(shortlist):
            commit()
    atomic_bytes(root / "survey/index.html", survey_index(shortlist))
    files["survey/index.html"] = file_sha256(root / "survey/index.html")
    summary = write_record(root / "survey_summary.json", {"analysis_fingerprint": request["analysis_fingerprint"],
                           "n_images": len(rows), "n_shortlist": len(shortlist),
                           "gt_stats_sha256": record["payload_sha256"], "files_sha256": files})
    write_record(root / "survey_progress.json", {"analysis_fingerprint": request["analysis_fingerprint"],
                 "stage": "complete", "completed": len(rows), "total": len(rows)})
    commit()
    return {"state": "survey complete", "n_images": len(rows), "n_shortlist": summary["n_shortlist"]}


def selected_region(case, gt, entry, cfg6, manifest6, measured):
    cached, state = cached_proposals(cfg6, manifest6, entry)
    if cached is None:
        raise ValueError(f"Required SAM cache is {state}; stop for recovery review. SAM will not be regenerated.")
    proposals = cached["proposals"]
    if fingerprint(proposals) != measured["proposal_masks_sha256"]:
        raise ValueError("SAM proposal content differs from the measured masks")
    filtered = filter_proposals(proposals, gt, cfg6["filter"], cfg6["distance_bin_edges"])
    if fingerprint(filtered["candidates"]) != measured["eligible_masks_sha256"]:
        raise ValueError("Eligible SAM masks differ from Phase 6")
    mask_id = case["candidate"]["mask_id"]
    actual = next((c for c in filtered["candidates"] if c["mask_id"] == mask_id), None)
    saved = next((c for c in measured["candidates"] if c["mask_id"] == mask_id), None)
    if actual is None or saved is None or any(actual[k] != v for k, v in saved.items() if k != "object_fractions"):
        raise ValueError("Selected candidate metadata does not match the cached eligible mask")
    if any(case["candidate"].get(k) != v for k, v in saved.items()):
        raise ValueError("Selected candidate fractions/identity changed")
    mask = decode_rle(actual["segmentation"])
    if int(mask.sum()) != saved["area"] or (mask & (gt != 0)).any():
        raise ValueError("Selected eligible-mask geometry is invalid")
    return mask, saved


def prediction_statistics(pred, gt):
    from src.phase7_render import object_errors
    fp, fn = object_errors(pred, gt)
    valid = gt != 255
    hist = np.bincount((gt[valid].astype(np.int64)*3 + pred[valid]).ravel(), minlength=9).reshape(3, 3)
    return {"iou": dict(zip(CLASSES, ious(hist))), "confusion_matrix": hist.tolist(),
            "object_fp_pixels": int(fp.sum()), "object_fn_pixels": int(fn.sum())}


def check_predictions(case, predictions, gt, cfg, region=None, candidate=None):
    runs = cfg["baseline_runs"] + cfg["cascade_runs"]
    if set(predictions) != set(runs):
        raise ValueError("Expected predictions for both seeds of both complete systems")
    results = {}
    for run in runs:
        stats = prediction_statistics(predictions[run], gt)
        for cls, value in stats["iou"].items():
            expected = case["scores"][run][cls]
            if (value is None) != (expected is None) or (value is not None and abs(value-expected) > 1e-8):
                raise ValueError(f"Selected prediction does not reproduce saved IoU: {run}/{cls}; {value} vs {expected}")
        if region is not None:
            fraction = int((region & (predictions[run] == 2)).sum()) / int(region.sum())
            if abs(fraction-candidate["object_fractions"][run]) > 1e-12:
                raise ValueError("Selected prediction does not reproduce saved SAM-region fraction")
            stats["candidate_object_fraction"] = fraction
        results[run] = stats
        stats["seed"] = runs.index(run) % 2
        stats["system"] = "baseline" if run in cfg["baseline_runs"] else "predicted_cascade"
    return results


def preview_token(selection):
    return fingerprint({"analysis_fingerprint": selection["analysis_fingerprint"],
                        "cases": [c["case_fingerprint"] for c in selection["cases"] if c["preview"]]})


def preview_content_identity(preview):
    # Timing and cache-hit metadata may change on resume. The reviewed pixels,
    # captions, cases and layout must remain identical.
    return fingerprint({"analysis_fingerprint": preview["analysis_fingerprint"],
                        "preview_token": preview["preview_token"], "n_cases": preview["n_cases"],
                        "visual_files_sha256": {p: sha for p, sha in preview["files_sha256"].items()
                                                if p.endswith((".png", ".html"))}})


def validate_approval(approval, preview, selection):
    sealed(approval)
    sealed(preview)
    if (approval.get("analysis_fingerprint") != selection["analysis_fingerprint"]
            or approval.get("preview_content_sha256") != preview_content_identity(preview)
            or approval.get("preview_token") != preview_token(selection)
            or preview.get("preview_token") != preview_token(selection)
            or approval.get("reviewed_by_user") is not True):
        raise ValueError("Full gallery requires user acceptance of these exact four preview cases/layout")


def _rebuild_selection(selection, request, root):
    shortlist = read_record(root / "shortlist.json")["images"]
    review = review_template(request, shortlist)
    review["categories"] = selection["categories"]
    chosen = {key(c): c for c in selection["cases"]}
    for r in review["cases"]:
        if key(r) in chosen:
            c = chosen[key(r)]
            r.update({k: c[k] for k in ("preview", "tags", "evidence", "caption", "region_kind", "crop_xyxy")})
            r.update(selected=True, candidate_id=c["candidate"]["mask_id"] if c["candidate"] else None)
    rebuilt = build_selection(request, shortlist, review)
    if rebuilt != bare(selection):
        raise ValueError("Selected cases no longer match the reviewed shortlist/data")


def render_selected(request, selection, preview=True, approval=None, commit=lambda: None):
    from src.clutter_workflow import SegmentationPredictor
    from src.phase7_render import gallery_index, render_case
    started = time.perf_counter()
    cfg, cfg6, manifest6, measurements = verify_request(request)
    sealed(selection)
    root = output_root(cfg)
    ensure_analysis(root, request)
    gt_lookup(read_record(root / "gt_stats.json"), request)
    _rebuild_selection(selection, request, root)
    validate_selection(selection, request, final=not preview)
    if not preview:
        validate_approval(approval, read_record(root / "preview_summary.json"), selection)
        write_record(root / "preview_approval.json", bare(approval))
    write_record(root / "case_selection.json", bare(selection))
    entries = {key(e): e for e in request["manifest"]["inventory"]}
    measured = {key(r): r for r in measurements}
    chosen = [c for c in selection["cases"] if c["preview"] or not preview]
    predictor, records, files = None, {}, {}
    new_predictions, prediction_seconds, model_load_seconds = 0, 0., 0.
    for i, case in enumerate(chosen):
        rel = f"case_records/{case['case_id']}.json"
        dest = root / rel
        old = read_record(dest) if dest.exists() else None
        if (old and old["analysis_fingerprint"] == request["analysis_fingerprint"]
                and old["case_fingerprint"] == case["case_fingerprint"]
                and all(_file_intact(root, p, sha) for p, sha in old["files_sha256"].items())):
            record = old
        else:
            rgb, truth = load_original(entries[key(case)], cfg6)
            region, candidate = (selected_region(case, truth, entries[key(case)], cfg6, manifest6, measured[key(case)])
                                 if case["candidate"] else (None, None))
            if predictor is None:
                loaded = time.perf_counter()
                predictor = SegmentationPredictor(cfg6, manifest6)
                model_load_seconds += time.perf_counter() - loaded
            predicted = time.perf_counter()
            predictions = predictor(rgb, truth)
            seconds = time.perf_counter() - predicted
            prediction_seconds += seconds
            new_predictions += 1
            statistics = check_predictions(case, predictions, truth, cfg, region, candidate)
            rendered_at = time.perf_counter()
            rendered, crop = render_case(case, rgb, truth, predictions, cfg, region)
            render_seconds = time.perf_counter() - rendered_at
            for p, content in rendered.items():
                atomic_bytes(root / p, content)
            record = write_record(dest, {"analysis_fingerprint": request["analysis_fingerprint"],
                "case_fingerprint": case["case_fingerprint"], "case": case, "image": image_identity(entries[key(case)]),
                "label_sha256": entries[key(case)]["label_sha256"], "models": statistics,
                "candidate": candidate, "closeup_xyxy": crop,
                "timing_seconds": {"prediction": seconds, "render": render_seconds},
                "files_sha256": {p: hashlib.sha256(v).hexdigest() for p, v in rendered.items()}})
            del predictions
        records[case["case_id"]] = record
        files.update(record["files_sha256"])
        files[rel] = file_sha256(dest)
        write_record(root / "render_progress.json", {"analysis_fingerprint": request["analysis_fingerprint"],
                     "mode": "preview" if preview else "full", "completed": i+1, "total": len(chosen),
                     "stage": "rendering", "last_case_id": case["case_id"]})
        commit()  # One committed, checksummed completed record per selected image.
        print(f"[phase7 {'preview' if preview else 'gallery'}] {i+1}/{len(chosen)} {case['case_id']}", flush=True)
    if set(records) != {c["case_id"] for c in chosen}:
        raise ValueError("Selected render coverage mismatch")
    index = "gallery/preview.html" if preview else "gallery/index.html"
    atomic_bytes(root / index, gallery_index(selection, records, preview))
    files[index] = file_sha256(root / index)
    gpu = None
    if predictor is not None and getattr(getattr(predictor, "device", None), "type", None) == "cuda":
        import torch
        gpu = {"name": torch.cuda.get_device_name(predictor.device), "cuda": torch.version.cuda,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(predictor.device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(predictor.device)}
    summary = write_record(root / ("preview_summary.json" if preview else "gallery_summary.json"), {
        "analysis_fingerprint": request["analysis_fingerprint"], "preview_token": preview_token(selection),
        "case_fingerprints": {c["case_id"]: c["case_fingerprint"] for c in chosen},
        "n_cases": len(chosen), "files_sha256": files,
        "new_prediction_cases_this_call": new_predictions, "gpu": gpu,
        "timing_seconds_this_call": {"total": time.perf_counter()-started, "model_load": model_load_seconds,
                                     "prediction": prediction_seconds},
        "state": "computed; user visual verification pending"})
    write_record(root / "render_progress.json", {"analysis_fingerprint": request["analysis_fingerprint"],
                 "mode": "preview" if preview else "full", "stage": "complete", "completed": len(chosen), "total": len(chosen)})
    commit()
    return {"state": summary["state"], "n_cases": len(chosen), "index": index}


def status(cfg):
    root = output_root(cfg)
    selection = read_record(root / "case_selection.json") if (root / "case_selection.json").exists() else None
    preview = read_record(root / "preview_summary.json") if (root / "preview_summary.json").exists() else None
    gallery = read_record(root / "gallery_summary.json") if (root / "gallery_summary.json").exists() else None
    current_preview = bool(preview and selection and preview["preview_token"] == preview_token(selection))
    current_gallery = bool(gallery and selection and gallery["case_fingerprints"] == {
        c["case_id"]: c["case_fingerprint"] for c in selection["cases"]})
    accepted = False
    if current_preview and (root / "preview_approval.json").exists():
        try:
            validate_approval(read_record(root / "preview_approval.json"), preview, selection)
            accepted = True
        except ValueError:
            pass
    return {"analysis_name": cfg["analysis_name"],
            **{p.removesuffix(".json"): read_record(root / p) for p in
               ("survey_progress.json", "render_progress.json") if (root / p).exists()},
            "survey_complete": (root / "survey_summary.json").exists(),
            "preview_complete": current_preview,
            "preview_accepted": accepted,
            "gallery_complete": current_gallery,
            "user_final_verification": "pending"}


def fetch_files(cfg):
    root = output_root(cfg)
    names = {p for p in ("manifest.json", "request.json", "gt_stats.json", "shortlist.json", "review.template.json",
                         "survey_summary.json", "survey_progress.json", "render_progress.json", "case_selection.json",
                         "preview_summary.json", "preview_approval.json", "gallery_summary.json") if (root / p).is_file()}
    checksums = {}
    for name in ("survey_summary.json", "preview_summary.json", "gallery_summary.json"):
        if (root / name).exists():
            checksums.update(read_record(root / name)["files_sha256"])
    for rel, digest in checksums.items():
        p = Path(rel)
        if (p.is_absolute() or ".." in p.parts or p.parts[0] not in ("survey", "gallery", "case_records")
                or p.suffix not in (".png", ".html", ".json") or (root / p).is_symlink()
                or not (root / p).resolve().is_relative_to(root.resolve()) or not _file_intact(root, rel, digest)):
            raise ValueError("Unsafe, missing or corrupt allowlisted gallery output")
        names.add(rel)
    if any((root / p).is_symlink() for p in names):
        raise ValueError("Refusing symlinked analysis outputs")
    if sum((root / p).stat().st_size for p in names) > cfg["execution"]["fetch_max_bytes"]:
        raise ValueError("Fetch exceeds the agreed small-output byte limit")
    return {p: (root / p).read_bytes() for p in sorted(names)}
