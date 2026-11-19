"""Meaningful synthetic CPU checks; no SAM installation, dataset or model download.

    .venv/bin/python scripts/check_phase6_local.py
"""

import copy
import gzip
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.clutter_config import DEFAULTS, _merge, canonical, load_config, semantic_config, validate_config
from src.clutter_lock import recover_writer, require_writer, writer_lease
from src.clutter_sam import (
    aggregate, compatible_record, decode_rle, distance_bin, encode_rle, file_sha256, filter_proposals,
    fingerprint, hand_distance, image_identity, owned_entries, read_record, score_candidates,
    validate_coverage, validate_proposals, validate_rle, write_record,
)
from src.clutter_workflow import (
    cached_proposals, measure_entry, measurement_path, merge_analysis, obtain_proposals, pick_pilot,
    proposal_path, small_output_files, validate_measurement,
    run_shard,
    approve_pilot, require_approval,
)

RESULTS = {}


def check(name, ok):
    RESULTS[name] = bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def raises(fn, exception=ValueError):
    try:
        fn()
    except exception:
        return True
    return False


def proposal(mask, mask_id=0, quality=0.99):
    return {"mask_id": mask_id, "area": int(mask.sum()), "segmentation": encode_rle(mask),
            "predicted_iou": quality, "stability_score": 0.99}


def filtered(mask, gt=None, config=None):
    gt = np.zeros(mask.shape, np.uint8) if gt is None else gt
    return filter_proposals([proposal(mask)], gt, config or DEFAULTS["filter"], DEFAULTS["distance_bin_edges"])


def main():
    tmp = Path(tempfile.mkdtemp(prefix="phase6_check_"))
    cfg = load_config(REPO / "configs/phase6/clutter_sam_v1.yaml")
    check("separate analysis config validates without entering the training-config glob", cfg == DEFAULTS)
    check("unknown and mistyped config fields are rejected", raises(lambda: _merge(DEFAULTS, {"sam": {"unknown": 1}}))
          and raises(lambda: validate_config(_merge(DEFAULTS, {"execution": {"shards": True}}))))
    check("unsupported model populations, ignore policy, bins and SAM variant are rejected", all((
        raises(lambda: validate_config(_merge(DEFAULTS, {"models": DEFAULTS["models"][:1]}))),
        raises(lambda: validate_config(_merge(DEFAULTS, {"filter": {"reject_any_ignore": False}}))),
        raises(lambda: validate_config(_merge(DEFAULTS, {"distance_bin_edges": [0, 0.1, 0.1, 1]}))),
        raises(lambda: validate_config(_merge(DEFAULTS, {"sam": {"variant": "vit_b"}}))),
    )))
    check("NaN and output paths overlapping completed model runs are rejected", raises(lambda: validate_config(
        _merge(DEFAULTS, {"filter": {"foreground_iou_threshold": float("nan")}}))) and raises(lambda: validate_config(
        _merge(DEFAULTS, {"analysis_root": "/data/runs", "analysis_name": "b0_full_ft"}))))
    shapes = [(1, 1), (5, 7), (30, 40)]
    rng = np.random.default_rng(7)
    masks = [m for shape in shapes for m in (np.zeros(shape, bool), np.ones(shape, bool), rng.random(shape) > 0.5)]
    reference_mask = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)
    reference_rle = {"size": [2, 3], "counts": [0, 1, 3, 1, 1]}
    check("Fortran-order RLE matches a manual SAM layout and round-trips empty/full/irregular masks",
          encode_rle(reference_mask) == reference_rle and np.array_equal(decode_rle(reference_rle), reference_mask)
          and all(np.array_equal(m, decode_rle(encode_rle(m))) for m in masks))
    check("RLE rejects wrong dimensions, negative runs and truncated coverage", all((
        raises(lambda: validate_rle({"size": [2, 2], "counts": [3]})),
        raises(lambda: validate_rle({"size": [2, 2], "counts": [5, -1]})),
        raises(lambda: validate_rle({"size": [2, 2], "counts": [4]}, (4, 1))),
    )))

    mask = np.zeros((10, 100), bool)
    mask.flat[0] = True
    exact_min = len(filtered(mask)["candidates"]) == 1
    mask.flat[:300] = True
    exact_max = len(filtered(mask)["candidates"]) == 1
    mask.flat[300] = True
    check("area boundaries 0.1% and 30% are inclusive; larger masks reject", exact_min and exact_max and filtered(mask)["counts"]["raw_area"] == 1)
    check("empty raw masks have an explicit rejection reason", filtered(np.zeros((20, 20), bool))["counts"]["empty_raw"] == 1)
    mask = np.zeros((20, 20), bool)
    mask[5:10, 5:10] = True
    gt = np.zeros(mask.shape, np.uint8)
    gt[5, 5] = 255
    check("one ignored/second-order pixel rejects the whole original proposal", filtered(mask, gt)["counts"]["ignore_overlap"] == 1)
    gt.fill(0)
    gt[5:10, 5:10] = 2
    check("direct interacting objects are rejected", filtered(mask, gt)["counts"]["foreground_overlap"] == 1)
    check("per-run binary target maps are not valid substitutes for the original remap", filtered(mask, gt)["counts"]["foreground_overlap"] == 1
          and filtered(mask, np.where(gt == 2, 0, gt))["counts"]["retained"] == 1)

    # 20/50 overlap with one small object; a distant 1000-pixel object hides it in union-only IoU.
    gt = np.zeros((100, 100), np.uint8)
    gt[0:20, 0:50] = 2
    gt[50:54, 50:55] = 2
    mask = np.zeros(gt.shape, bool)
    mask[50:55, 50:60] = True
    audit = filtered(mask, gt)["audit"][0]
    check("component IoU catches small-object overlap hidden by a large foreground union", audit["joint_foreground_iou"] < 0.3
          and audit["foreground_inside_fraction"] == 0.4 and audit["max_component_iou"] == 0.4 and audit["reason"] == "foreground_overlap")
    mask = np.zeros((20, 20), bool)
    mask[5, 5:15] = True
    gt = np.zeros(mask.shape, np.uint8)
    gt[5, 5:8] = 2
    exact_iou = filtered(mask, gt)
    check("IoU exactly 0.3 survives; foreground pixels are clipped before scoring", exact_iou["counts"]["retained"] == 1
          and exact_iou["candidates"][0]["area"] == 7 and not (decode_rle(exact_iou["candidates"][0]["segmentation"]) & (gt == 2)).any())
    relaxed = {**DEFAULTS["filter"], "foreground_iou_threshold": 1.0}
    gt[5, 5:10] = 2
    equal_inside = filtered(mask, gt, relaxed)["counts"]["retained"] == 1
    gt[5, 10] = 2
    check("inside fraction uses candidate area and rejects strictly above 0.5", equal_inside and filtered(mask, gt, relaxed)["counts"]["foreground_overlap"] == 1)
    gt[5, 5:15] = 2
    relaxed["foreground_inside_threshold"] = 1.0
    check("zero eligible pixels are explicitly rejected before division", filtered(mask, gt, relaxed)["counts"]["empty_eligible"] == 1)
    gt.fill(0)
    gt[5, 5] = 2
    min_after = {**relaxed, "min_area_fraction": 10 / gt.size}
    check("area is checked again after foreground clipping", filtered(mask, gt, min_after)["counts"]["eligible_area"] == 1)

    hands = np.zeros((10, 12), bool)
    hands[0, 0] = True
    mask = np.zeros(hands.shape, bool)
    mask[3, 4] = True
    check("distance is exact minimum Euclidean pixel distance, not centroid/bbox", hand_distance(mask, hands) == 5)
    mask[0, 1] = True
    check("adjacent and diagonal pixel distances are 1 and sqrt(2)", hand_distance(mask, hands) == 1
          and math.isclose(hand_distance(np.eye(10, 12, dtype=bool) & ~hands, hands), math.sqrt(2)))
    check("overlap gives zero; no hands is undefined; empty candidates raise", hand_distance(hands, hands) == 0
          and hand_distance(mask, np.zeros_like(hands)) is None and raises(lambda: hand_distance(np.zeros_like(mask), hands)))
    gt = hands.astype(np.uint8)
    mask = np.zeros_like(hands)
    mask[1, 1] = True
    result = filtered(mask, gt)
    check("normalisation uses the original image diagonal", math.isclose(result["candidates"][0]["distance"], math.sqrt(2) / math.hypot(10, 12)))
    mask = np.zeros_like(hands)
    mask[0, :5] = True
    clipped = filtered(mask, hands.astype(np.uint8))
    check("distance is measured after clipping: raw overlap zero becomes one-pixel adjacency", hand_distance(mask, hands) == 0
          and clipped["candidates"][0]["distance_px"] == 1)
    edges = DEFAULTS["distance_bin_edges"]
    check("bin endpoints are left-closed, final endpoint included; no hands has no bin", [distance_bin(x, edges) for x in (0, .01, .02, .05, .1, .2, 1, None)] == [0, 1, 2, 3, 4, 5, 5, None]
          and raises(lambda: distance_bin(-.01, edges)))

    # Exact 0.90 IoU is retained; 0.91 is suppressed, preferring higher quality.
    base = np.zeros((20, 20), bool)
    base.flat[:100] = True
    sub = np.zeros_like(base)
    sub.flat[:90] = True
    props = [proposal(base, 0, .9), proposal(sub, 1, .99)]
    a = filter_proposals(props, np.zeros(base.shape, np.uint8), DEFAULTS["filter"], edges)
    sub.flat[90] = True
    props[1] = proposal(sub, 1, .99)
    b = filter_proposals(props, np.zeros(base.shape, np.uint8), DEFAULTS["filter"], edges)
    check("mask NMS is strict above 0.9 and deterministically favours higher SAM quality", len(a["candidates"]) == 2
          and [c["mask_id"] for c in b["candidates"]] == [1] and b["audit"][0]["duplicate_of"] == 1)
    nested = np.zeros_like(base)
    nested.flat[:20] = True
    n = filter_proposals([proposal(base, 0), proposal(nested, 1)], np.zeros(base.shape, np.uint8), DEFAULTS["filter"], edges)
    check("distinct nested parts remain proposal-level regions", len(n["candidates"]) == 2)
    pair = np.zeros((2, 2), bool)
    pair[:] = True
    c = {"mask_id": 0, "segmentation": encode_rle(pair), "area": 4, "distance": 0.01, "distance_px": 1, "bin": 1}
    fractions = score_candidates([c], {"half": np.array([[2, 2], [0, 0]]), "more": np.array([[2, 2], [2, 1]])}, pair.shape)[0]["object_fractions"]
    check("scoring uses class 2, continuous fractions and candidate area", fractions == {"half": .5, "more": .75})
    check("raw binary stage-2 class 1 cannot be mistaken for final class 2", score_candidates([c], {"raw": np.ones((2, 2), np.uint8)}, pair.shape)[0]["object_fractions"]["raw"] == 0)

    entry = {"split": "test_indomain", "name": "ego4d_clip_0001", "width": 128, "height": 96,
             "image_sha256": "a" * 64, "label_sha256": "b" * 64, "display_source": "ego4d", "video": "ego4d_clip"}
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(analysis_root=str(tmp / "analysis"), cache_root=str(tmp / "cache"))
    manifest = {"proposal_fingerprint": "proposal-test", "analysis_fingerprint": "analysis-test"}
    rgb = np.full((96, 128, 3), 100, np.uint8)
    gt = np.zeros((96, 128), np.uint8)
    gt[10:20, 10:20] = 1
    m = np.zeros_like(gt, bool)
    m[20:30, 50:60] = True
    calls = []
    def fake_sam(image):
        calls.append(image.copy())
        return [proposal(m)], {"sam_seconds": 1.0, "gpu": "synthetic"}
    r, hit1 = obtain_proposals(cfg, manifest, entry, rgb, fake_sam)
    r2, hit2 = obtain_proposals(cfg, manifest, entry, rgb, fake_sam)
    check("expensive cache round-trip skips valid completed work and receives raw RGB uint8", not hit1 and hit2 and r == r2
          and len(calls) == 1 and calls[0].dtype == np.uint8 and np.array_equal(calls[0], rgb))
    check("changed proposal identity is never reused", compatible_record(proposal_path(cfg, manifest, entry), {"proposal_fingerprint": "other"})[1] == "incompatible")
    empty_entry = {**entry, "name": "ego4d_empty_0002"}
    obtain_proposals(cfg, manifest, empty_entry, rgb, lambda image: ([], {"sam_seconds": 1}))
    empty, hit = obtain_proposals(cfg, manifest, empty_entry, rgb, lambda image: (_ for _ in ()).throw(AssertionError("reran valid empty cache")))
    check("valid empty proposal caches are distinguishable from missing work", hit and empty["proposals"] == [] and cached_proposals(cfg, manifest, {**entry, "name": "missing"})[1] == "missing")
    p = proposal_path(cfg, manifest, entry)
    p.write_bytes(b"truncated gzip")
    check("corrupt/partial caches are refused", cached_proposals(cfg, manifest, entry)[1] == "invalid")
    obtain_proposals(cfg, manifest, entry, rgb, fake_sam)
    check("corrupt caches can be regenerated atomically", len(calls) == 2 and cached_proposals(cfg, manifest, entry)[1] == "valid")
    record_path = tmp / "sealed.json"
    sealed = write_record(record_path, {"value": 1})
    write_record(record_path, {**sealed, "value": 2})
    check("records can be resealed without carrying an old checksum", read_record(record_path)["value"] == 2)
    changed = json.loads(record_path.read_text())
    changed["value"] = 3
    record_path.write_bytes(canonical(changed))
    check("content corruption is detected even when JSON still parses", compatible_record(record_path, {})[1] == "invalid")
    counts = {m["name"]: 0 for m in cfg["models"]}
    def fake_models(image, target):
        for name in counts:
            counts[name] += 1
        return {m["name"]: np.where(np.indices(gt.shape)[1] % 2 == 0, 2, 0).astype(np.uint8) for m in cfg["models"]}
    measured = measure_entry(cfg, manifest, entry, rgb, gt, [proposal(m)], fake_models)
    validate_measurement(measured, manifest, entry, cfg)
    check("all four models share candidate IDs, pixels and distances", len(measured["candidates"]) == 1
          and set(measured["candidates"][0]["object_fractions"]) == set(counts) and set(counts.values()) == {1})
    empty_measured = measure_entry(cfg, manifest, empty_entry, rgb, gt, [], lambda *args: (_ for _ in ()).throw(AssertionError("unnecessary inference")))
    check("empty retained populations complete without model inference", empty_measured["candidates"] == [])
    all_entries = [{"split": split, "name": f"image_{i:05d}"} for split, size in DEFAULTS["splits"].items() for i in range(size)]
    groups = [owned_entries(all_entries, shard, 4) for shard in range(4)]
    combined = sum(groups, [])
    check("four deterministic shards cover exactly 1626 unique images", len(combined) == 1626 and len({(e["split"], e["name"]) for e in combined}) == 1626
          and [len(g) for g in groups] == [407, 407, 406, 406])
    rng.shuffle(all_entries)
    check("shard ownership is independent of input listing order", groups == [owned_entries(all_entries, shard, 4) for shard in range(4)])
    check("merge refuses missing, duplicated and unexpected processed images", raises(lambda: validate_coverage(combined, combined[:-1]))
          and raises(lambda: validate_coverage(combined, combined + combined[:1]))
          and raises(lambda: validate_coverage(combined, combined[:-1] + [{"split": "other", "name": "extra"}])) )

    # Different numbers of proposals per image expose candidate- versus image-weighting.
    row = {**measured, "candidates": [{**measured["candidates"][0], "object_fractions": {name: .5 for name in counts}}]}
    row2 = copy.deepcopy(row)
    row2.update(name="ego4d_second_0003", has_hands=False)
    row2["candidates"][0].update(distance=None, distance_px=None)
    row2["candidates"][0]["object_fractions"] = {name: .75 for name in counts}
    summary = aggregate([row, row2], cfg)["test_indomain"]
    check("FP events are strictly above 0.5; no-hand rows stay outside bins", sum(b["models"][next(iter(counts))]["fp_count"] for b in summary["bins"]) == 0
          and summary["overall_including_no_hands"][next(iter(counts))]["fp_count"] == 1 and summary["n_no_hand_candidates"] == 1)
    check("empty bins have undefined rates, not zero", any(b["n_candidates"] == 0 and b["models"][next(iter(counts))]["rate"] is None for b in summary["bins"]))
    second = copy.deepcopy(row)
    second["name"] = "ego4d_third_0004"
    second["candidates"] = [{**copy.deepcopy(row["candidates"][0]), "mask_id": i, "object_fractions": {name: 1. for name in counts}} for i in range(3)]
    s = aggregate([row, second], cfg)["test_indomain"]
    b = next(b for b in s["bins"] if b["n_candidates"])
    check("candidate rates and secondary per-image averages have distinct denominators", b["n_candidates"] == 4 and b["n_images"] == 2
          and b["models"][next(iter(counts))]["rate"] == .75 and b["models"][next(iter(counts))]["per_image_mean_rate"] == .5)

    io_and_render_checks(tmp, cfg, manifest, entry, empty_entry, rgb, gt, m, measured, empty_measured)
    interruption_checks(tmp, cfg, manifest, entry, empty_entry, rgb, gt, m)
    inference_checks(cfg, rgb, gt)
    lease_checks()
    print(f"\n{sum(RESULTS.values())}/{len(RESULTS)} checks passed. Synthetic outputs: {tmp}")
    if not all(RESULTS.values()):
        raise SystemExit(1)


def io_and_render_checks(tmp, cfg, manifest, entry, empty_entry, rgb, gt, mask, measured, empty_measured):
    from src.clutter_preview import render_pilot
    from phase6_report import build
    root = Path(cfg["analysis_root"]) / cfg["analysis_name"]
    inventory = [entry, empty_entry]
    full_manifest = {**manifest, "config": semantic_config(cfg), "inventory": inventory,
                     "models": {}, "versions": {}, "source_hashes": {}, "proposal_recipe": {}, "pilot_images": inventory}
    write_record(root / "manifest.json", full_manifest)
    write_record(root / "execution.json", {"shards": 2, "analysis_fingerprint": manifest["analysis_fingerprint"]})
    by_name = {entry["name"]: measured, empty_entry["name"]: empty_measured}
    for shard in range(2):
        for e in owned_entries(inventory, shard, 2):
            write_record(measurement_path(root, shard, e), by_name[e["name"]])
    with patch("src.clutter_workflow.load_manifest", return_value=full_manifest), patch("src.clutter_workflow.require_approval"):
        merged = merge_analysis(cfg)
        resumed = merge_analysis(cfg)
    check("merge/recovery preserves completed-empty images and reproduces summary exactly", merged == resumed and merged["n_images"] == 2)
    check("full analysis is blocked before explicit pilot approval", raises(lambda: require_approval(cfg, full_manifest), FileNotFoundError))
    pilot_images = [{**e, "preview_files": [], "preview_sha256": {}} for e in inventory]
    pilot = write_record(root / "pilot.json", {"analysis_fingerprint": manifest["analysis_fingerprint"], "images": pilot_images})
    with patch("src.clutter_workflow.load_manifest", return_value=full_manifest):
        approve_pilot(cfg)
    require_approval(cfg, full_manifest)
    write_record(root / "pilot.json", {**pilot, "changed": True})
    check("pilot approval is bound to the exact completed pilot record", raises(lambda: require_approval(cfg, full_manifest)))
    local = build(root, tmp / "report")
    check("local plot/report reproduce merged continuous measurements", local["splits"] == merged["splits"]
          and (tmp / "report/clutter_fp_vs_distance.png").stat().st_size > 10000)
    check("alternate thresholds need a separate output prefix", raises(lambda: build(root, tmp / "report", fp_threshold=.8)))
    alternate = build(root, tmp / "report", fp_threshold=.8, bin_edges=[0, .1, 1], prefix="sensitivity")
    check("thresholds and bins rebuild locally without SAM or prediction caches", len(alternate["splits"]["test_indomain"]["bins"]) == 2)
    (root / "workers/00/images/bulk.pt").write_bytes(b"not an output")
    selected = small_output_files(cfg)
    check("small-output fetching excludes bulk masks, weights and worker image records", "measurements.jsonl.gz" in selected
          and not any("/images/" in name or name.endswith(".pt") for name in selected))
    vis_gt = gt.copy()
    vis_gt[30:40, 20:40] = 2
    vis_gt[50:60, 90:100] = 255
    props = [proposal(mask, 0), proposal(vis_gt == 2, 1)]
    rejected = np.zeros_like(gt, bool)
    rejected[:50, :70] = True
    props.append(proposal(rejected, 2))
    props.append(proposal(vis_gt == 255, 3))
    result = filter_proposals(props, vis_gt, cfg["filter"], cfg["distance_bin_edges"])
    previews = render_pilot(entry, rgb, vis_gt, props, result)
    for name, data in previews.items():
        (tmp / name).write_bytes(data)
    check("pilot renderer produces overview plus retained/rejected detail pages", len(previews) >= 3
          and any("retained" in name for name in previews) and any("rejected" in name for name in previews))
    entries = []
    for source, size in (("ego4d", (1920, 1440)), ("epic", (342, 256)), ("thu", (640, 480)), ("own", (3840, 2160))):
        for i in range(4):
            entries.append({"split": "test_indomain", "name": f"{source}_{i}", "display_source": source,
                            "width": size[0], "height": size[1], "video": f"{source}_{i}"})
    for i in range(5):
        for j in range(2):
            entries.append({"split": "test_outdomain", "name": f"youtube_{i}_{j}", "display_source": "youtube",
                            "width": 1920, "height": 1080, "video": f"youtube_{i}"})
    picks = pick_pilot(entries, cfg)
    check("pilot selection is fixed, spans all sources and uses four distinct YouTube videos", picks == pick_pilot(list(reversed(entries)), cfg)
          and len(picks) == 10 and {e["display_source"] for e in picks} == {"ego4d", "epic", "thu", "own", "youtube"}
          and len({e["video"] for e in picks if e["split"] == "test_outdomain"}) == 4)


def inference_checks(cfg, rgb, gt):
    import torch
    import torch.nn.functional as F
    from src.cascade import cascade_predictions
    from src.clutter_workflow import SegmentationPredictor
    from src.data import EvalTransform
    torch.set_num_threads(2)
    class FakeModel(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.logits = torch.tensor(logits, dtype=torch.float32)[None]
        def forward(self, pixel_values):
            return SimpleNamespace(logits=self.logits)
    baseline = FakeModel([[[0, 0], [0, 0]], [[3, 0], [0, 0]], [[0, 2], [2, 0]]])
    stage1 = FakeModel([[[0, 3], [3, 3]], [[4, 0], [0, 0]]])
    stage2 = FakeModel([[[0, 0], [0, 0]], [[4, 4], [4, 4]]])
    predictor = SegmentationPredictor.__new__(SegmentationPredictor)
    predictor.cfg, predictor.device, predictor.stage1 = cfg, torch.device("cpu"), stage1
    predictor.transform = EvalTransform()
    predictor.models = {m["name"]: (baseline if m["kind"] == "baseline" else stage2, False) for m in cfg["models"]}
    with patch("src.cascade.cascade_predictions", wraps=cascade_predictions) as spy:
        predictions = predictor(rgb, gt)
    expected = F.interpolate(baseline.logits, gt.shape, mode="bilinear", align_corners=False).argmax(1)[0].numpy()
    check("baseline inference upsamples fp32 logits before original-resolution argmax", np.array_equal(predictions["b0_full_ft"], expected))
    check("cascade inference explicitly disables oracle and supplies no GT", spy.call_count == 2
          and all(c.kwargs["gt"] is None and c.kwargs["modes"] == ("predicted",) for c in spy.call_args_list))
    check("cascade scoring receives composed class 2 with hands pasted last", set(np.unique(predictions["b0_cascade_stage2"])) == {1, 2}
          and predictions["b0_cascade_stage2"][0, 0] == 1 and predictions["b0_cascade_stage2"][-1, -1] == 2)


def interruption_checks(tmp, original_cfg, original_manifest, entry, empty_entry, rgb, gt, mask):
    cfg = copy.deepcopy(original_cfg)
    cfg.update(analysis_name="interruption", cache_root=str(tmp / "interrupt_cache"))
    inventory = [entry, empty_entry]
    manifest = {**original_manifest, "inventory": inventory}
    root = Path(cfg["analysis_root"]) / cfg["analysis_name"]
    write_record(root / "execution.json", {"shards": 2, "analysis_fingerprint": manifest["analysis_fingerprint"]})
    counts = {"sam": 0, "predict": 0}
    class Proposer:
        def __init__(self, *args):
            pass
        def __call__(self, image):
            counts["sam"] += 1
            return ([proposal(mask)] if image[0, 0, 0] == 100 else []), {"sam_seconds": 1.}
        def close(self):
            pass
    class Predictor:
        def __init__(self, *args):
            pass
        def __call__(self, image, target):
            counts["predict"] += 1
            return {m["name"]: np.zeros(gt.shape, np.uint8) for m in cfg["models"]}
    class Interrupted(Exception):
        pass
    def interrupt_at_commit(n):
        calls = [0]
        def commit():
            calls[0] += 1
            if calls[0] == n:
                raise Interrupted()
        return commit
    def original(e, unused):
        return (rgb if e["name"] == entry["name"] else np.full_like(rgb, 101)), gt
    with patch("src.clutter_workflow.load_manifest", return_value=manifest), patch("src.clutter_workflow.require_approval"), \
         patch("src.clutter_workflow.load_original", side_effect=original), \
         patch("src.clutter_workflow.SamProposer", Proposer), patch("src.clutter_workflow.SegmentationPredictor", Predictor):
        # Choose the shard with the nonempty image, independent of alphabetical names.
        shard = next(i for i in range(2) if owned_entries(inventory, i, 2)[0]["name"] == entry["name"])
        first = raises(lambda: run_shard(cfg, shard, 2, interrupt_at_commit(1)), Interrupted)
        second = raises(lambda: run_shard(cfg, shard, 2, interrupt_at_commit(2)), Interrupted)
        run_shard(cfg, shard, 2)
        check("worker resumes across SAM and measurement interruptions without repeating completed work", first and second and counts == {"sam": 1, "predict": 1})
        run_shard(cfg, 1-shard, 2)
        run_shard(cfg, 1-shard, 2)
        check("worker resume preserves valid empty results without SAM/inference reruns", counts == {"sam": 2, "predict": 1})
        check("workers reject a changed ownership plan", raises(lambda: run_shard(cfg, 0, 3)))


def lease_checks():
    class Store(dict):
        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self:
                return False
            self[key] = value
            return True
    store = Store()
    with writer_lease(store, "analysis", "full", "call-1") as token:
        require_writer(store, token)
        blocked = raises(lambda: enter_other_lease(store), RuntimeError)
    check("atomic writer lease excludes overlapping launches and releases on success", blocked and not store)
    try:
        with writer_lease(store, "analysis", "full", "call-2"):
            raise RuntimeError("interrupted")
    except RuntimeError:
        pass
    check("interrupted full launches retain their lease; recovery needs explicit stopped confirmation", "writer" in store
          and raises(lambda: recover_writer(store, "analysis")) and raises(lambda: require_writer(store, "wrong"), RuntimeError))
    recover_writer(store, "analysis", confirm_stopped=True)
    check("confirmed recovery permits reuse of completed work without stale ownership", not store)


def enter_other_lease(store):
    with writer_lease(store, "analysis", "full", "call-other"):
        pass


if __name__ == "__main__":
    main()
