"""Phase-6 preparation, pilot, streaming inference and recovery.

Heavy dependencies are imported inside the operations that need them. This module
never trains a model or calls the segmentation evaluation writers.
"""

import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import random
import resource
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.clutter_config import canonical, semantic_config, validate_config
from src.clutter_sam import (
    SCHEMA, aggregate, atomic_bytes, compatible_record, file_sha256, filter_proposals,
    fingerprint, image_identity, image_key, owned_entries, read_record, score_candidates,
    validate_coverage, validate_proposals, write_record,
)


def analysis_dir(cfg):
    return Path(cfg["analysis_root"]) / cfg["analysis_name"]


def source_hashes():
    return {name: file_sha256(Path(__file__).with_name(name)) for name in
            ("clutter_config.py", "clutter_sam.py", "clutter_workflow.py", "clutter_preview.py", "clutter_lock.py",
             "data.py", "model.py", "evaluate.py", "cascade.py")}


def versions():
    names = ("torch", "torchvision", "transformers", "numpy", "pillow", "scipy",
             "albumentations", "opencv-python-headless", "segment-anything")
    return {name: importlib.metadata.version(name) for name in names}


def check_sam_install():
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from torchvision.ops.boxes import batched_nms
    import torch
    from src.clutter_config import SAM_REVISION

    dist = importlib.metadata.distribution("segment-anything")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id", "")
    if revision != SAM_REVISION and SAM_REVISION not in direct.get("url", ""):
        raise ValueError("SAM installation must identify the agreed official source revision")
    if "vit_h" not in sam_model_registry:
        raise ValueError("Original SAM ViT-H registry entry missing")
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
    assert batched_nms(boxes, torch.tensor([0.9, 0.8]), torch.zeros(2, dtype=torch.int64), 0.7).tolist() == [0]
    return direct


def _checkpoint_config(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not {"model", "model_config", "config"} <= ckpt.keys():
        raise ValueError(f"Expected an offline-reloadable final checkpoint: {path}")
    return ckpt["config"], ckpt["model_config"]


def checkpoint_provenance(cfg):
    records = {}
    for m in cfg["models"]:
        p = Path(m["checkpoint"])
        saved, arch = _checkpoint_config(p)
        is_cascade = m["kind"] == "cascade"
        wanted_classes = ["background", "interacting_object"] if is_cascade else ["background", "hand", "interacting_object"]
        wanted_map = {1: 0, 2: 1} if is_cascade else None
        if (saved["seed"] != m["seed"] or saved["run_name"] != m["name"]
                or saved["model"]["class_names"] != wanted_classes
                or saved["data"].get("label_map") != wanted_map
                or bool(saved.get("cascade", {}).get("enabled")) != is_cascade
                or arch.get("num_channels", 3) != (4 if is_cascade else 3)):
            raise ValueError(f"Checkpoint/config mismatch for {m['name']}")
        if is_cascade and saved["cascade"]["stage1_checkpoint"] != cfg["stage1_checkpoint"]:
            raise ValueError("Both cascade seeds must use the agreed frozen stage 1")
        records[m["name"]] = {**m, "sha256": file_sha256(p), "bytes": p.stat().st_size,
                                "run_config": saved, "architecture": arch}
    stage1 = Path(cfg["stage1_checkpoint"])
    saved, arch = _checkpoint_config(stage1)
    if (saved["model"]["class_names"] != ["background", "hand"] or saved["seed"] != 0
            or saved["data"].get("label_map") != {2: 0} or arch.get("num_channels", 3) != 3):
        raise ValueError("Stage 1 must be the saved seed-0 hands-only model")
    records["stage1"] = {"checkpoint": str(stage1), "sha256": file_sha256(stage1),
                         "run_config": saved, "architecture": arch}
    return records


def prepare_sam_checkpoint(cfg):
    """Only the user-run prepare command downloads. Partial files are never usable caches."""
    import torch
    p = Path(cfg["sam"]["checkpoint"])
    sidecar = p.with_suffix(p.suffix + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if not p.exists():
        part = p.with_suffix(p.suffix + ".part")
        print(f"Downloading official ViT-H weights to {part} (interrupted downloads restart)", flush=True)
        with urllib.request.urlopen(cfg["sam"]["checkpoint_url"], timeout=120) as response, part.open("wb") as output:
            expected_length = response.headers.get("Content-Length")
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if expected_length is not None and part.stat().st_size != int(expected_length):
            raise ValueError("Incomplete SAM checkpoint download")
        candidate = part
        downloaded = True
    else:
        candidate = p
    digest = file_sha256(candidate)
    if sidecar.exists() and read_record(sidecar)["sha256"] != digest:
        raise ValueError("Existing SAM checkpoint differs from its recorded SHA-256; inspect it explicitly")
    weights = torch.load(candidate, map_location="cpu", weights_only=True)
    if (not isinstance(weights, dict) or "image_encoder.blocks.31.norm1.weight" not in weights
            or tuple(weights["image_encoder.pos_embed"].shape) != (1, 64, 64, 1280)):
        raise ValueError("Checkpoint does not have the official SAM ViT-H architecture")
    del weights
    gc.collect()
    if downloaded:
        os.replace(candidate, p)
    info = {"path": str(p), "sha256": digest, "bytes": p.stat().st_size,
            "source_url": cfg["sam"]["checkpoint_url"],
            "integrity_note": "Observed SHA-256, not an independently published upstream checksum"}
    if not sidecar.exists():
        write_record(sidecar, info)
    return info


def build_inventory(cfg):
    from src.data import EgoHOSDataset
    from src.evaluate import display_source, video_of

    tasks = []
    for split, expected in cfg["splits"].items():
        ds = EgoHOSDataset(split, cfg["data_root"])
        if set(ds.images) != set(ds.labels) or len(ds) != expected:
            raise ValueError(f"{split}: expected {expected} fully paired original images")
        tasks.extend((split, name, ds.images[name], ds.labels[name]) for name in ds.names)

    def inspect(task):
        split, name, image, label = task
        with Image.open(image) as im, Image.open(label) as lb:
            if im.size != lb.size:
                raise ValueError(f"Unaligned image/label: {name}")
            width, height = im.size
        return {"split": split, "name": name, "width": width, "height": height,
                "image_sha256": file_sha256(image), "label_sha256": file_sha256(label),
                "display_source": display_source(name), "video": video_of(name)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return sorted(pool.map(inspect, tasks), key=lambda e: (e["split"], e["name"]))


def pick_pilot(entries, cfg):
    """Six in-domain, including both area extremes and every source; four distinct YouTube videos."""
    rng = random.Random(cfg["pilot"]["seed"])
    inside = sorted((e for e in entries if e["split"] == "test_indomain"),
                    key=lambda e: (e["width"] * e["height"], e["name"]))
    picked = [inside[0], inside[-1]]
    for source in ("ego4d", "epic", "thu", "own"):
        if not any(e["display_source"] == source for e in picked):
            pool = sorted((e for e in inside if e["display_source"] == source), key=lambda e: e["name"])
            if not pool:
                raise ValueError(f"Pilot is missing source {source}")
            picked.append(rng.choice(pool))
    rest = sorted((e for e in inside if e not in picked), key=lambda e: e["name"])
    picked.extend(rng.sample(rest, cfg["pilot"]["indomain_images"] - len(picked)))
    videos = {}
    for e in entries:
        if e["split"] == "test_outdomain":
            videos.setdefault(e["video"], []).append(e)
    for video in rng.sample(sorted(videos), cfg["pilot"]["outdomain_videos"]):
        picked.append(rng.choice(sorted(videos[video], key=lambda e: e["name"])))
    return picked


def prepare_analysis(cfg, commit=lambda: None):
    validate_config(cfg)
    install = check_sam_install()
    sam = prepare_sam_checkpoint(cfg)
    commit()  # Keep the validated large download even if later inventory checks fail.
    model_info = checkpoint_provenance(cfg)
    inventory = build_inventory(cfg)
    runtime = versions()
    proposal_recipe = {"schema": SCHEMA, "sam": cfg["sam"], "checkpoint_sha256": sam["sha256"],
                       "versions": {k: runtime[k] for k in ("torch", "torchvision", "numpy", "pillow", "opencv-python-headless", "segment-anything")},
                       "sam_install": install, "tf32": False, "seed": 0,
                       "rgb_input": "original_HWC_uint8_ignore_EXIF"}
    proposal_id = fingerprint(proposal_recipe)
    identity = {"config": semantic_config(cfg), "inventory": inventory, "models": model_info,
                "proposal_fingerprint": proposal_id, "versions": runtime, "source_hashes": source_hashes()}
    manifest = {**identity, "analysis_fingerprint": fingerprint(identity), "proposal_recipe": proposal_recipe,
                "sam_checkpoint": sam, "pilot_images": pick_pilot(inventory, cfg),
                "prepared_execution": cfg["execution"],
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = analysis_dir(cfg) / "manifest.json"
    if path.exists():
        old = read_record(path)
        if old["analysis_fingerprint"] != manifest["analysis_fingerprint"]:
            raise FileExistsError("Analysis name already contains a different recipe, data, code or checkpoints. Choose a new analysis_name; SAM proposals can still be reused.")
        return old
    result = write_record(path, manifest)
    commit()
    return result


def load_manifest(cfg, verify_environment=True):
    manifest = read_record(analysis_dir(cfg) / "manifest.json")
    if manifest["config"] != semantic_config(cfg) or manifest["source_hashes"] != source_hashes():
        raise ValueError("Config or analysis code changed since prepare; use a new analysis_name")
    if verify_environment and manifest["versions"] != versions():
        raise ValueError("Runtime dependency versions changed since prepare")
    return manifest


@lru_cache(maxsize=4)
def _dataset(split, root):
    from src.data import EgoHOSDataset
    ds = EgoHOSDataset(split, root)
    return ds, {name: i for i, name in enumerate(ds.names)}


def load_original(entry, cfg):
    """Check the frozen source identities before loading original RGB and the original remap."""
    from src.data import remap_label
    ds, index = _dataset(entry["split"], cfg["data_root"])
    if (file_sha256(ds.images[entry["name"]]) != entry["image_sha256"]
            or file_sha256(ds.labels[entry["name"]]) != entry["label_sha256"]):
        raise ValueError(f"Image/GT changed after prepare: {entry['name']}")
    rgb, original = ds.load(index[entry["name"]])
    if rgb.shape != (entry["height"], entry["width"], 3) or not np.isin(original, np.arange(9)).all():
        raise ValueError("Invalid original image/label coordinates or IDs")
    return rgb, remap_label(original)


def proposal_path(cfg, manifest, entry):
    # The image hash prevents a changed JPEG from overwriting that image's previous proposal cache.
    return (Path(cfg["cache_root"]) / manifest["proposal_fingerprint"] / entry["split"] /
            f"{image_key(entry)}_{entry['image_sha256'][:16]}.json.gz")


def cached_proposals(cfg, manifest, entry):
    record, state = compatible_record(proposal_path(cfg, manifest, entry), {
        "proposal_fingerprint": manifest["proposal_fingerprint"], "image": image_identity(entry)})
    if record is not None:
        try:
            validate_proposals(record["proposals"], (entry["height"], entry["width"]))
        except (ValueError, KeyError, TypeError):
            return None, "invalid"
    return record, state


class SamProposer:
    def __init__(self, cfg, manifest):
        self.cfg, self.manifest, self.generator = cfg, manifest, None
        self.previous_tf32 = None

    def __call__(self, rgb):
        import torch
        if self.generator is None:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            if file_sha256(self.cfg["sam"]["checkpoint"]) != self.manifest["sam_checkpoint"]["sha256"]:
                raise ValueError("SAM weights changed since prepare")
            check_sam_install()
            self.previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.manual_seed(0)
            self.generator = SamAutomaticMaskGenerator(
                sam_model_registry["vit_h"](checkpoint=self.cfg["sam"]["checkpoint"]).cuda().eval().requires_grad_(False),
                **self.cfg["sam"]["settings"],
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            proposals = self.generator.generate(rgb)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        proposals = [{**p, "mask_id": i, "area": int(p["area"])} for i, p in enumerate(proposals)]
        return proposals, {"sam_seconds": seconds, "gpu": torch.cuda.get_device_name(),
                           "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                           "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                           "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

    def close(self):
        import torch
        self.generator = None
        if self.previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.previous_tf32
            self.previous_tf32 = None
        gc.collect()
        torch.cuda.empty_cache()


def obtain_proposals(cfg, manifest, entry, rgb, proposer):
    record, state = cached_proposals(cfg, manifest, entry)
    if record is not None:
        return record, True
    if state != "missing":
        print(f"Rebuilding {state} proposal cache: {entry['name']}", flush=True)
    proposals, timing = proposer(rgb)
    validate_proposals(proposals, rgb.shape[:2])
    record = write_record(proposal_path(cfg, manifest, entry), {
        "proposal_fingerprint": manifest["proposal_fingerprint"], "image": image_identity(entry),
        "proposals": proposals, "timing": timing})
    return record, False


def pilot_analysis(cfg, commit=lambda: None):
    from src.clutter_preview import render_pilot
    manifest = load_manifest(cfg)
    root = analysis_dir(cfg)
    if (root / "pilot_approval.json").exists():
        raise ValueError("Pilot is already approved and frozen. Fetch its existing previews instead.")
    proposer, images = SamProposer(cfg, manifest), []
    started = time.perf_counter()
    try:
        for i, entry in enumerate(manifest["pilot_images"]):
            rgb, gt = load_original(entry, cfg)
            before = time.perf_counter()
            record, hit = obtain_proposals(cfg, manifest, entry, rgb, proposer)
            filtered = filter_proposals(record["proposals"], gt, cfg["filter"], cfg["distance_bin_edges"])
            files = render_pilot(entry, rgb, gt, record["proposals"], filtered, cfg["execution"]["preview_page_size"])
            for rel, content in files.items():
                atomic_bytes(root / "pilot" / rel, content)
            images.append({**entry, "filter_counts": filtered["counts"], "has_hands": filtered["has_hands"],
                           "audit": filtered["audit"], "timing": record["timing"], "cache_hit": hit,
                           "proposal_masks_sha256": fingerprint(record["proposals"]),
                           "eligible_masks_sha256": fingerprint(filtered["candidates"]),
                           "cache_bytes": proposal_path(cfg, manifest, entry).stat().st_size,
                           "processing_seconds": time.perf_counter() - before, "preview_files": sorted(files),
                           "preview_sha256": {rel: hashlib.sha256(content).hexdigest() for rel, content in files.items()}})
            write_record(root / "pilot_progress.json", {"analysis_fingerprint": manifest["analysis_fingerprint"],
                         "completed": len(images), "total": len(manifest["pilot_images"])})
            commit()
            print(f"[pilot {i+1}/10] {entry['display_source']} masks={len(record['proposals'])} retained={len(filtered['candidates'])} SAM={record['timing']['sam_seconds']:.2f}s cache_hit={hit}", flush=True)
    finally:
        proposer.close()
    samples = [r["timing"]["sam_seconds"] for r in images]
    # A deliberately conservative pilot projection, not a measured full-run duration.
    n = len(manifest["inventory"])
    estimate = {"pilot_median_sam_seconds": float(np.median(samples)), "pilot_max_sam_seconds": max(samples),
                "median_projection_sam_gpu_hours": n * float(np.median(samples)) / 3600,
                "max_projection_sam_gpu_hours": n * max(samples) / 3600,
                "four_worker_max_projection_sam_minutes": math.ceil(n / 4) * max(samples) / 60,
                "note": "SAM-only projections from 10 samples; exclude startup, download, CPU filtering, previews, segmentation inference and storage. Cache hits reuse the original GPU/time observations. Choose full timeout after review."}
    pilot = write_record(root / "pilot.json", {"analysis_fingerprint": manifest["analysis_fingerprint"],
                         "images": images, "n_images": len(images), "estimates": estimate,
                         "pilot_wall_seconds": time.perf_counter() - started})
    _pilot_index(root, pilot)
    commit()
    return {"n_images": len(images), "estimates": estimate, "output": str(root / 'pilot')}


def _pilot_index(root, pilot):
    import html
    lines = ["# Phase-6 candidate pilot", "", "Review every image and the paginated proposal sheets before approving.",
             "Overview: RGB | GT (blue hands, yellow interacting objects, grey hatched ignore) | retained C | rejected raw proposals.",
             "Detail sheets: one row per mask ID; raw proposal at left, background-only pixels at right. Green denotes retained C; red denotes rejected background pixels (unscored). Magenta contours show the original proposal.",
             "Rejection reasons are the first failing test in the documented order. Nested masks/parts/surfaces may remain; these are proposal regions, not independent instances.", ""]
    cards = []
    for im in pilot["images"]:
        lines += [f"## {im['split']} / {im['name']}", "", f"Counts: `{json.dumps(im['filter_counts'], sort_keys=True)}`", ""]
        for rel in im["preview_files"]:
            lines += [f"![{rel}]({rel})", ""]
            cards.append(f'<figure><figcaption>{html.escape(im["name"])} / {html.escape(rel)}</figcaption><img loading="lazy" src="{html.escape(rel)}" style="max-width:100%"></figure>')
    atomic_bytes(root / "pilot" / "README.md", "\n".join(lines).encode())
    intro = "Review the overview and every proposal sheet. GT: blue hands, yellow objects, grey hatched ignore. Magenta: original proposal. Green: retained background region. Red: rejected background region (unscored). IDs refer to the cached SAM proposals."
    atomic_bytes(root / "pilot" / "index.html", ('<!doctype html><meta charset="utf-8"><title>Phase 6 pilot</title><body style="font-family:system-ui;max-width:1500px;margin:2em auto"><h1>Candidate pilot</h1><p>' + intro + '</p>' + ''.join(cards)).encode())


def approve_pilot(cfg, commit=lambda: None):
    manifest = load_manifest(cfg)
    root = analysis_dir(cfg)
    pilot = read_record(root / "pilot.json")
    if pilot["analysis_fingerprint"] != manifest["analysis_fingerprint"]:
        raise ValueError("Pilot and manifest differ")
    validate_coverage(manifest["pilot_images"], pilot["images"])
    for image in pilot["images"]:
        if set(image["preview_files"]) != set(image["preview_sha256"]):
            raise ValueError("Pilot preview manifest is incomplete")
        for rel, digest in image["preview_sha256"].items():
            if file_sha256(root / "pilot" / rel) != digest:
                raise ValueError(f"Pilot preview changed after rendering: {rel}")
    result = write_record(root / "pilot_approval.json", {
        "analysis_fingerprint": manifest["analysis_fingerprint"], "pilot_sha256": pilot["payload_sha256"],
        "approval": "User explicitly ran approve_pilot after inspecting candidate previews",
        "approved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    commit()
    return result


def require_approval(cfg, manifest):
    root = analysis_dir(cfg)
    approval = read_record(root / "pilot_approval.json")
    pilot = read_record(root / "pilot.json")
    if (approval["analysis_fingerprint"] != manifest["analysis_fingerprint"]
            or approval["pilot_sha256"] != pilot["payload_sha256"]):
        raise ValueError("Pilot approval does not match this analysis")
    return approval


class SegmentationPredictor:
    """Load final weights offline once; stream predictions without saving dense masks."""
    def __init__(self, cfg, manifest, device="cuda"):
        import torch
        from src.cascade import load_stage1
        from src.data import EvalTransform
        from src.model import load_model
        self.cfg, self.device, self.models = cfg, torch.device(device), {}
        self.transform = EvalTransform()
        for name, info in manifest["models"].items():
            if file_sha256(info["checkpoint"]) != info["sha256"]:
                raise ValueError(f"Final weights changed: {name}")
        self.stage1 = load_stage1(cfg["stage1_checkpoint"], device)
        for m in cfg["models"]:
            model, ckpt = load_model(m["checkpoint"])
            self.models[m["name"]] = (model.to(device).requires_grad_(False).eval(), ckpt["config"]["train"]["bf16"])

    def __call__(self, rgb, gt):
        import torch
        import torch.nn.functional as F
        from src.cascade import cascade_predictions
        transformed, _ = self.transform(rgb, gt)
        image = torch.from_numpy(np.ascontiguousarray(transformed.transpose(2, 0, 1), dtype=np.float32))[None].to(self.device)
        predictions = {}
        with torch.inference_mode():
            for m in self.cfg["models"]:
                model, bf16 = self.models[m["name"]]
                if m["kind"] == "baseline":
                    with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
                        logits = model(pixel_values=image).logits
                    pred = F.interpolate(logits.float(), size=gt.shape, mode="bilinear", align_corners=False).argmax(1)[0]
                else:
                    pred = cascade_predictions(self.stage1, model, image, gt.shape, gt=None,
                                               modes=("predicted",), bf16=bf16)["predicted"]["prediction"]
                predictions[m["name"]] = pred.to(torch.uint8).cpu().numpy()
        return predictions


def measurement_expected(manifest, entry):
    return {"analysis_fingerprint": manifest["analysis_fingerprint"], "image": image_identity(entry),
            "label_sha256": entry["label_sha256"]}


def measurement_path(root, shard, entry):
    return root / "workers" / f"{shard:02d}" / "images" / f"{image_key(entry)}.json.gz"


def measure_entry(cfg, manifest, entry, rgb, gt, proposals, predictor):
    filtered = filter_proposals(proposals, gt, cfg["filter"], cfg["distance_bin_edges"])
    # Empty results are completed work, including images where every proposal was rejected.
    predictions = predictor(rgb, gt) if filtered["candidates"] else {}
    candidates = score_candidates(filtered["candidates"], predictions, gt.shape)
    return {**measurement_expected(manifest, entry), **{k: entry[k] for k in
            ("split", "name", "display_source", "video", "width", "height")},
            "has_hands": filtered["has_hands"], "filter_counts": filtered["counts"],
            "audit": filtered["audit"], "candidates": candidates,
            "proposal_masks_sha256": fingerprint(proposals),
            "eligible_masks_sha256": fingerprint(filtered["candidates"])}


def validate_measurement(record, manifest, entry, cfg):
    if any(record.get(k) != v for k, v in measurement_expected(manifest, entry).items()):
        raise ValueError("Incompatible measurement identity")
    if (record.get("split"), record.get("name")) != (entry["split"], entry["name"]):
        raise ValueError("Incompatible measurement image")
    if record["filter_counts"]["retained"] != len(record["candidates"]):
        raise ValueError("Inconsistent retained candidate count")
    if len(record["audit"]) != sum(record["filter_counts"].values()):
        raise ValueError("Incomplete filtering audit")
    retained = {r["mask_id"]: r["eligible_area"] for r in record["audit"] if r["reason"] == "retained"}
    if retained != {c["mask_id"]: c["area"] for c in record["candidates"]}:
        raise ValueError("Candidate population differs from the filtering audit")
    for key in ("proposal_masks_sha256", "eligible_masks_sha256"):
        if not isinstance(record.get(key), str) or len(record[key]) != 64:
            raise ValueError("Missing actual proposal/eligible-mask provenance")
    aggregate([record], cfg)  # Validates all continuous fractions and bin/no-hand consistency.


def cached_measurement(cfg, manifest, entry, destination):
    saved, _ = compatible_record(destination, measurement_expected(manifest, entry))
    if saved is not None:
        try:
            validate_measurement(saved, manifest, entry, cfg)
            proposals, _ = cached_proposals(cfg, manifest, entry)
            if proposals is not None and saved["proposal_masks_sha256"] != fingerprint(proposals["proposals"]):
                return None
        except (ValueError, KeyError, TypeError):
            return None
    return saved


def run_shard(cfg, shard, shards, commit=lambda: None):
    manifest = load_manifest(cfg)
    require_approval(cfg, manifest)
    entries = owned_entries(manifest["inventory"], shard, shards)
    root = analysis_dir(cfg)
    state_path = root / "workers" / f"{shard:02d}" / "progress.json"
    plan = read_record(root / "execution.json")
    if plan["shards"] != shards or plan["analysis_fingerprint"] != manifest["analysis_fingerprint"]:
        raise ValueError("Worker ownership does not match the committed execution plan")
    proposer = SamProposer(cfg, manifest)
    start = time.perf_counter()

    def progress(stage, count, **extra):
        write_record(state_path, {"analysis_fingerprint": manifest["analysis_fingerprint"], "shard": shard,
                     "shards": shards, "stage": stage, "completed": count, "total": len(entries),
                     "elapsed_seconds": time.perf_counter() - start,
                     "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})
        if count % cfg["execution"]["commit_every"] == 0 or count == len(entries):
            commit()
            print(f"[shard {shard}/{shards} {stage}] {count}/{len(entries)} elapsed={time.perf_counter()-start:.1f}s", flush=True)

    try:
        for i, entry in enumerate(entries):
            measured = cached_measurement(cfg, manifest, entry, measurement_path(root, shard, entry))
            if measured is not None:
                progress("sam", i + 1, skipped_completed_measurement=True)
                continue
            cached, _ = cached_proposals(cfg, manifest, entry)
            if cached is None:
                rgb, _ = load_original(entry, cfg)
                obtain_proposals(cfg, manifest, entry, rgb, proposer)
            progress("sam", i + 1)
    finally:
        proposer.close()
    predictor = None
    try:
        for i, entry in enumerate(entries):
            destination = measurement_path(root, shard, entry)
            saved = cached_measurement(cfg, manifest, entry, destination)
            if saved is None:
                rgb, gt = load_original(entry, cfg)
                proposals, state = cached_proposals(cfg, manifest, entry)
                if proposals is None:
                    raise ValueError(f"Proposal cache disappeared/became {state} before measurement")
                def predict(image, target):
                    nonlocal predictor
                    if predictor is None:
                        predictor = SegmentationPredictor(cfg, manifest)
                    return predictor(image, target)
                measured = measure_entry(cfg, manifest, entry, rgb, gt, proposals["proposals"], predict)
                validate_measurement(measured, manifest, entry, cfg)
                write_record(destination, measured)
            progress("measure", i + 1)
        progress("complete", len(entries))
    finally:
        predictor = None
        gc.collect()
    return {"shard": shard, "n_images": len(entries), "seconds": time.perf_counter() - start}


def merge_analysis(cfg, commit=lambda: None):
    manifest = load_manifest(cfg)
    require_approval(cfg, manifest)
    root = analysis_dir(cfg)
    plan = read_record(root / "execution.json")
    records, expected_paths = [], set()
    for shard in range(plan["shards"]):
        for entry in owned_entries(manifest["inventory"], shard, plan["shards"]):
            path = measurement_path(root, shard, entry)
            expected_paths.add(path)
            record = read_record(path)
            validate_measurement(record, manifest, entry, cfg)
            records.append(record)
    actual_paths = set((root / "workers").glob("*/images/*.json.gz"))
    if actual_paths != expected_paths:
        raise ValueError("Unexpected/duplicated worker image files; refusing an ambiguous merge")
    validate_coverage(manifest["inventory"], records)
    records.sort(key=lambda r: (r["split"], r["name"]))
    combined = gzip.compress(b"\n".join(canonical(r) for r in records) + b"\n", mtime=0)
    atomic_bytes(root / "measurements.jsonl.gz", combined)
    result = write_record(root / "summary.json", {"analysis_fingerprint": manifest["analysis_fingerprint"],
                          "n_images": len(records), "splits": aggregate(records, cfg),
                          "measurements_sha256": file_sha256(root / "measurements.jsonl.gz"),
                          "coverage": "exact inventory match; unique (split,name), no missing or extra worker files"})
    commit()
    return result


def status_analysis(cfg):
    root = analysis_dir(cfg)
    result = {"analysis_name": cfg["analysis_name"], "prepared": (root / "manifest.json").exists(),
              "pilot_complete": (root / "pilot.json").exists(), "pilot_approved": (root / "pilot_approval.json").exists(),
              "merged": (root / "summary.json").exists(), "workers": []}
    for p in sorted((root / "workers").glob("*/progress.json")):
        result["workers"].append(read_record(p))
    if (root / "pilot.json").exists():
        result["pilot_estimates"] = read_record(root / "pilot.json")["estimates"]
    if (root / "execution.json").exists():
        result["execution"] = read_record(root / "execution.json")
    return result


def small_output_files(cfg):
    """Explicit allowlist. Never walk the SAM cache or fetch per-image worker files."""
    root = analysis_dir(cfg)
    selected = [root / name for name in ("manifest.json", "pilot.json", "pilot_progress.json",
                "pilot_approval.json", "execution.json", "summary.json", "measurements.jsonl.gz")]
    if (root / "pilot.json").exists():
        pilot = read_record(root / "pilot.json")
        selected += [root / "pilot" / rel for image in pilot["images"] for rel in image["preview_files"]]
    else:
        selected += sorted((root / "pilot").glob("*.png"))
    selected += [root / "pilot" / name for name in ("README.md", "index.html")]
    selected += sorted((root / "workers").glob("*/progress.json"))
    selected = [p for p in selected if p.is_file()]
    total = sum(p.stat().st_size for p in selected)
    if total > cfg["execution"]["fetch_max_bytes"]:
        raise ValueError(f"Small-output fetch would exceed configured limit: {total} bytes")
    return {str(p.relative_to(root)): p.read_bytes() for p in selected}
