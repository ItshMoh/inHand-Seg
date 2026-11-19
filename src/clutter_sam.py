"""Phase-6 CPU arithmetic and compact caches. No torch or SAM imports are needed here.

Candidates are proposal-level regions, not independently annotated object instances.
All masks and distance calculations use original image coordinates.
"""

import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.clutter_config import canonical

SCHEMA = 1
REASONS = ("empty_raw", "raw_area", "ignore_overlap", "foreground_overlap", "empty_eligible",
           "eligible_area", "duplicate", "retained")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_key(entry):
    return fingerprint([entry["split"], entry["name"]])[:24]


def image_identity(entry):
    return {k: entry[k] for k in ("split", "name", "width", "height", "image_sha256")}


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_record(path, value):
    """Checksum plus an explicit completion marker, including valid empty results."""
    value = {k: v for k, v in value.items() if k != "payload_sha256"}
    value.update(schema=SCHEMA, complete=True)
    value["payload_sha256"] = fingerprint(value)
    raw = canonical(value)
    atomic_bytes(path, gzip.compress(raw, mtime=0) if str(path).endswith(".gz") else raw)
    return value


def read_record(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    value = json.loads(raw)
    checksum = value.pop("payload_sha256")
    if (value.get("schema") != SCHEMA or value.get("complete") is not True
            or checksum != fingerprint(value)):
        raise ValueError(f"Invalid/incomplete record: {path}")
    value["payload_sha256"] = checksum
    return value


def compatible_record(path, expected):
    """Return (record, reason). Missing, corrupt and incompatible files are never cache hits."""
    try:
        value = read_record(path)
        if any(value.get(k) != v for k, v in expected.items()):
            return None, "incompatible"
        return value, "valid"
    except FileNotFoundError:
        return None, "missing"
    except (ValueError, KeyError, TypeError, OSError, EOFError):
        return None, "invalid"


def encode_rle(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("RLE requires a nonempty HW array (an all-false mask is valid)")
    flat = mask.astype(bool).ravel(order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, changes, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def validate_rle(rle, shape=None):
    if not isinstance(rle, dict) or set(rle) != {"size", "counts"}:
        raise ValueError("Expected uncompressed SAM RLE")
    size, counts = rle["size"], rle["counts"]
    if (not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
            or (shape is not None and tuple(size) != tuple(shape))
            or not isinstance(counts, list) or not counts
            or any(type(v) is not int or v < 0 for v in counts) or sum(counts) != math.prod(size)):
        raise ValueError("Invalid RLE dimensions or run lengths")
    return sum(counts[1::2])


def decode_rle(rle):
    validate_rle(rle)
    values = np.arange(len(rle["counts"]), dtype=np.int64) % 2 == 1
    return np.repeat(values, rle["counts"]).reshape(rle["size"], order="F")


def validate_proposals(proposals, shape):
    if not isinstance(proposals, list):
        raise ValueError("proposals must be a list")
    for i, p in enumerate(proposals):
        if p.get("mask_id") != i or p.get("area") != validate_rle(p["segmentation"], shape):
            raise ValueError("SAM mask IDs/areas must match the cached RLEs")
        for key in ("predicted_iou", "stability_score"):
            if type(p.get(key)) not in (int, float) or not math.isfinite(p[key]):
                raise ValueError(f"Invalid SAM quality: {key}")


def distance_bin(distance, edges):
    if distance is None:
        return None
    if not math.isfinite(distance) or not edges[0] <= distance <= edges[-1]:
        raise ValueError("Distance outside configured bin range")
    return min(int(np.searchsorted(edges, distance, side="right") - 1), len(edges) - 2)


def hand_distance(mask, hands):
    """Exact Euclidean pixel-centre distance; no-hand results are undefined."""
    mask, hands = np.asarray(mask, bool), np.asarray(hands, bool)
    if mask.shape != hands.shape or mask.ndim != 2 or not mask.any():
        raise ValueError("Distance requires a nonempty candidate and aligned HW hand mask")
    if not hands.any():
        return None
    return float(distance_transform_edt(~hands)[mask].min())


def _bbox(mask):
    yy, xx = np.nonzero(mask)
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] if xx.size else None


def _iou(intersection, area_a, area_b):
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _overlap_components(gt):
    regions = []
    for cls in (1, 2):
        count, labels = cv2.connectedComponents((gt == cls).astype(np.uint8), connectivity=8)
        areas = np.bincount(labels.ravel(), minlength=count)
        regions.append((labels, areas))
    return regions


def _packed_crop(mask):
    box = _bbox(mask)
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1]
    return box, crop.shape, np.packbits(crop)


def _intersection_packed(mask, packed):
    (x0, y0, x1, y1), shape, bits = packed
    return int((mask[y0:y1, x0:x1] & np.unpackbits(bits, count=math.prod(shape)).reshape(shape).astype(bool)).sum())


def filter_proposals(proposals, gt, cfg, edges):
    """Return audit rows plus retained RLEs. First rejection reason follows the agreed order.

    IoU is tested against the joint foreground and every separate hand/object component.
    The inside denominator is the ORIGINAL candidate area. Final areas, distances and
    deduplication use C = candidate & (GT == background). No component splitting follows.
    """
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.isin(gt, [0, 1, 2, 255]).all():
        raise ValueError("Filtering needs the original three-class remap, including 255")
    validate_proposals(proposals, gt.shape)
    total = gt.size
    foreground = (gt == 1) | (gt == 2)
    fg_area = int(foreground.sum())
    components = _overlap_components(gt)
    rows, pending = [], []

    def area_ok(area):
        return cfg["min_area_fraction"] * total <= area <= cfg["max_area_fraction"] * total

    for p in proposals:
        m = decode_rle(p["segmentation"])
        area = p["area"]
        row = {"mask_id": p["mask_id"], "raw_area": area, "raw_area_fraction": area / total,
               "predicted_iou": p["predicted_iou"], "stability_score": p["stability_score"],
               "reason": None, "eligible_area": None, "distance_px": None, "distance": None,
               "bin": None, "duplicate_of": None}
        rows.append(row)
        if not area:
            row["reason"] = "empty_raw"
            continue
        if not area_ok(area):
            row["reason"] = "raw_area"
            continue
        row["ignore_pixels"] = int((m & (gt == 255)).sum())
        if row["ignore_pixels"]:
            row["reason"] = "ignore_overlap"
            continue
        inside = int((m & foreground).sum())
        row["foreground_pixels"] = inside
        row["foreground_inside_fraction"] = inside / area
        row["joint_foreground_iou"] = _iou(inside, area, fg_area)
        max_component = 0.0
        for labels, areas in components:
            intersections = np.bincount(labels[m], minlength=len(areas))[1:]
            if intersections.size:
                unions = area + areas[1:] - intersections
                max_component = max(max_component, float((intersections / unions).max()))
        row["max_component_iou"] = max_component
        if (max(row["joint_foreground_iou"], max_component) > cfg["foreground_iou_threshold"]
                or row["foreground_inside_fraction"] > cfg["foreground_inside_threshold"]):
            row["reason"] = "foreground_overlap"
            continue
        eligible = m & (gt == 0)
        row["eligible_area"] = int(eligible.sum())
        row["clipped_pixels"] = area - row["eligible_area"]
        if not row["eligible_area"]:
            row["reason"] = "empty_eligible"
            continue
        if not area_ok(row["eligible_area"]):
            row["reason"] = "eligible_area"
            continue
        pending.append((p, row, encode_rle(eligible)))

    # Rank without looking at any segmentation model's predictions.
    pending.sort(key=lambda item: (-item[0]["predicted_iou"], -item[0]["stability_score"], item[0]["mask_id"]))
    kept, packed = [], []
    distance_map = distance_transform_edt(gt != 1) if (gt == 1).any() else None
    diagonal = math.hypot(*gt.shape)
    for p, row, rle in pending:
        m = decode_rle(rle)
        box = _bbox(m)
        for previous, crop in packed:
            # Area ratio bounds IoU; disjoint bounding boxes need no mask decoding.
            if min(row["eligible_area"], previous["area"]) / max(row["eligible_area"], previous["area"]) <= cfg["mask_nms_iou_threshold"]:
                continue
            b = crop[0]
            if box[0] >= b[2] or b[0] >= box[2] or box[1] >= b[3] or b[1] >= box[3]:
                continue
            inter = _intersection_packed(m, crop)
            if _iou(inter, row["eligible_area"], previous["area"]) > cfg["mask_nms_iou_threshold"]:
                row.update(reason="duplicate", duplicate_of=previous["mask_id"])
                break
        if row["reason"]:
            continue
        px = float(distance_map[m].min()) if distance_map is not None else None
        distance = px / diagonal if px is not None else None
        row.update(reason="retained", distance_px=px, distance=distance, bin=distance_bin(distance, edges))
        candidate = {"mask_id": p["mask_id"], "segmentation": rle, "area": row["eligible_area"],
                     "bbox_xyxy": box, "distance_px": px, "distance": distance, "bin": row["bin"]}
        kept.append(candidate)
        packed.append((candidate, _packed_crop(m)))
    counts = Counter(r["reason"] for r in rows)
    return {"audit": rows, "candidates": sorted(kept, key=lambda c: c["mask_id"]),
            "counts": {reason: counts[reason] for reason in REASONS}, "has_hands": distance_map is not None}


def score_candidates(candidates, predictions, shape):
    """Persist continuous fractions; thresholds and distance bins are applied by aggregation."""
    for pred in predictions.values():
        if pred.shape != tuple(shape) or not np.isin(pred, [0, 1, 2]).all():
            raise ValueError("Use original-resolution, composed three-class predictions")
    rows = []
    for candidate in candidates:
        mask = decode_rle(candidate["segmentation"])
        area = int(mask.sum())
        if area <= 0 or area != candidate["area"]:
            raise ValueError("Cannot score an empty or inconsistent eligible mask")
        rows.append({k: v for k, v in candidate.items() if k != "segmentation"} | {
            "object_fractions": {name: int((mask & (pred == 2)).sum()) / area for name, pred in predictions.items()},
        })
    return rows


def owned_entries(entries, shard, shards):
    if type(shards) is not int or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Invalid shard index/count")
    entries = sorted(entries, key=lambda e: (e["split"], e["name"]))
    keys = [(e["split"], e["name"]) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate images in inventory")
    return entries[shard::shards]


def validate_coverage(entries, records):
    expected = {(e["split"], e["name"]) for e in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate inventory entries")
    actual = [(r["split"], r["name"]) for r in records]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate processed images")
    if set(actual) != expected:
        raise ValueError(f"Coverage mismatch: missing={len(expected - set(actual))}, unexpected={len(set(actual) - expected)}")


def aggregate(records, cfg):
    """Each seed has the SAME candidate denominator. No-hand rows remain outside distance bins."""
    names = [m["name"] for m in cfg["models"]]
    edges, threshold = cfg["distance_bin_edges"], cfg["fp_threshold"]
    result = {}
    for split in cfg["splits"]:
        images = [r for r in records if r["split"] == split]
        bins = [{"lower": a, "upper": b, "n_candidates": 0, "n_images": 0,
                 "models": {name: {"fp_count": 0, "rate": None, "per_image_mean_rate": None} for name in names}}
                for a, b in zip(edges, edges[1:])]
        sources, counts, no_hand_candidates, total_candidates = {}, Counter(), 0, 0
        overall = {name: 0 for name in names}
        image_means = [{name: [] for name in names} for _ in bins]
        for image in images:
            counts.update(image["filter_counts"])
            source = sources.setdefault(image["display_source"], {"n_images": 0, "n_proposals": 0,
                                       "n_candidates": 0, "n_distance_candidates": 0, "n_no_hand_images": 0})
            source["n_images"] += 1
            source["n_proposals"] += sum(image["filter_counts"].values())
            source["n_candidates"] += len(image["candidates"])
            source["n_no_hand_images"] += int(not image["has_hands"])
            by_bin = [[] for _ in bins]
            seen_ids = set()
            for c in image["candidates"]:
                if c["mask_id"] in seen_ids or c["area"] <= 0 or set(c["object_fractions"]) != set(names):
                    raise ValueError("Invalid candidate IDs, areas or compared model population")
                seen_ids.add(c["mask_id"])
                fractions = c["object_fractions"]
                if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in fractions.values()):
                    raise ValueError("Invalid continuous object fraction")
                total_candidates += 1
                for name in names:
                    overall[name] += int(fractions[name] > threshold)
                index = distance_bin(c["distance"], edges)
                if (index is None) != (not image["has_hands"]):
                    raise ValueError("Distance/no-hand inconsistency")
                if index is None:
                    no_hand_candidates += 1
                else:
                    by_bin[index].append(c)
                    source["n_distance_candidates"] += 1
            for i, candidates in enumerate(by_bin):
                if not candidates:
                    continue
                bins[i]["n_images"] += 1
                bins[i]["n_candidates"] += len(candidates)
                for name in names:
                    events = sum(c["object_fractions"][name] > threshold for c in candidates)
                    bins[i]["models"][name]["fp_count"] += events
                    image_means[i][name].append(events / len(candidates))
        for i, b in enumerate(bins):
            for name, m in b["models"].items():
                m["rate"] = m["fp_count"] / b["n_candidates"] if b["n_candidates"] else None
                m["per_image_mean_rate"] = float(np.mean(image_means[i][name])) if image_means[i][name] else None
            b["systems"] = {}
            for kind in ("baseline", "cascade"):
                rates = [b["models"][m["name"]]["rate"] for m in cfg["models"] if m["kind"] == kind]
                b["systems"][kind] = {"mean_rate": float(np.mean(rates)) if all(v is not None for v in rates) else None,
                                      "seed_min": min(rates) if rates[0] is not None else None,
                                      "seed_max": max(rates) if rates[0] is not None else None}
        if sum(b["n_candidates"] for b in bins) + no_hand_candidates != total_candidates:
            raise ValueError("Candidate bin coverage mismatch")
        result[split] = {"n_images": len(images), "n_no_hand_images": sum(not r["has_hands"] for r in images),
                         "n_empty_images": sum(not r["candidates"] for r in images),
                         "n_candidates": total_candidates, "n_no_hand_candidates": no_hand_candidates,
                         "filter_counts": dict(counts), "sources": sources, "bins": bins,
                         "overall_including_no_hands": {name: {"fp_count": n,
                             "rate": n / total_candidates if total_candidates else None} for name, n in overall.items()}}
    return result
