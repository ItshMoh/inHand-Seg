"""Deterministic exploratory shortlists and explicitly reviewed gallery cases."""

from collections import Counter

from src.clutter_sam import fingerprint, image_key
from src.phase7_config import FAILURE_TAGS, REGION_KINDS, TAGS


def key(row):
    return row["split"], row["name"]


def clutter_hints(measurements, cfg):
    hints = []
    for row in measurements:
        for c in row["candidates"]:
            f = c["object_fractions"]
            b, p = ([f[r] for r in cfg[group]] for group in ("baseline_runs", "cascade_runs"))
            event = ("cascade_only" if all(v <= .5 for v in b) and all(v > .5 for v in p)
                     else "baseline_only" if all(v > .5 for v in b) and all(v <= .5 for v in p) else None)
            if event:
                hints.append({"split": row["split"], "name": row["name"], **c, "event": event,
                              "threshold_margin": min(abs(v-.5) for v in b+p)})
    return sorted(hints, key=lambda h: (-h["threshold_margin"], h["split"], h["name"], h["mask_id"]))


def make_request(manifest, rows, measurements):
    return {"analysis_fingerprint": manifest["analysis_fingerprint"], "manifest": manifest,
            "images": rows, "clutter_hints": clutter_hints(measurements, manifest["config"])}


def choose_shortlist(request, gt):
    """Round-robin diagnostic buckets after four initial slots per source.

    Ranking is post-evaluation diagnosis, not an unbiased performance sample.
    GT-dependent buckets use annotations only; all named appearance tags remain manual.
    """
    rows, cfg = request["images"], request["manifest"]["config"]
    by_key = {key(r): r for r in rows}
    if len(by_key) != len(rows) or set(gt) != set(by_key):
        raise ValueError("Shortlist needs exact, unique CPU-label coverage")
    eligible = [r for r in rows if r["paired"]["interacting_object"]["eligible"]]

    def worst(r):
        p = r["paired"]["interacting_object"]
        return min(p["baseline_mean"], p["cascade_mean"]), *key(r)

    ranked = sorted(eligible, key=worst)
    buckets = {}
    for src in sorted({r["source"] for r in rows}):
        buckets[f"source:{src}"] = [r for r in ranked if r["source"] == src]
    buckets["worst_object_iou"] = ranked
    for positive, label in ((True, "paired_improvement"), (False, "paired_regression")):
        pool = [r for r in eligible if (r["paired"]["interacting_object"]["mean_delta"] > 0 if positive
                                      else r["paired"]["interacting_object"]["mean_delta"] < 0)]
        buckets[label] = sorted(pool, key=lambda r: (
            not r["paired"]["interacting_object"]["same_direction"],
            (-1 if positive else 1)*r["paired"]["interacting_object"]["mean_delta"], *key(r)))
    buckets["small_component_annotation"] = [r for r in ranked if any(c["small"] for c in gt[key(r)]["object_components"])]
    buckets["two_hand_id5_annotation"] = [r for r in ranked if gt[key(r)]["two_hand_pixels"] > 0]
    for event in ("cascade_only", "baseline_only"):
        buckets[f"clutter:{event}"] = [by_key[key(h)] for h in request["clutter_hints"] if h["event"] == event]

    picked, videos, reasons = [], Counter(), {}
    def add(row, reason):
        k = key(row)
        if k in reasons:
            return False
        if videos[(row["source"], row["diversity_video"])] >= cfg["gallery"]["max_per_video"]:
            return False
        if len(picked) >= cfg["gallery"]["shortlist_images"]:
            return False
        picked.append(row)
        reasons[k] = [reason]
        videos[(row["source"], row["diversity_video"])] += 1
        return True
    # Give each source real support before filling diagnostic buckets.
    for reason, pool in buckets.items():
        if reason.startswith("source:"):
            taken = 0
            for row in pool:
                if add(row, reason):
                    taken += 1
                if taken == 4:
                    break
    active = {reason: iter(pool) for reason, pool in buckets.items() if not reason.startswith("source:")}
    while active and len(picked) < cfg["gallery"]["shortlist_images"]:
        for reason in list(active):
            for row in active[reason]:
                if add(row, reason):
                    break
            else:
                del active[reason]
    return [{**r, "image_id": image_key(r), "selection_reasons": reasons[key(r)],
             "gt": gt[key(r)], "candidate_hints": [h for h in request["clutter_hints"] if key(h) == key(r)]}
            for r in picked]


def review_template(request, shortlist):
    return {"analysis_fingerprint": request["analysis_fingerprint"],
            "instructions": "Set selected/preview, tags, evidence per tag and an evidence-based caption after visual review. No appearance category is assigned automatically.",
            "categories": {tag: {"status": "pending", "note": ""} for tag in FAILURE_TAGS},
            "cases": [{"image_id": r["image_id"], "split": r["split"], "name": r["name"],
                       "selected": False, "preview": False, "tags": [], "evidence": {}, "caption": "",
                       "candidate_id": None, "region_kind": "uncertain", "crop_xyxy": None}
                      for r in shortlist]}


def build_selection(request, shortlist, review):
    cfg = request["manifest"]["config"]
    if review.get("analysis_fingerprint") != request["analysis_fingerprint"]:
        raise ValueError("Review belongs to a different analysis")
    if set(review.get("categories", {})) != set(FAILURE_TAGS):
        raise ValueError("Review every named failure category")
    lookup = {r["image_id"]: r for r in shortlist}
    if len(lookup) != len(shortlist):
        raise ValueError("Image-ID collision")
    cases, seen = [], set()
    for c in review["cases"]:
        if c["image_id"] in seen or c["image_id"] not in lookup:
            raise ValueError("Unknown/duplicate reviewed image")
        seen.add(c["image_id"])
        row = lookup[c["image_id"]]
        if key(row) != key(c) or type(c["selected"]) is not bool or type(c["preview"]) is not bool:
            raise ValueError("Review identity or selection flags invalid")
        if c["preview"] and not c["selected"]:
            raise ValueError("Preview must be a selected case")
        if not c["selected"]:
            continue
        if (not isinstance(c["tags"], list) or len(c["tags"]) != len(set(c["tags"]))
                or not set(c["tags"]) <= set(TAGS) or set(c["evidence"]) != set(c["tags"])
                or any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in c["evidence"].values())
                or not isinstance(c["caption"], str) or not c["caption"].strip() or len(c["caption"]) > 1500):
            raise ValueError("Selected cases need valid tags, per-tag evidence and a caption")
        if "small_object" in c["tags"] and not any(v["small"] for v in row["gt"]["object_components"]):
            raise ValueError("Small-object tag lacks annotation geometry")
        if "two_hand_object" in c["tags"] and row["gt"]["two_hand_pixels"] <= 0:
            raise ValueError("Two-hand tag requires original ID 5 plus visual review")
        p = row["paired"]["interacting_object"]
        for tag, sign in (("improvement", 1), ("regression", -1)):
            if tag in c["tags"] and (not p["eligible"] or sign*p["mean_delta"] <= 0 or sign*p["per_seed_delta"][0] <= 0):
                raise ValueError("Improvement/regression tag contradicts displayed seed or mean delta")
        candidate = None
        if c["candidate_id"] is not None:
            if type(c["candidate_id"]) is not int:
                raise ValueError("Candidate ID must be an integer")
            candidate = next((h for h in row["candidate_hints"] if h["mask_id"] == c["candidate_id"]), None)
            if candidate is None or "clutter_diagnostic" not in c["tags"] or c["region_kind"] not in REGION_KINDS:
                raise ValueError("SAM case needs an existing disagreement ID and reviewed region kind")
        elif "clutter_diagnostic" in c["tags"]:
            raise ValueError("Clutter diagnostic requires the exact SAM candidate ID")
        crop = c["crop_xyxy"]
        if crop is not None:
            w, h = row["size"]
            if (not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop)
                    or not 0 <= crop[0] < crop[2] <= w or not 0 <= crop[1] < crop[3] <= h):
                raise ValueError("Invalid original-coordinate crop")
        case = {k: v for k, v in row.items() if k not in ("candidate_hints", "image_id")}
        case.update({k: c[k] for k in ("tags", "evidence", "caption", "preview", "region_kind", "crop_xyxy")})
        case["candidate"] = candidate
        case["case_id"] = image_key(row) + (f"_sam{candidate['mask_id']}" if candidate else "")
        case["case_fingerprint"] = fingerprint({"analysis": request["analysis_fingerprint"], "case": case})
        cases.append(case)
    if seen != set(lookup):
        raise ValueError("Review dropped or added shortlist images")
    selection = {"analysis_fingerprint": request["analysis_fingerprint"], "cases": cases,
                 "categories": review["categories"], "quantitative_population": "full frozen test splits",
                 "qualitative_population": "deliberately selected exploratory cases; no prevalence estimate"}
    validate_selection(selection, request)
    return selection


def validate_selection(selection, request, final=False):
    cfg = request["manifest"]["config"]
    cases = selection["cases"]
    if (selection["analysis_fingerprint"] != request["analysis_fingerprint"]
            or not 4 <= len(cases) <= cfg["gallery"]["max_cases"]
            or sum(c["preview"] for c in cases) != cfg["gallery"]["preview_cases"]
            or len({key(c) for c in cases}) != len(cases)
            or len({c["case_id"] for c in cases}) != len(cases)):
        raise ValueError("Invalid selection identity/counts; choose exactly four previews")
    videos = Counter((c["source"], c["diversity_video"]) for c in cases)
    if any(n > cfg["gallery"]["max_per_video"] for n in videos.values()):
        raise ValueError("Selection exceeds video diversity cap")
    if final and {c["source"] for c in cases} != {r["source"] for r in request["images"]}:
        raise ValueError("The final gallery must cover all five available sources")
    for c in cases:
        payload = {k: v for k, v in c.items() if k != "case_fingerprint"}
        if c["case_fingerprint"] != fingerprint({"analysis": request["analysis_fingerprint"], "case": payload}):
            raise ValueError("Case content changed without rebuilding the selection")
    if set(selection["categories"]) != set(FAILURE_TAGS):
        raise ValueError("Missing failure-category review")
    for tag, v in selection["categories"].items():
        count = sum(tag in c["tags"] for c in cases)
        if v["status"] not in ("pending", "supported", "unsupported", "uncertain") or not isinstance(v["note"], str):
            raise ValueError("Invalid category review status")
        if v["status"] == "supported" and count == 0:
            raise ValueError("Supported category has no selected example")
        if v["status"] in ("unsupported", "uncertain") and (count or not v["note"].strip()):
            raise ValueError("Unsupported/uncertain categories need an explanation and no confirmed tags")
        if final and v["status"] == "pending":
            raise ValueError("Resolve each category honestly before full-gallery rendering")
    return selection
