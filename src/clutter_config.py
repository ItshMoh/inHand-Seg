"""Validated Phase-6 configuration, independent of the training config schema."""

import copy
import json
import math
import re
from pathlib import Path

import yaml

SAM_REVISION = "dca509fe793f601edb92606367a655c15ac00fdf"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DEFAULTS = {
    "analysis_name": "clutter_sam_v1",
    "data_root": "/data/egohos",
    "analysis_root": "/data/analysis",
    "cache_root": "/data/clutter_cache/sam",
    "splits": {"test_indomain": 1126, "test_outdomain": 500},
    "stage1_checkpoint": "/data/runs/b0_hands_only/model_final.pt",
    "models": [
        {"name": "b0_full_ft", "kind": "baseline", "seed": 0,
         "checkpoint": "/data/runs/b0_full_ft/model_final.pt"},
        {"name": "b0_full_ft_s1", "kind": "baseline", "seed": 1,
         "checkpoint": "/data/runs/b0_full_ft_s1/model_final.pt"},
        {"name": "b0_cascade_stage2", "kind": "cascade", "seed": 0,
         "checkpoint": "/data/runs/b0_cascade_stage2/model_final.pt"},
        {"name": "b0_cascade_stage2_s1", "kind": "cascade", "seed": 1,
         "checkpoint": "/data/runs/b0_cascade_stage2_s1/model_final.pt"},
    ],
    "sam": {
        "variant": "vit_h", "revision": SAM_REVISION, "checkpoint_url": SAM_URL,
        "checkpoint": "/data/checkpoints/sam/sam_vit_h_4b8939.pth", "precision": "fp32",
        "settings": {
            "points_per_side": 32, "points_per_batch": 32, "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95, "stability_score_offset": 1.0,
            "box_nms_thresh": 0.7, "crop_n_layers": 0, "crop_nms_thresh": 0.7,
            "crop_overlap_ratio": 512 / 1500, "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0, "output_mode": "uncompressed_rle",
        },
    },
    "filter": {
        "min_area_fraction": 0.001, "max_area_fraction": 0.30,
        "foreground_iou_threshold": 0.30, "foreground_inside_threshold": 0.50,
        "reject_any_ignore": True, "component_connectivity": 8,
        "mask_nms_iou_threshold": 0.90,
    },
    "distance_bin_edges": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0],
    "fp_threshold": 0.50,
    "pilot": {"seed": 0, "indomain_images": 6, "outdomain_videos": 4},
    "execution": {
        "shards": 4, "worker_timeout_seconds": 7200, "commit_every": 5,
        "preview_page_size": 4, "fetch_max_bytes": 536870912,
    },
}


def _merge(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown analysis config field: {prefix}{key}")
        out[key] = _merge(base[key], value, f"{prefix}{key}.") if isinstance(base[key], dict) else value
    return out


def validate_config(cfg):
    """Reject unknown fields, wrong types, nonfinite numbers and unsupported protocols."""
    def types(value, template, prefix=""):
        if isinstance(template, dict):
            if not isinstance(value, dict) or set(value) != set(template):
                raise ValueError(f"{prefix or 'config'} has missing/unknown fields")
            for key in template:
                types(value[key], template[key], f"{prefix}{key}.")
        elif isinstance(template, bool):
            if type(value) is not bool:
                raise ValueError(f"{prefix} must be boolean")
        elif isinstance(template, int):
            if type(value) is not int:
                raise ValueError(f"{prefix} must be an integer")
        elif isinstance(template, float):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{prefix} must be finite numeric")
        elif not isinstance(value, type(template)):
            raise ValueError(f"{prefix} must be {type(template).__name__}")
    types(cfg, DEFAULTS)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", cfg["analysis_name"]):
        raise ValueError("analysis_name must be a safe name of at most 64 characters")
    if cfg["splits"] != DEFAULTS["splits"]:
        raise ValueError("Phase 6 covers exactly the two original test splits (1126 + 500)")
    for key in ("data_root", "analysis_root", "cache_root", "stage1_checkpoint"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    if cfg["sam"]["variant"] != "vit_h" or cfg["sam"]["precision"] != "fp32":
        raise ValueError("The agreed SAM protocol is original ViT-H in fp32")
    if cfg["sam"]["revision"] != SAM_REVISION or cfg["sam"]["checkpoint_url"] != SAM_URL:
        raise ValueError("Use the agreed official SAM revision and checkpoint URL")
    if not Path(cfg["sam"]["checkpoint"]).is_absolute():
        raise ValueError("SAM checkpoint must be absolute")
    models = cfg["models"]
    if len(models) != 4 or any(not isinstance(m, dict) or set(m) != {"name", "kind", "seed", "checkpoint"} for m in models):
        raise ValueError("models must contain the four explicitly specified model records")
    if len({m["name"] for m in models}) != 4 or {(m["kind"], m["seed"]) for m in models} != {
        ("baseline", 0), ("baseline", 1), ("cascade", 0), ("cascade", 1)
    }:
        raise ValueError("Compare baseline and predicted cascade at both seeds 0 and 1")
    for m in models:
        if (type(m["seed"]) is not int or not isinstance(m["name"], str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", m["name"])
                or not isinstance(m["checkpoint"], str) or not Path(m["checkpoint"]).is_absolute()):
            raise ValueError("Invalid model name, seed or checkpoint")
    if len({m["checkpoint"] for m in models}) != 4:
        raise ValueError("Models must use four distinct final checkpoints")
    # Never let an analysis/cache output overlap datasets or completed model runs.
    destinations = [Path(cfg["analysis_root"]) / cfg["analysis_name"], Path(cfg["cache_root"])]
    if any(p == Path("/data/runs") or Path("/data/runs") in p.parents for p in destinations):
        raise ValueError("New analysis/cache outputs must stay outside /data/runs")
    protected = [Path(cfg["data_root"]), *(Path(m["checkpoint"]).parent for m in models),
                 Path(cfg["stage1_checkpoint"]).parent, Path(cfg["sam"]["checkpoint"]).parent]
    if any(a == b or a in b.parents or b in a.parents for a in destinations for b in protected):
        raise ValueError("Analysis/cache directories must be separate from data and checkpoint directories")
    a, b = destinations
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("Analysis and proposal cache roots must be separate")
    f = cfg["filter"]
    if not 0 <= f["min_area_fraction"] <= f["max_area_fraction"] <= 1:
        raise ValueError("Invalid area fractions")
    for key in ("foreground_iou_threshold", "foreground_inside_threshold", "mask_nms_iou_threshold"):
        if not 0 <= f[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if f["reject_any_ignore"] is not True or f["component_connectivity"] != 8:
        raise ValueError("The agreed policy rejects any ignore overlap and uses 8-connected components")
    edges = cfg["distance_bin_edges"]
    if (len(edges) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in edges)
            or edges[0] != 0 or edges[-1] != 1 or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError("Distance bin edges must increase strictly from 0 to 1")
    if not 0 <= cfg["fp_threshold"] <= 1:
        raise ValueError("fp_threshold must be in [0,1]")
    s = cfg["sam"]["settings"]
    for key in ("points_per_side", "points_per_batch", "crop_n_points_downscale_factor"):
        if s[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ("pred_iou_thresh", "stability_score_thresh", "box_nms_thresh", "crop_nms_thresh", "crop_overlap_ratio"):
        if not 0 <= s[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if (s["crop_n_layers"] != 0 or s["min_mask_region_area"] != 0
            or s["output_mode"] != "uncompressed_rle" or s["stability_score_offset"] < 0):
        raise ValueError("Use no crop layers or region cleanup and uncompressed_rle output")
    if cfg["pilot"] != DEFAULTS["pilot"]:
        raise ValueError("The agreed fixed pilot is six in-domain images and four YouTube videos, seed 0")
    ex = cfg["execution"]
    if (not 1 <= ex["shards"] <= 4 or not 60 <= ex["worker_timeout_seconds"] <= 7200
            or not 1 <= ex["commit_every"] <= 100 or not 1 <= ex["preview_page_size"] <= 6
            or ex["fetch_max_bytes"] < 1024):
        raise ValueError("Invalid execution bounds (at most four workers, two hours each)")
    return cfg


def load_config(path):
    return validate_config(_merge(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {}))


def semantic_config(cfg):
    """Execution bounds may change after the timed pilot without changing measurements."""
    return {k: v for k, v in cfg.items() if k != "execution"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
