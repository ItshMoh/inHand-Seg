"""Strict, independent Phase-7 schema. No training configuration is changed."""

import copy
import math
import re
from pathlib import Path

import yaml

DEFAULTS = {
    "analysis_name": "phase7_v1",
    "analysis_root": "/data/analysis",
    "phase6_name": "clutter_sam_v1",
    "phase6_fingerprint": "029ee11ad0192b08535327cfd58fe5ea2c4dafe03bdf2004e569f887c4b6f36c",
    "splits": {"test_indomain": 1126, "test_outdomain": 500},
    "baseline_runs": ["b0_full_ft", "b0_full_ft_s1"],
    "cascade_runs": ["b0_cascade_stage2", "b0_cascade_stage2_s1"],
    "gallery": {
        "display_seed": 0, "preview_cases": 4, "max_cases": 20,
        "shortlist_images": 60, "max_per_video": 2,
        "small_component_fraction": 0.01, "closeup_max_frame_fraction": 0.6,
    },
    "execution": {
        "cpu_timeout_seconds": 1800, "gpu_timeout_seconds": 1800,
        "scan_commit_every": 50, "fetch_max_bytes": 268435456,
    },
}

FAILURE_TAGS = ("transparent", "finger_occlusion", "motion_blur", "small_object", "two_hand_object", "lighting")
TAGS = FAILURE_TAGS + ("improvement", "regression", "clutter_diagnostic", "ambiguous")
REGION_KINDS = ("object_like", "surface", "object_part", "clothing", "uncertain")


def validate_config(cfg):
    def visit(value, template, path="config"):
        if isinstance(template, dict):
            if not isinstance(value, dict) or set(value) != set(template):
                raise ValueError(f"{path}: missing/unknown fields")
            for key in template:
                visit(value[key], template[key], f"{path}.{key}")
        elif type(template) is int:
            if type(value) is not int:
                raise ValueError(f"{path}: expected integer")
        elif type(template) is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path}: expected finite number")
        elif type(value) is not type(template):
            raise ValueError(f"{path}: wrong type")
    visit(cfg, DEFAULTS)
    for key in ("analysis_name", "phase6_name"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", cfg[key]):
            raise ValueError(f"Unsafe {key}")
    if cfg["analysis_root"] != "/data/analysis" or cfg["analysis_name"] == cfg["phase6_name"]:
        raise ValueError("Use a separate Phase-7 directory under /data/analysis")
    if not re.fullmatch(r"[0-9a-f]{64}", cfg["phase6_fingerprint"]):
        raise ValueError("Invalid Phase-6 fingerprint")
    for key in ("splits", "baseline_runs", "cascade_runs"):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError(f"The agreed Phase-7 population fixes {key}")
    g, ex = cfg["gallery"], cfg["execution"]
    if (g["display_seed"] != 0 or g["preview_cases"] != 4 or not 4 <= g["max_cases"] <= 20
            or not g["max_cases"] <= g["shortlist_images"] <= 60 or g["max_per_video"] != 2
            or not 0 < g["small_component_fraction"] <= 0.01
            or not 0 < g["closeup_max_frame_fraction"] < 1):
        raise ValueError("Gallery limits differ from the agreed bounded recipe")
    if (not 60 <= ex["cpu_timeout_seconds"] <= 1800 or not 60 <= ex["gpu_timeout_seconds"] <= 1800
            or not 1 <= ex["scan_commit_every"] <= 100 or not 1024 <= ex["fetch_max_bytes"] <= 268435456):
        raise ValueError("Invalid execution bounds")
    return cfg


def load_config(path):
    def merge(base, override):
        if not isinstance(override, dict) or not set(override) <= set(base):
            raise ValueError("Unknown Phase-7 config fields or non-mapping")
        out = copy.deepcopy(base)
        for key, value in override.items():
            out[key] = merge(base[key], value) if isinstance(base[key], dict) else value
        return out
    return validate_config(merge(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {}))
