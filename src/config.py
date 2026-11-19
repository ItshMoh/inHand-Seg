"""Run configs: YAML files in configs/, merged over DEFAULTS. Torch-free (read by the local entrypoints)."""

import copy
from pathlib import Path

import yaml

from src.labels import label_lut

DEFAULTS = {
    "run_name": None,  # default: the config file stem
    "seed": 0,
    "data": {
        "train_root": "/data/egohos_512",
        "train_file_list": None,  # text file of stems (relative to the repo root); null = full train split
        "subset": None,  # e.g. {per_source: 2, min_class_share: 0.01}: pick a few images from the file list
        "augment": True,  # false = fixed 512 px shorter side + pad, no randomness (overfit test)
        "label_map": None,  # remapped id -> training id, e.g. {2: 0} for hands-only; same map in train, eval and panels
        "num_workers": 8,
    },
    "model": {
        "pretrained": "nvidia/mit-b0",
        "class_names": ["background", "hand", "interacting_object"],
        "freeze_encoder": False,
    },
    "train": {
        "max_iters": 1000,
        "batch_size": 16,
        "warmup_iters": 100,
        "lr": 6.0e-5,
        "head_lr_mult": 10.0,
        "weight_decay": 0.01,
        "poly_power": 1.0,
        "bf16": True,
        "log_every": 20,
        "ckpt_every": 500,
    },
    "eval": {
        "splits": ["val"],  # "train" = the images this run trained on, scored at original resolution
        "boundary_dilation_ratio": 0.02,
        "n_vis": 6,
        "min_fg_iou": None,  # if set, every non-background class IoU must reach it (overfit check)
        "val_every": 0,  # > 0: full val every N iterations -> val_log.jsonl + val_curve.png (curve only, no selection)
    },
    "cascade": {
        "enabled": False,
        "stage1_checkpoint": "/data/runs/b0_hands_only/model_final.pt",
        "corruption": {
            "morph_probs": [0.4, 0.3, 0.3],  # none, dilation, erosion; on the final crop
            "kernel_sizes": [3, 5, 7, 9, 11, 13, 15],
            "drop_prob": 0.1,  # one of the two largest connected components
            "blob_prob": 0.2,
            "blob_count": [1, 3],
            "blob_radius": [3, 10],
        },
    },
}


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"Unknown config field: {path}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = _merge(base[key], value, f"{path}{key}.")
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = _merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    if cfg["run_name"] is None:
        cfg["run_name"] = path.stem
    label_lut(cfg["data"]["label_map"], len(cfg["model"]["class_names"]))
    if cfg["cascade"]["enabled"]:
        if cfg["model"]["class_names"] != ["background", "interacting_object"]:
            raise ValueError("Cascade stage 2 needs [background, interacting_object]")
        if cfg["data"]["label_map"] != {1: 0, 2: 1}:
            raise ValueError("Phase 5 uses label_map {1: 0, 2: 1}; 255 stays ignored")
        if cfg["model"]["freeze_encoder"]:
            raise ValueError("Cascade stage 2 must fine-tune its encoder")
        from src.cascade_data import validate_corruption

        validate_corruption(cfg["cascade"]["corruption"])
    return cfg
