"""Sanity overlays: label colours, 3-class remap, training augmentation and eval transform."""

import io
import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

from src.data import (
    IGNORE_INDEX,
    SHORT_SIDE,
    EgoHOSDataset,
    EvalTransform,
    TrainTransform,
    denormalize,
    remap_label,
    source_of,
)
from src.dataset_stats import ORIGINAL_LABELS

ORIGINAL_COLOURS = {
    0: (0, 0, 0),
    1: (230, 25, 75),
    2: (245, 130, 48),
    3: (60, 180, 75),
    4: (210, 245, 60),
    5: (0, 128, 128),
    6: (0, 130, 200),
    7: (145, 30, 180),
    8: (70, 240, 240),
}
CLASS_COLOURS = {1: (0, 120, 255), 2: (255, 200, 0)}
IGNORE_COLOUR = (150, 150, 150)
DISPLAY_MAX_SIDE = 960


def _display_size(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = label.shape[:2]
    scale = DISPLAY_MAX_SIDE / max(h, w)
    if scale >= 1:
        return image, label
    size = (round(w * scale), round(h * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.resize(label, size, interpolation=cv2.INTER_NEAREST_EXACT)


def colourise_original(label: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for v, c in ORIGINAL_COLOURS.items():
        lut[v] = c
    return lut[label]


def overlay_classes(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint hand / interacting object, hatch ignore in grey, background untouched; class contours on top."""
    out = image.astype(np.float32)
    for cls, colour in CLASS_COLOURS.items():
        m = mask == cls
        out[m] = (1 - alpha) * out[m] + alpha * np.array(colour)

    ignore = mask == IGNORE_INDEX
    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
    stripes = ((xx + yy) // 4) % 2 == 0
    out[ignore] = 0.5 * out[ignore] + 0.5 * np.array(IGNORE_COLOUR)
    out[ignore & stripes] = IGNORE_COLOUR
    out = np.ascontiguousarray(out, dtype=np.uint8)

    for cls, colour in CLASS_COLOURS.items():
        contours, _ = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, colour, 1)
    return out


def _class_legend() -> list[Patch]:
    return [
        Patch(color=np.array(CLASS_COLOURS[1]) / 255, label="1 hand"),
        Patch(color=np.array(CLASS_COLOURS[2]) / 255, label="2 interacting object"),
        Patch(color=np.array(IGNORE_COLOUR) / 255, label="255 ignore (hatched)"),
    ]


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _figure(n_cols: int, h: int, w: int, col_width: float = 6.0):
    fig, axes = plt.subplots(1, n_cols, figsize=(col_width * n_cols, col_width * h / w + 1.2), squeeze=False)
    for ax in axes[0]:
        ax.axis("off")
    return fig, axes[0]


def render_label_panel(name: str, image: np.ndarray, label: np.ndarray) -> bytes:
    """image | original 0-8 label (colour per id) | remapped 3-class overlay."""
    orig_h, orig_w = label.shape
    image, label = _display_size(image, label)
    fig, axes = _figure(3, *label.shape)
    fig.suptitle(f"{name}   source={source_of(name)}   original {orig_w}x{orig_h}", fontsize=10)

    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(colourise_original(label))
    axes[1].set_title("original label (0-8)")
    present = sorted(np.unique(label).tolist())
    axes[1].legend(
        handles=[Patch(color=np.array(ORIGINAL_COLOURS[v]) / 255, label=f"{v} {ORIGINAL_LABELS[v]}") for v in present],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False,
    )
    axes[2].imshow(overlay_classes(image, remap_label(label)))
    axes[2].set_title("remapped 3-class overlay")
    axes[2].legend(handles=_class_legend(), loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
    return _to_png(fig)


def render_aug_panel(name: str, image: np.ndarray, mask: np.ndarray, samples: list[tuple[np.ndarray, np.ndarray]]) -> bytes:
    """Stored pre-resized image with its remapped mask | several training-augmentation draws."""
    fig, axes = _figure(1 + len(samples), 1, 1, col_width=4.5)
    fig.suptitle(f"{name}   source={source_of(name)}   stored {image.shape[1]}x{image.shape[0]}", fontsize=10)
    axes[0].imshow(overlay_classes(image, mask))
    axes[0].set_title("stored (pre-resized), no aug", fontsize=9)
    for ax, (aug_image, aug_mask) in zip(axes[1:], samples):
        ax.imshow(overlay_classes(aug_image, aug_mask))
        ax.set_title(f"aug {aug_mask.shape[1]}x{aug_mask.shape[0]}, ignore {np.mean(aug_mask == IGNORE_INDEX):.0%}", fontsize=9)
    axes[0].legend(handles=_class_legend(), loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=8, frameon=False)
    return _to_png(fig)


def render_eval_panel(name: str, eval_image: np.ndarray, image: np.ndarray, mask: np.ndarray) -> bytes:
    """Model input after the eval transform | original image with the original-resolution mask."""
    fig, axes = _figure(2, *mask.shape)
    fig.suptitle(f"{name}   source={source_of(name)}", fontsize=10)
    axes[0].imshow(eval_image)
    axes[0].set_title(f"eval input {eval_image.shape[1]}x{eval_image.shape[0]} (shorter side {SHORT_SIDE})", fontsize=9)
    disp_image, disp_mask = _display_size(image, mask)
    axes[1].imshow(overlay_classes(disp_image, disp_mask))
    axes[1].set_title(f"scored against: original {mask.shape[1]}x{mask.shape[0]} mask", fontsize=9)
    axes[1].legend(handles=_class_legend(), loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
    return _to_png(fig)


def _pick_by_source(names: list[str], per_source: int, total: int, rng: random.Random) -> list[str]:
    by_source: dict[str, list[str]] = {}
    for n in names:
        by_source.setdefault(source_of(n), []).append(n)
    picked = [n for s in sorted(by_source) for n in rng.sample(by_source[s], min(per_source, len(by_source[s])))]
    rest = sorted(set(names) - set(picked))
    return picked + rng.sample(rest, total - len(picked))


def build_sanity(data_root: str, preresized_root: str, n_label: int = 20, n_aug: int = 5, n_aug_draws: int = 3, seed: int = 0) -> dict:
    rng = random.Random(seed)
    files: dict[str, bytes] = {}
    remapped_values: set[int] = set()
    summary: dict = {"label_panels": [], "aug_panels": [], "eval_panels": []}

    # 1. Original training samples: colour per original id + remapped overlay.
    train = EgoHOSDataset("train", data_root)
    index = {n: i for i, n in enumerate(train.names)}
    for i, name in enumerate(_pick_by_source(train.names, 2, n_label, rng)):
        image, label = train.load(index[name])
        remapped_values |= set(np.unique(remap_label(label)).tolist())
        files[f"label_{i:02d}_{name}.png"] = render_label_panel(name, image, label)
        summary["label_panels"].append({"name": name, "size": f"{label.shape[1]}x{label.shape[0]}", "ids": np.unique(label).tolist()})

    # 2. Training augmentation on the pre-resized copy (as used in training).
    train_512 = EgoHOSDataset("train", preresized_root, TrainTransform(seed=seed))
    index_512 = {n: i for i, n in enumerate(train_512.names)}
    by_source = {s: sorted(n for n in train_512.names if source_of(n) == s) for s in ("epic", "thu", "own", "ego4d")}
    aug_names = [rng.choice(by_source["epic"]), rng.choice(by_source["thu"]), rng.choice(by_source["own"])]
    aug_names += rng.sample(by_source["ego4d"], n_aug - len(aug_names))
    for i, name in enumerate(aug_names):
        idx = index_512[name]
        image, label = train_512.load(idx)
        draws = []
        for _ in range(n_aug_draws):
            x, mask = train_512[idx]
            assert x.shape == (3, 512, 512) and x.dtype == np.float32 and mask.shape == (512, 512) and mask.dtype == np.int64
            remapped_values |= set(np.unique(mask).tolist())
            draws.append((denormalize(x), mask.astype(np.uint8)))
        files[f"aug_{i}_{name}.png"] = render_aug_panel(name, image, remap_label(label), draws)
        summary["aug_panels"].append({"name": name, "stored_size": f"{label.shape[1]}x{label.shape[0]}"})

    # 3. Eval transform on val: smallest image, largest image, one random.
    val = EgoHOSDataset("val", data_root, EvalTransform())
    sizes = {}
    for n in val.names:
        with Image.open(val.images[n]) as im:
            sizes[n] = im.size[0] * im.size[1]
    by_area = sorted(val.names, key=lambda n: (sizes[n], n))
    eval_names = [by_area[0], by_area[-1], rng.choice(by_area[1:-1])]
    val_index = {n: i for i, n in enumerate(val.names)}
    for i, name in enumerate(eval_names):
        idx = val_index[name]
        image, label = val.load(idx)
        x, mask = val[idx]
        remapped_values |= set(np.unique(mask).tolist())
        files[f"eval_{i}_{name}.png"] = render_eval_panel(name, denormalize(x), image, mask.astype(np.uint8))
        summary["eval_panels"].append({
            "name": name,
            "original_size": f"{label.shape[1]}x{label.shape[0]}",
            "eval_input_size": f"{x.shape[2]}x{x.shape[1]}",
            "eval_shorter_side": int(min(x.shape[1:])),
            "mask_size_equals_original": mask.shape == label.shape,
        })

    summary["remapped_values_seen"] = sorted(remapped_values)
    summary["checks"] = {
        "remapped_values_in_0_1_2_255": remapped_values <= {0, 1, 2, 255},
        "eval_shorter_side_is_512": all(p["eval_shorter_side"] == SHORT_SIDE for p in summary["eval_panels"]),
        "eval_mask_at_original_size": all(p["mask_size_equals_original"] for p in summary["eval_panels"]),
    }
    return {"files": files, "summary": summary}
