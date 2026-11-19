"""Per-run label maps on top of the 3-class remap, e.g. hands-only: {2: 0} (interacting object -> background).

Torch-free. The map is applied to the remapped mask *before* any transform: there 255 still means only 2nd-order
objects, whereas after the training transform it also marks crop padding.
"""

import numpy as np

from src.data import IGNORE_INDEX

REMAPPED_IDS = (0, 1, 2, IGNORE_INDEX)


def label_lut(label_map: dict | None, num_classes: int) -> np.ndarray | None:
    """uint8 lookup table over the remapped ids {0, 1, 2, 255}, or None for no map. Raises if any remapped id would
    land outside range(num_classes) and 255 (e.g. 2 class_names without a map)."""
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in (label_map or {}).items():
        src, dst = int(src), int(dst)
        if src not in REMAPPED_IDS:
            raise ValueError(f"label_map key {src} is not a remapped id {REMAPPED_IDS}")
        if not (0 <= dst < 256):
            raise ValueError(f"label_map value {dst} for key {src} is not a uint8 id")
        lut[src] = dst
    produced = sorted({int(lut[i]) for i in REMAPPED_IDS})
    allowed = set(range(num_classes)) | {IGNORE_INDEX}
    if not set(produced) <= allowed:
        raise ValueError(
            f"label_map {label_map} with {num_classes} classes produces ids {produced}; allowed are 0..{num_classes - 1} and 255"
        )
    return lut if label_map else None


def apply_label_map(mask: np.ndarray, lut: np.ndarray | None) -> np.ndarray:
    return mask if lut is None else lut[mask]


class LabelMapTransform:
    """Map the remapped mask with the run's lookup table, then run the inner transform. Exposes the inner `aug`
    so train.seed_worker still reseeds the per-worker augmentation."""

    def __init__(self, inner, lut: np.ndarray):
        self.inner = inner
        self.lut = lut

    @property
    def aug(self):
        return getattr(self.inner, "aug", None)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.inner(image, self.lut[mask])


def with_label_map(transform, lut: np.ndarray | None):
    return transform if lut is None else LabelMapTransform(transform, lut)
