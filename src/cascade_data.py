"""Phase-5 training inputs. Torch-free; src.data and its augmentation recipe stay unchanged.

Run the existing RGB transform on the *clean 3-class mask*, derive the aligned hand channel
from that transformed mask, corrupt only the channel, then map the clean loss target.
Deriving after nearest-neighbour geometry is equivalent to transforming the binary mask
alongside the label, and avoids padding a hand channel with the label's ignore value 255.
"""

import cv2
import numpy as np

from src.labels import label_lut


def validate_corruption(cfg: dict) -> None:
    probs = cfg["morph_probs"]
    if len(probs) != 3 or any(not np.isfinite(p) or p < 0 for p in probs) or not np.isclose(sum(probs), 1):
        raise ValueError("morph_probs must be three probabilities summing to 1")
    kernels = cfg["kernel_sizes"]
    if not kernels or any(not isinstance(k, int) or k < 3 or k % 2 == 0 for k in kernels):
        raise ValueError("kernel_sizes must contain odd integers >= 3")
    for key in ("drop_prob", "blob_prob"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    for key in ("blob_count", "blob_radius"):
        bounds = cfg[key]
        if len(bounds) != 2 or any(not isinstance(v, int) for v in bounds) or not 1 <= bounds[0] <= bounds[1]:
            raise ValueError(f"{key} must be [positive integer min, max]")


def corrupt_hand(hand: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Return a fresh binary float32 mask and a record of this draw. Never mutate the input."""
    out = hand.astype(np.uint8, copy=True)
    meta = {"dropped_component": False, "morphology": "none", "kernel": None, "blobs": []}
    if rng.random() < cfg["drop_prob"]:
        n, components, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        largest = sorted(range(1, n), key=lambda c: int(stats[c, cv2.CC_STAT_AREA]), reverse=True)[:2]
        if largest:
            chosen = int(rng.choice(largest))
            out[components == chosen] = 0
            meta["dropped_component"] = True
    morph = int(rng.choice(3, p=cfg["morph_probs"]))
    if morph:
        k = int(rng.choice(cfg["kernel_sizes"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        operation = cv2.dilate if morph == 1 else cv2.erode
        out = operation(out, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        meta.update(morphology="dilation" if morph == 1 else "erosion", kernel=k)
    if rng.random() < cfg["blob_prob"]:
        count = int(rng.integers(cfg["blob_count"][0], cfg["blob_count"][1] + 1))
        for _ in range(count):
            radius = int(rng.integers(cfg["blob_radius"][0], cfg["blob_radius"][1] + 1))
            x, y = int(rng.integers(out.shape[1])), int(rng.integers(out.shape[0]))
            cv2.circle(out, (x, y), radius, 1, -1)
            meta["blobs"].append([x, y, radius])
    return out.astype(np.float32), meta


class CascadeTrainTransform:
    def __init__(self, rgb_transform, corruption: dict, seed: int = 0):
        validate_corruption(corruption)
        self.rgb_transform = rgb_transform
        self.corruption = corruption
        self.lut = label_lut({1: 0, 2: 1}, 2)
        self.set_random_seed(seed)

    @property
    def aug(self):
        # seed_worker reaches both the RGB augmentation and the independent corruption RNG.
        return self

    def set_random_seed(self, seed: int) -> None:
        self.rgb_transform.aug.set_random_seed(seed)
        self.rng = np.random.default_rng(seed)

    def draw(self, image: np.ndarray, mask: np.ndarray):
        rgb, clean = self.rgb_transform(image, mask)
        hand, meta = corrupt_hand(clean == 1, self.corruption, self.rng)
        inputs = np.concatenate([rgb, hand[..., None]], axis=2).astype(np.float32)
        return inputs, self.lut[clean], clean, meta

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        inputs, target, _, _ = self.draw(image, mask)
        return inputs, target
