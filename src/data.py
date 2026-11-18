"""EgoHOS data pipeline: label remap, sources, dev subset, transforms and the dataset.

Torch-free on purpose (the CPU data image has no torch). The dataset returns numpy arrays; the PyTorch
DataLoader's default collate turns them into tensors.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np

cv2.setNumThreads(0)

DATA_ROOT = "/data/egohos"
PRERESIZED_ROOT = "/data/egohos_512"
SHORT_SIDE = 512
CROP_SIZE = 512
IGNORE_INDEX = 255

CLASSES = {0: "background", 1: "hand", 2: "interacting_object"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 0 background | 1,2 left/right hand | 3,4,5 1st-order object | 6,7,8 2nd-order object (ignored)
REMAP_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
REMAP_LUT[0] = 0
REMAP_LUT[[1, 2]] = 1
REMAP_LUT[[3, 4, 5]] = 2

SOURCES = ("ego4d", "epic", "thu", "own")

IMREAD_IMAGE = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
IMREAD_LABEL = cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION


def remap_label(label: np.ndarray) -> np.ndarray:
    """Original 0-8 EgoHOS ids -> {0 background, 1 hand, 2 interacting object, 255 ignore}."""
    return REMAP_LUT[label]


def source_of(name: str) -> str:
    """Data source from a file name: ego4d_*, epic_*, thu_*; anything else (escape_*) is the authors' own videos."""
    prefix = Path(name).stem.split("_", 1)[0]
    return prefix if prefix in SOURCES else "own"


def make_dev_subset(names: list[str], n: int = 500, seed: int = 0) -> list[str]:
    """Sample n names without replacement, in proportion to source (largest-remainder rounding)."""
    by_source: dict[str, list[str]] = {}
    for name in sorted(names):
        by_source.setdefault(source_of(name), []).append(name)

    total = sum(len(v) for v in by_source.values())
    exact = {s: n * len(v) / total for s, v in by_source.items()}
    quota = {s: int(q) for s, q in exact.items()}
    for s in sorted(exact, key=lambda s: exact[s] - quota[s], reverse=True)[: n - sum(quota.values())]:
        quota[s] += 1

    rng = np.random.default_rng(seed)
    picked = []
    for s in sorted(by_source):
        picked += rng.choice(by_source[s], size=quota[s], replace=False).tolist()
    return sorted(picked)


def read_file_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def denormalize(image: np.ndarray) -> np.ndarray:
    """Normalised CHW float32 -> HWC uint8 RGB, for visualisation."""
    hwc = image.transpose(1, 2, 0) * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.ascontiguousarray(np.clip(hwc * 255 + 0.5, 0, 255), dtype=np.uint8)


class TrainTransform:
    """Random rescale to a 256-1024 px shorter side (0.5-2.0x of the 512 eval scale), random 512 crop
    (padding: image 0, mask 255), horizontal flip, colour jitter, ImageNet normalisation."""

    def __init__(self, crop_size: int = CROP_SIZE, scale_range: tuple[float, float] = (0.5, 2.0), seed: int | None = None):
        lo, hi = (round(SHORT_SIDE * s) for s in scale_range)
        self.aug = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=list(range(lo, hi + 1, 8)),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST_EXACT,
                    area_for_downscale="image",
                ),
                A.RandomCrop(
                    crop_size,
                    crop_size,
                    pad_if_needed=True,
                    pad_position="random",
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=IGNORE_INDEX,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02, p=0.8),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ],
            seed=seed,
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.aug(image=image, mask=mask)
        return out["image"], out["mask"]


class EvalTransform:
    """Resize the image to a 512 px shorter side and normalise. The mask stays at original resolution:
    logits are upsampled to the original size and scored against the original label."""

    def __init__(self, short_side: int = SHORT_SIDE):
        self.aug = A.Compose(
            [
                A.SmallestMaxSize(max_size=short_side, interpolation=cv2.INTER_LINEAR, area_for_downscale="image"),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.aug(image=image)["image"], mask


class EgoHOSDataset:
    """Map-style dataset over {root}/{split}/{image,label}, paired by file stem.

    Returns (image, mask): image is float32 CHW (normalised if the transform normalises), mask is int64 HW
    with values {0, 1, 2, 255}. `file_list` is a list of stems or a text file with one stem per line.
    """

    def __init__(self, split: str, root: str | Path, transform=None, file_list: str | Path | list[str] | None = None):
        self.image_dir = Path(root) / split / "image"
        self.label_dir = Path(root) / split / "label"
        images = {p.stem: p for p in self.image_dir.iterdir() if p.suffix.lower() == ".jpg"}
        labels = {p.stem: p for p in self.label_dir.iterdir() if p.suffix.lower() == ".png"}

        if file_list is None:
            names = sorted(set(images) & set(labels))
        else:
            names = read_file_list(file_list) if isinstance(file_list, (str, Path)) else list(file_list)
            names = [Path(n).stem for n in names]
            missing = [n for n in names if n not in images or n not in labels]
            if missing:
                raise FileNotFoundError(f"{len(missing)} names in file_list not found in {root}/{split}: {missing[:5]}")

        self.split = split
        self.names = names
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw RGB uint8 HWC image and original 0-8 label, before remap and transform."""
        name = self.names[index]
        image = cv2.imread(str(self.images[name]), IMREAD_IMAGE)
        label = cv2.imread(str(self.labels[name]), IMREAD_LABEL)
        if image is None or label is None:
            raise OSError(f"Could not read {name}")
        if image.shape[:2] != label.shape[:2]:
            raise ValueError(f"{name}: image {image.shape[:2]} vs label {label.shape[:2]}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), label

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, label = self.load(index)
        mask = remap_label(label)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = image.astype(np.float32) / 255.0
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32), mask.astype(np.int64)
