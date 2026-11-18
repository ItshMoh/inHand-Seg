"""Dataset inspection: file counts, image sizes and label values for each EgoHOS split."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ["train", "val", "test_indomain", "test_outdomain"]
EXPECTED_COUNTS = {"train": 8993, "val": 1124, "test_indomain": 1126, "test_outdomain": 500}
ORIGINAL_LABELS = {
    0: "background",
    1: "left_hand",
    2: "right_hand",
    3: "obj1_left",
    4: "obj1_right",
    5: "obj1_both",
    6: "obj2_left",
    7: "obj2_right",
    8: "obj2_both",
}
MAX_EXAMPLES = 10


def _files_by_stem(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file() and not p.name.startswith(".")}


def _inspect_sample(paths: tuple[Path, Path, Path | None]) -> dict:
    image_path, label_path, contact_path = paths
    with Image.open(image_path) as im:
        image_size = im.size
    with Image.open(label_path) as lb:
        label_mode = lb.mode
        label = np.array(lb)
    values, counts = np.unique(label, return_counts=True)
    result = {
        "image_size": image_size,
        "label_mode": label_mode,
        "label_shape": label.shape,
        "label_counts": dict(zip(values.tolist(), counts.tolist())),
    }
    if contact_path is not None:
        with Image.open(contact_path) as ct:
            contact = np.array(ct)
        result["contact_shape"] = contact.shape
        result["contact_values"] = np.unique(contact).tolist()
    return result


def inspect_split(root: Path, split: str, workers: int) -> dict:
    images = _files_by_stem(root / split / "image")
    labels = _files_by_stem(root / split / "label")
    contacts = _files_by_stem(root / split / "contact")

    paired = sorted(set(images) & set(labels))
    jobs = [(images[s], labels[s], contacts.get(s)) for s in paired]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(_inspect_sample, jobs, chunksize=32))

    image_sizes = Counter()
    label_modes = Counter()
    pixel_counts = Counter()
    contact_values = set()
    size_mismatch = []
    contact_mismatch = []
    for stem, s in zip(paired, samples):
        w, h = s["image_size"]
        image_sizes[f"{w}x{h}"] += 1
        label_modes[s["label_mode"]] += 1
        pixel_counts.update(s["label_counts"])
        if s["label_shape"][:2] != (h, w):
            size_mismatch.append(stem)
        if "contact_shape" in s:
            contact_values.update(s["contact_values"])
            if s["contact_shape"][:2] != (h, w):
                contact_mismatch.append(stem)

    total_pixels = sum(pixel_counts.values())
    return {
        "n_images": len(images),
        "n_labels": len(labels),
        "n_contact": len(contacts),
        "n_paired": len(paired),
        "expected": EXPECTED_COUNTS.get(split),
        "images_without_label": sorted(set(images) - set(labels))[:MAX_EXAMPLES],
        "labels_without_image": sorted(set(labels) - set(images))[:MAX_EXAMPLES],
        "image_extensions": dict(Counter(p.suffix.lower() for p in images.values())),
        "label_extensions": dict(Counter(p.suffix.lower() for p in labels.values())),
        "label_modes": dict(label_modes),
        "image_sizes_top": dict(image_sizes.most_common(10)),
        "n_distinct_image_sizes": len(image_sizes),
        "label_size_mismatch": {"count": len(size_mismatch), "examples": size_mismatch[:MAX_EXAMPLES]},
        "contact_size_mismatch": {"count": len(contact_mismatch), "examples": contact_mismatch[:MAX_EXAMPLES]},
        "label_values": sorted(pixel_counts),
        "label_pixel_share": {
            f"{v} {ORIGINAL_LABELS.get(v, '?')}": round(c / total_pixels, 6) for v, c in sorted(pixel_counts.items())
        },
        "contact_values": sorted(contact_values),
    }


def inspect_dataset(root: str, workers: int = 8) -> dict:
    root_path = Path(root)
    splits = {split: inspect_split(root_path, split, workers) for split in SPLITS}
    checks = {
        "counts_match_paper": {s: v["n_paired"] == v["expected"] for s, v in splits.items()},
        "every_image_has_label": {
            s: not v["images_without_label"] and not v["labels_without_image"] for s, v in splits.items()
        },
        "label_values_within_0_8": {s: set(v["label_values"]) <= set(range(9)) for s, v in splits.items()},
        "label_sizes_match_images": {s: v["label_size_mismatch"]["count"] == 0 for s, v in splits.items()},
    }
    return {"root": root, "splits": splits, "checks": checks}
