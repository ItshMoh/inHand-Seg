"""Pre-resized training copy: shorter side 512 px (never upscaled), labels kept as original 0-8 ids."""

import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.data import IMREAD_IMAGE, IMREAD_LABEL, SHORT_SIDE, remap_label

JPEG_QUALITY = 95


def _resize_pair(job: tuple[Path, Path, Path, Path]) -> str:
    image_path, label_path, image_out, label_out = job
    if image_out.exists() and label_out.exists():
        return "skipped"

    image = cv2.imread(str(image_path), IMREAD_IMAGE)
    h, w = image.shape[:2]
    if min(h, w) <= SHORT_SIDE:
        shutil.copyfile(image_path, image_out)
        shutil.copyfile(label_path, label_out)
        return "copied"

    scale = SHORT_SIDE / min(h, w)
    size = (SHORT_SIDE, round(h * scale)) if w <= h else (round(w * scale), SHORT_SIDE)
    label = cv2.imread(str(label_path), IMREAD_LABEL)
    cv2.imwrite(str(image_out), cv2.resize(image, size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cv2.imwrite(str(label_out), cv2.resize(label, size, interpolation=cv2.INTER_NEAREST_EXACT))
    return "resized"


def _check_pair(paths: tuple[Path, Path]) -> dict:
    image = cv2.imread(str(paths[0]), IMREAD_IMAGE)
    label = cv2.imread(str(paths[1]), IMREAD_LABEL)
    return {
        "image_hw": image.shape[:2],
        "label_hw": label.shape[:2],
        "label_ndim": label.ndim,
        "label_values": np.unique(label).tolist(),
        "remapped_values": np.unique(remap_label(label)).tolist(),
    }


def preresize_split(src_root: str, dst_root: str, split: str = "train", workers: int = 8) -> dict:
    src_img, src_lbl = Path(src_root, split, "image"), Path(src_root, split, "label")
    dst_img, dst_lbl = Path(dst_root, split, "image"), Path(dst_root, split, "label")
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = {p.stem: p for p in src_img.iterdir() if p.suffix.lower() == ".jpg"}
    labels = {p.stem: p for p in src_lbl.iterdir() if p.suffix.lower() == ".png"}
    stems = sorted(set(images) & set(labels))
    jobs = [(images[s], labels[s], dst_img / images[s].name, dst_lbl / labels[s].name) for s in stems]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        actions = Counter(pool.map(_resize_pair, jobs, chunksize=16))
        out_images = {p.stem: p for p in dst_img.iterdir() if p.suffix.lower() == ".jpg"}
        out_labels = {p.stem: p for p in dst_lbl.iterdir() if p.suffix.lower() == ".png"}
        out_stems = sorted(set(out_images) & set(out_labels))
        checks = list(pool.map(_check_pair, [(out_images[s], out_labels[s]) for s in out_stems], chunksize=16))

    sizes = Counter(f"{c['image_hw'][1]}x{c['image_hw'][0]}" for c in checks)
    label_values = sorted({v for c in checks for v in c["label_values"]})
    remapped_values = sorted({v for c in checks for v in c["remapped_values"]})
    max_short = max(min(c["image_hw"]) for c in checks)
    mismatched = [s for s, c in zip(out_stems, checks) if c["image_hw"] != c["label_hw"]]

    return {
        "split": split,
        "dst": str(Path(dst_root, split)),
        "n_source_pairs": len(stems),
        "actions": dict(actions),
        "n_out_images": len(out_images),
        "n_out_labels": len(out_labels),
        "n_out_pairs": len(out_stems),
        "out_sizes_top": dict(sizes.most_common(12)),
        "max_shorter_side": max_short,
        "label_values": label_values,
        "remapped_values": remapped_values,
        "label_size_mismatch": mismatched[:10],
        "checks": {
            "pairs_match_source": len(out_stems) == len(stems) == len(out_images) == len(out_labels),
            "shorter_side_le_512": max_short <= SHORT_SIDE,
            "labels_single_channel": all(c["label_ndim"] == 2 for c in checks),
            "label_values_within_0_8": set(label_values) <= set(range(9)),
            "remapped_values_in_0_1_2_255": set(remapped_values) <= {0, 1, 2, 255},
            "label_sizes_match_images": not mismatched,
        },
    }
