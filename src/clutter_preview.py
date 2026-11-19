"""Candidate-only pilot panels. Segmentation model predictions never influence this review."""

import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.clutter_sam import decode_rle, image_key
from src.sanity import _display_size, _to_png, overlay_classes

GREEN, RED, MAGENTA = (20, 205, 90), (240, 65, 65), (230, 40, 210)


def _overlay(rgb, mask, colour, alpha=0.35):
    result = rgb.copy()
    result[mask] = ((1 - alpha) * rgb[mask] + alpha * np.array(colour)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, colour, 2)
    return result


def _outline(ax, mask, colour):
    # Contours preserve a visible outline even for nested proposals.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        coords = contour[:, 0]
        if len(coords) >= 2:
            ax.plot(np.r_[coords[:, 0], coords[0, 0]], np.r_[coords[:, 1], coords[0, 1]],
                    color=np.array(colour) / 255, linewidth=1)


def _context_box(mask):
    yy, xx = np.nonzero(mask)
    if not xx.size:
        return 0, 0, mask.shape[1], mask.shape[0]
    margin = max(24, int(0.2 * max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)))
    return max(0, int(xx.min()) - margin), max(0, int(yy.min()) - margin), min(mask.shape[1], int(xx.max()) + margin + 1), min(mask.shape[0], int(yy.max()) + margin + 1)


def render_pilot(entry, rgb, gt, proposals, filtered, page_size=4):
    stem = f"{entry['split']}_{image_key(entry)}"
    disp_rgb, disp_gt = _display_size(rgb, gt)
    gt_panel = overlay_classes(disp_rgb, disp_gt)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(disp_rgb)
    axes[0].set_title("Original RGB (display resized)")
    axes[1].imshow(gt_panel)
    axes[1].set_title("GT: blue hands / yellow objects / grey ignore")
    axes[2].imshow(gt_panel)
    axes[3].imshow(gt_panel)
    rows = {r["mask_id"]: r for r in filtered["audit"]}
    candidates = {c["mask_id"]: c for c in filtered["candidates"]}
    for p in proposals:
        retained = p["mask_id"] in candidates
        mask = decode_rle(candidates[p["mask_id"]]["segmentation"] if retained else p["segmentation"])
        _, small = _display_size(rgb, mask.astype(np.uint8))
        mask = small.astype(bool)
        ax, colour = (axes[2], GREEN) if retained else (axes[3], RED)
        _outline(ax, mask, colour)
        yy, xx = np.nonzero(mask)
        if xx.size:
            ax.text(float(np.median(xx)), float(np.median(yy)), str(p["mask_id"]), color="white", fontsize=7,
                    bbox={"facecolor": "black", "alpha": 0.6, "pad": 1})
    axes[2].set_title(f"Retained background C: {len(candidates)} regions")
    axes[3].set_title(f"Rejected raw masks: {len(proposals)-len(candidates)}")
    title = f"{entry['split']} | {entry['display_source']} | {entry['width']}x{entry['height']} | {entry['name']}"
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    files = {f"{stem}_00_overview.png": _to_png(fig)}
    for retained, label in ((True, "retained"), (False, "rejected")):
        group = [p for p in proposals if (p["mask_id"] in candidates) == retained]
        for page in range(math.ceil(len(group) / page_size)):
            subset = group[page * page_size:(page + 1) * page_size]
            fig, axes = plt.subplots(len(subset), 2, figsize=(11, 3.5 * len(subset)), squeeze=False)
            fig.suptitle(f"{entry['display_source']} | {entry['name']}\n{label} proposals, page {page+1}; IDs are stable SAM mask IDs", fontsize=9, wrap=True)
            for pair, p in zip(axes, subset):
                row = rows[p["mask_id"]]
                raw = decode_rle(p["segmentation"])
                eligible = raw & (gt == 0)
                x0, y0, x1, y1 = _context_box(raw)
                image = rgb[y0:y1, x0:x1]
                truth = gt[y0:y1, x0:x1]
                raw_crop, c_crop = raw[y0:y1, x0:x1], eligible[y0:y1, x0:x1]
                base = overlay_classes(image, truth)
                pair[0].imshow(_overlay(base, raw_crop, MAGENTA))
                pair[1].imshow(_overlay(base, c_crop, GREEN if retained else RED))
                _outline(pair[1], raw_crop, MAGENTA)
                pair[0].set_title(f"ID {p['mask_id']} raw M: {p['area']} px ({p['area']/gt.size:.3%})", fontsize=9)
                info = f"ID {p['mask_id']} {row['reason']} | background {int(eligible.sum())} px"
                info += f" | removed {p['area'] - int(eligible.sum())} px"
                if row.get("ignore_pixels"):
                    info += f"\nignore overlap={row['ignore_pixels']} px"
                if row.get("foreground_inside_fraction") is not None:
                    info += f"\ninside={row['foreground_inside_fraction']:.3f}, max IoU={max(row['joint_foreground_iou'], row['max_component_iou']):.3f}"
                if row["duplicate_of"] is not None:
                    info += f"; duplicate of ID {row['duplicate_of']}"
                if retained:
                    info += f"\nd/diag={row['distance']:.5f} ({row['distance_px']:.2f} px)" if row["distance"] is not None else "\nno labelled hands: distance undefined"
                else:
                    info += "\nbackground shown for explanation; NOT scored"
                pair[1].set_title(info, fontsize=8)
                for ax in pair:
                    ax.axis("off")
            legend = [Line2D([0], [0], color=np.array(c)/255, lw=3, label=t) for c, t in
                      ((MAGENTA, "original M"), (GREEN, "retained C"), (RED, "rejected background (unscored)"))]
            fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8)
            fig.tight_layout(rect=(0, 0.035, 1, 0.93))
            files[f"{stem}_{label}_{page+1:02d}.png"] = _to_png(fig)
    return files
