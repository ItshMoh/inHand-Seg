"""Aligned Phase-7 panels. Rendering never changes original scoring coordinates."""

import html
import io
import textwrap

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

from src.sanity import overlay_classes

FP = (235, 55, 55)
FN = (0, 220, 220)
REGION = (220, 50, 210)


def object_errors(pred, gt):
    if pred.shape != gt.shape or pred.ndim != 2 or not np.isin(pred, [0, 1, 2]).all() or not np.isin(gt, [0, 1, 2, 255]).all():
        raise ValueError("FP/FN require aligned original three-class masks")
    fp = (pred == 2) & (gt != 2) & (gt != 255)
    fn = (gt == 2) & (pred != 2)
    return fp, fn


def _resize(rgb, arrays, max_side=800):
    h, w = rgb.shape[:2]
    if any(a.shape != (h, w) for a in arrays):
        raise ValueError("Display arrays are not aligned")
    scale = min(1., max_side / max(h, w))
    size = max(1, round(w*scale)), max(1, round(h*scale))
    return (cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
            [cv2.resize(a.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST_EXACT) for a in arrays])


def _outline(image, mask):
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, REGION, 2)
    return out


def panel_arrays(rgb, gt, baseline, cascade, region=None, crop=None):
    if rgb.dtype != np.uint8 or rgb.shape != (*gt.shape, 3):
        raise ValueError("Expected aligned original RGB uint8")
    bfp, bfn = object_errors(baseline, gt)
    cfp, cfn = object_errors(cascade, gt)
    region = np.zeros(gt.shape, bool) if region is None else region
    if region.shape != gt.shape or (region & (gt != 0)).any():
        raise ValueError("SAM overlay must be the eligible original GT-background mask")
    arrays = [gt, baseline, cascade, bfp, bfn, cfp, cfn, region]
    if crop is not None:
        x0, y0, x1, y1 = crop
        rgb = rgb[y0:y1, x0:x1]
        arrays = [a[y0:y1, x0:x1] for a in arrays]
    rgb, arrays = _resize(rgb, arrays)
    truth, base, casc, bf, bn, cf, cn, reg = arrays
    ignore = truth == 255
    panels = [rgb, overlay_classes(rgb, np.where(ignore, 255, base).astype(np.uint8)),
              overlay_classes(rgb, np.where(ignore, 255, casc).astype(np.uint8)), overlay_classes(rgb, truth)]
    for fp, fn in ((bf, bn), (cf, cn)):
        error = overlay_classes(rgb, np.where(ignore, 255, 0).astype(np.uint8))
        error[fp.astype(bool)] = FP
        error[fn.astype(bool)] = FN
        panels.append(error)
    return [_outline(p, reg) if reg.any() else p for p in panels]


def auto_crop(gt, region, max_fraction):
    mask = (gt == 2) | (region if region is not None else False)
    yy, xx = np.nonzero(mask)
    if not xx.size:
        return None
    margin = max(24, round(.15 * max(int(np.ptp(xx))+1, int(np.ptp(yy))+1)))
    crop = [max(0, int(xx.min())-margin), max(0, int(yy.min())-margin),
            min(gt.shape[1], int(xx.max())+margin+1), min(gt.shape[0], int(yy.max())+margin+1)]
    return crop if (crop[2]-crop[0])*(crop[3]-crop[1]) < max_fraction*gt.size else None


def render_panel(case, rgb, gt, predictions, cfg, region=None, crop=None):
    base, cascade = (predictions[cfg[g][0]] for g in ("baseline_runs", "cascade_runs"))
    panels = panel_arrays(rgb, gt, base, cascade, region, crop)
    h, w = panels[0].shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(15, max(8, 10*h/w+3.8)), squeeze=False)
    titles = ["Original RGB", f"{cfg['baseline_runs'][0]} — seed 0",
              f"{cfg['cascade_runs'][0]} — stage 2 seed 0\nfrozen stage 1 seed 0; predicted hands",
              "Original three-class GT", "Baseline object FP / FN", "Cascade object FP / FN"]
    for ax, panel, title in zip(axes.ravel(), panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    heading = f"{case['case_id']} | {case['split']} | {case['source']} | original {gt.shape[1]}×{gt.shape[0]}"
    heading += " | FULL FRAME" if crop is None else f" | MATCHED CLOSE-UP xyxy={crop}"
    fig.suptitle(heading + "\n" + "\n".join(textwrap.wrap(case["name"], 125)), fontsize=10, y=.99)
    legend = [Patch(color=np.array(c)/255, label=t) for c, t in
              (((0,120,255), "hand"), ((255,200,0), "interacting object"), ((150,150,150), "ignore: grey hatched"),
               (FP, "object FP (includes hand/object confusion)"), (FN, "object FN"))]
    if region is not None:
        candidate = case["candidate"]
        legend.append(Patch(facecolor="none", edgecolor=np.array(REGION)/255,
                            label=f"SAM eligible GT-background C, ID {candidate['mask_id']} ({case['region_kind']})"))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5,.065), ncol=3, fontsize=8)
    fig.text(.5, .012, "\n".join(textwrap.wrap(case["caption"], 160)), ha="center", fontsize=9)
    # Fixed reserved bands keep legends/titles outside both equal-aspect image rows.
    fig.subplots_adjust(left=.015, right=.985, top=.86, bottom=.20, wspace=.025, hspace=.23)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=110)
    plt.close(fig)
    return stream.getvalue()


def render_case(case, rgb, gt, predictions, cfg, region=None):
    files = {f"gallery/{case['case_id']}_full.png": render_panel(case, rgb, gt, predictions, cfg, region)}
    crop = case["crop_xyxy"] or auto_crop(gt, region, cfg["gallery"]["closeup_max_frame_fraction"])
    if crop is not None:
        files[f"gallery/{case['case_id']}_crop.png"] = render_panel(case, rgb, gt, predictions, cfg, region, crop)
    return files, crop


def survey_thumbnail(rgb, gt):
    rgb, (gt,) = _resize(rgb, [gt], max_side=480)
    pair = np.concatenate([rgb, overlay_classes(rgb, gt)], axis=1)
    stream = io.BytesIO()
    Image.fromarray(pair).save(stream, format="PNG")
    return stream.getvalue()


def html_page(title, text, cards):
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{html.escape(title)}</title><style>body{{font:16px system-ui;margin:24px auto;max-width:1500px;padding:0 16px;background:#f5f6f8;color:#17202b}}"
            "article{background:white;padding:18px;margin:20px 0;border:1px solid #ccd2da;border-radius:8px}img{max-width:100%;height:auto}"
            "code{overflow-wrap:anywhere}p{line-height:1.5}.pending{color:#925000}</style>"
            f"<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p>" + "".join(cards) + "</html>").encode()


def survey_index(shortlist):
    cards = []
    for r in shortlist:
        p = r["paired"]["interacting_object"]
        delta = "undefined" if p["mean_delta"] is None else f"{100*p['mean_delta']:+.2f} points"
        hints = "; ".join(f"ID {h['mask_id']}: {h['event']}; fractions={h['object_fractions']}; d/diag={h['distance']:.5f}"
                          for h in r["candidate_hints"])
        small = sum(c["small"] for c in r["gt"]["object_components"])
        cards.append(f"<article><h2>{r['image_id']} · {html.escape(r['source'])}</h2><code>{html.escape(r['name'])}</code>"
                     f"<p>{html.escape(', '.join(r['selection_reasons']))}; mean paired delta: {delta}. "
                     f"GT small components: {small}; original ID-5 pixels: {r['gt']['two_hand_pixels']}.</p>"
                     f"<a href='{r['image_id']}.png'><img loading='lazy' src='{r['image_id']}.png' alt='Original RGB and GT'></a>"
                     f"<p>{html.escape(hints)}</p><p>Appearance tags: pending visual review.</p></article>")
    return html_page("Phase 7: diagnostic review shortlist", "RGB | original GT. This selected shortlist is not a prevalence estimate. Blue=hand, yellow=object, grey hatch=ignore. Geometry suggestions are not confirmed appearance categories.", cards)


def gallery_index(selection, records, preview=False):
    cards = []
    for c in selection["cases"]:
        if preview and not c["preview"]:
            continue
        rec = records.get(c["case_id"])
        body = "<p class='pending'>Rendering pending.</p>"
        if rec:
            names = sorted(rec["files_sha256"], key=lambda n: ("_crop" in n, n))
            body = "".join(f"<a href='{html.escape(n.split('/')[-1])}'><img loading='lazy' src='{html.escape(n.split('/')[-1])}' alt='Aligned comparison panel'></a>" for n in names)
            body += f"<p>Both-seed object IoUs: <code>{html.escape(str({r: v['iou']['interacting_object'] for r,v in rec['models'].items()}))}</code></p>"
            if rec.get("candidate"):
                body += f"<p>SAM eligible region: <code>{html.escape(str(rec['candidate']))}</code>. A proposal region is not a certified untouched object instance.</p>"
        cards.append(f"<article><h2>{html.escape(c['case_id'])} · {html.escape(c['source'])}</h2><code>{html.escape(c['name'])}</code>"
                     f"<p>{html.escape(', '.join(c['tags']))}</p><p>{html.escape(c['caption'])}</p>{body}</article>")
    category_text = "; ".join(f"{t}: {v['status']} ({v['note']})" for t,v in selection["categories"].items())
    cards.insert(0, f"<article><h2>Category review</h2><p>{html.escape(category_text)}</p></article>")
    return html_page("Phase 7: four-case layout preview" if preview else "Phase 7: selected failure and disagreement gallery",
                     "Fixed seed-0 display pair; both-seed scores recorded. Selected qualitative evidence is separate from the full-split quantitative evaluation. Category captions and final outputs still require user verification.", cards)
