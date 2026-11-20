# README asset provenance

These six PNGs are **byte-identical copies** of existing real panels and plots, with portable display names. There is no image generation, re-inference, resizing, retouching or mask reconstruction in Phase 8. Original outputs remain in place. [manifest.json](manifest.json) records source paths, hashes, dimensions, original image/label identities, case fingerprints, run/seed identities and both-seed object scores.

## Prediction panels

All four use `b0_full_ft` seed 0 versus `b0_cascade_stage2` stage-2 seed 0, with frozen `b0_hands_only` seed 0. Layout is RGB / baseline / cascade, then GT / baseline object errors / cascade object errors. Original embedded captions use **two-seed per-image means**, while the displayed masks use seed 0. The README states both explicitly.

| Copy | Case ID | Source / original image name | Complete original / matched crop / checked record |
|---|---|---|---|
| [Tray recovery](tray-recovery-seed0.png) | `9884ad4ff80c7e5cb7fc814e` | YouTube / `youtube_POV Cooking MONGOLIAN BEEF NOODLES  JONAH IN THE KITCHEN-seV_bNo_fLE-1080p-1644528285252_003060` | [Full](../../results/phase7/phase7_v1/gallery/9884ad4ff80c7e5cb7fc814e_full.png) · [Crop](../../results/phase7/phase7_v1/gallery/9884ad4ff80c7e5cb7fc814e_crop.png) · [Record](../../results/phase7/phase7_v1/case_records/9884ad4ff80c7e5cb7fc814e.json) |
| [Book regression](book-regression-seed0.png) | `819bb07cb30e12e169a83fc7` | own / `escape_GX010035_006540` | [Full](../../results/phase7/phase7_v1/gallery/819bb07cb30e12e169a83fc7_full.png) · [Crop](../../results/phase7/phase7_v1/gallery/819bb07cb30e12e169a83fc7_crop.png) · [Record](../../results/phase7/phase7_v1/case_records/819bb07cb30e12e169a83fc7.json) |
| [Headlamp part](headlamp-sam60-seed0.png) | `09ce8413ab9b0a71cb210772_sam60` | Ego4D / `ego4d_11f51f72-c0bd-4631-aabb-6ecbe17d95c9_25950` | [Full](../../results/phase7/phase7_v1/gallery/09ce8413ab9b0a71cb210772_sam60_full.png) · [Crop](../../results/phase7/phase7_v1/gallery/09ce8413ab9b0a71cb210772_sam60_crop.png) · [Record](../../results/phase7/phase7_v1/case_records/09ce8413ab9b0a71cb210772_sam60.json) |
| [Packaging-looking region](packaging-sam18-seed0.png) | `b9e5a553830bcf66609d2202_sam18` | EPIC / `epic_P07_101_frame_0000018721` | [Full](../../results/phase7/phase7_v1/gallery/b9e5a553830bcf66609d2202_sam18_full.png) · [Crop](../../results/phase7/phase7_v1/gallery/b9e5a553830bcf66609d2202_sam18_crop.png) · [Record](../../results/phase7/phase7_v1/case_records/b9e5a553830bcf66609d2202_sam18.json) |

Blue = hand; yellow = object; grey hatch = ignore; red = object FP; cyan = object FN; magenta = exact eligible SAM region. Object-error panels include hand/object confusion; the selected SAM regions specifically contain GT background. The headlamp part is candidate 60 (3,090 pixels); the packaging-looking proposal is candidate 18 (3,541 pixels).

The original SAM-panel captions include the earlier phrase “semantic extent awaits the reconstructed outline.” Those frozen captions are retained. Both displayed regions were subsequently inspected; use the [post-render review](../../results/phase7/phase7_v1/preview_review.md) for the current assessment. A proposal remains distinct from certified semantic untouched-object ground truth.

## Plots

| Copy | Complete original and numerical evidence | Population |
|---|---|---|
| [Background FP/distance](background-fp-distance.png) | [PNG](../../results/clutter_fp_vs_distance.png) · [JSON](../../results/clutter_fp_vs_distance.json) · [Report](../../results/phase6_report.md) | All 102,774 retained SAM regions; shared denominators, split-separated two-seed rates, strict object fraction >0.5. Curves use defined GT-hand distances. |
| [Source comparison](source-comparison.png) | [PNG](../../results/phase7/phase7_v1/source_comparison.png) · [Record](../../results/phase7/phase7_v1/source_metrics.json) | Four-run defined-score intersection. Dots are mean seed-averaged image IoUs; bars are image IQRs, not CIs or pooled source IoUs. Object counts: Ego4D 754 / EPIC 212 / own 44 / THU 85 / YouTube 496. |

Only four actual prediction cases are available here. The 16-case selection spans five sources, but these rendered examples span four; THU's selected cases remain unrendered. Neither these copies nor this write-up complete the original gallery or record the user's acceptance. All files are for local review. Publication rights for dataset-derived images remain unresolved; no license is assigned by this manifest.
