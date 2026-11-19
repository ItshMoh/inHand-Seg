# Phase-7 verification and remaining visual work

Audit date: 2026-11-19. Analysis fingerprint:
`d95f14a370407ff1c0aacdda4dfd3639c5188a880c9178b19175c289b21418f9`.

## Verified computed outputs

- All sealed Phase-6/7 records checked during reconstruction have valid checksums.
  Saved run configs/evaluations, Phase-7 code hashes and frozen Phase-6 source
  hashes reconcile with the prepared manifest. No frozen analysis code was changed.
- The original-label scan covers exactly 1,626 distinct `(split, original name)`
  identities: 1,126 in-domain and 500 YouTube. Label/image identities, original
  sizes, class counts, ID-5 counts and connected-component geometry validate.
- Aggregated background/hand/object pixel counts equal the GT row sums of all
  eight primary saved split confusion matrices (four runs times two splits).
  Ignored pixels are accounted for separately.
- All source summaries, per-run valid counts, shared paired-image populations,
  GT-present/absent cohorts, pooled split/context metrics, signed differences and
  pixel precision/recall reproduce exactly from saved inputs.
- The deterministic 60-image shortlist reproduces exactly, covers all five
  sources and respects the video cap. All 61 survey-file checksums pass; all 60
  RGB/GT PNGs decode. Their RGB and GT halves share display geometry.
- Before preparing the gallery draft, the saved Markdown report reproduced
  byte-for-byte. It has since been rebuilt with the reviewed draft selection;
  quantitative values and the analysis/code identity are unchanged.
- Read-only inspection of `/data/analysis/phase7_v1` confirmed survey outputs
  only. Missing gallery files were not merely an incomplete local fetch. The
  user clarified that only survey/fetch/build had run and requested help with
  the gallery.

| test split | images | GT object present | GT object absent | ignored pixels |
|---|---:|---:|---:|---:|
| test_indomain | 1,126 | 1,037 | 89 | 62,132,490 |
| test_outdomain | 500 | 480 | 20 | 59,328,416 |

The 1,095/496 shared-defined object-score cohorts are not GT-presence counts:
they include some GT-absent images with false predictions and valid zero IoUs.

| source | GT-present images | GT-absent images | GT-present mean paired object-IoU delta, points |
|---|---:|---:|---:|
| Ego4D | 714 | 60 | +1.61 |
| EPIC | 204 | 13 | +1.53 |
| own | 42 | 5 | -0.13 |
| THU | 77 | 11 | +2.86 |
| YouTube | 480 | 20 | +2.10 |

These are unweighted per-image differences, averaged across the two same-seed
pairs, on the original-label GT-present cohort. They are not pooled source IoUs.

## Reviewed gallery draft

The assistant inspected all 60 RGB/GT survey thumbnails, with selected individual
views, and prepared 16 cases across five sources. The original untouched review
template is preserved in `review.before_gallery_review.json`. The editable
`review.json` and sealed `case_selection.json` contain the draft captions, saved
both-seed metrics, evidence and four exact SAM candidate IDs.

Four cases are marked for the initial prediction/layout preview:

| source | original image | purpose |
|---|---|---|
| own | `escape_GX010035_006540` | 3840×2160 input; displayed-seed regression with an opposite seed-1 change |
| YouTube | `youtube_POV Cooking MONGOLIAN BEEF NOODLES  JONAH IN THE KITCHEN-seV_bNo_fLE-1080p-1644528285252_003060` | improvement at both seeds |
| Ego4D | `ego4d_11f51f72-c0bd-4631-aabb-6ecbe17d95c9_25950` | cascade-only SAM event, candidate 60, adjacent headlamp-part context |
| EPIC | `epic_P07_101_frame_0000018721` | 456×256 input; improvement and baseline-only SAM event, candidate 18 |

The full draft contains two cascade-only and two baseline-only SAM diagnostics,
using the same saved regions and strict majority threshold. Their proposed surface,
part and packaging-looking descriptions remain subject to inspection of the actual
reconstructed eligible masks. Bounding boxes were not used as mask substitutes.

Small-object, finger-occlusion, two-hand and lighting tags describe visible RGB/GT
features in this draft. They do not establish causes of errors. Transparent-object
and motion-blur categories remain uncertain: clear background containers do not
establish an interacting transparent-object failure, and single soft frames do not
resolve motion versus defocus. Missing/uncertain categories are retained honestly.

## Remaining gates

1. **Computed, fetched and assistant-verified:** four actual preview cases, with
   full-frame/matched-crop panels. All 16 model/image records reproduce saved IoUs,
   original GT counts and FP/FN arithmetic; both SAM regions reproduce the saved
   fractions and metadata. All 13 referenced file hashes and eight PNGs validate.
   The assistant inspected all eight panels; details are in `preview_review.md`.
2. **User acceptance pending:** personally inspect the real `gallery/preview.html`
   layout, seed labels, ignore/error colours, book close-up and exact SAM
   outlines/captions. Accept these four previews before running `approve-preview`
   and full rendering. The assistant has not recorded user acceptance.
3. Render/fetch the remaining selected cases and inspect the real categories,
   improvements, regressions, seed disagreement and background-proposal semantics.
4. User verifies the final gallery and source report. Only then mark Phase 7
   complete and proceed with Phase-8 write-up.

Four real Phase-7 prediction previews exist and pass the assistant's checks.
User preview acceptance, the remaining 12 selected cases and final verification
are pending. Source analysis and the CPU survey are verified; Phase 7 is incomplete.
