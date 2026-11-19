# Phase 7: source analysis and selected failure gallery

Analysis `phase7_v1`; fingerprint `7cca8b230189079b20e3141a5c29bc6fbcdf99183452ee370aa7d8318a60d214`.

Status: saved-result analysis computed. CPU label scan pending. Real panels, category captions and final source comparisons require user verification. Phase 7 is not complete.

The primary quantitative population is the complete 1,126-image in-domain and 500-image YouTube test splits. Both baseline seeds and both predicted-cascade stage-2 seeds are included. Both cascades share frozen seed-0 stage 1. Existing val, head-only, hands-only and oracle metrics are context; they add no inference.

Input shorter side 512; saved bf16; fp32 bilinear logit upsampling to original resolution, align_corners=False, then argmax. Original GT is background/hand/interacting object; second-order IDs 6–8 become 255 and are excluded. Split IoU is pooled from confusion counts; Boundary IoU pools boundary intersections/unions with width 2% of the original diagonal.

## Saved pooled split metrics

| split | run/system | mode | images | IoU bg | IoU hand | IoU object | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|---|
| val | b0_full_ft | single | 1124 | 94.03 | 82.27 | 44.67 | 68.95 | 33.62 |
| val | b0_full_ft_s1 | single | 1124 | 94.07 | 82.15 | 44.18 | 68.83 | 33.68 |
| val | b0_cascade_stage2 | predicted | 1124 | 94.31 | 84.58 | 46.63 | 72.05 | 35.23 |
| val | b0_cascade_stage2 | oracle | 1124 | 95.98 | 100.00 | 48.58 | 100.00 | 38.37 |
| val | b0_cascade_stage2_s1 | predicted | 1124 | 94.40 | 84.58 | 46.63 | 72.05 | 34.98 |
| val | b0_cascade_stage2_s1 | oracle | 1124 | 96.04 | 100.00 | 48.48 | 100.00 | 37.94 |
| val | b0_head_only | single | 1124 | 90.62 | 66.11 | 19.05 | 52.48 | 16.43 |
| val | b0_hands_only | single | 1124 | 98.09 | 84.58 | N/A | 72.05 | N/A |
| val | b0_hands_only_s1 | single | 1124 | 98.12 | 84.74 | N/A | 72.21 | N/A |
| val | baseline mean | seeds 0,1 | 1124 | 94.05 | 82.21 | 44.42 | 68.89 | 33.65 |
| val | predicted cascade mean | seeds 0,1 | 1124 | 94.36 | 84.58 | 46.63 | 72.05 | 35.11 |
| val | oracle mean | seeds 0,1 | 1124 | 96.01 | 100.00 | 48.53 | 100.00 | 38.15 |
| test_indomain | b0_full_ft | single | 1126 | 93.73 | 82.40 | 45.85 | 68.48 | 34.30 |
| test_indomain | b0_full_ft_s1 | single | 1126 | 93.82 | 82.33 | 46.52 | 68.29 | 33.68 |
| test_indomain | b0_cascade_stage2 | predicted | 1126 | 93.83 | 84.50 | 46.66 | 71.80 | 34.89 |
| test_indomain | b0_cascade_stage2 | oracle | 1126 | 95.70 | 100.00 | 49.32 | 100.00 | 38.17 |
| test_indomain | b0_cascade_stage2_s1 | predicted | 1126 | 93.98 | 84.50 | 46.60 | 71.80 | 34.96 |
| test_indomain | b0_cascade_stage2_s1 | oracle | 1126 | 95.85 | 100.00 | 49.18 | 100.00 | 38.37 |
| test_indomain | b0_head_only | single | 1126 | 90.15 | 67.03 | 17.31 | 52.68 | 15.84 |
| test_indomain | b0_hands_only | single | 1126 | 97.96 | 84.50 | N/A | 71.80 | N/A |
| test_indomain | b0_hands_only_s1 | single | 1126 | 97.98 | 84.64 | N/A | 71.94 | N/A |
| test_indomain | baseline mean | seeds 0,1 | 1126 | 93.77 | 82.37 | 46.18 | 68.39 | 33.99 |
| test_indomain | predicted cascade mean | seeds 0,1 | 1126 | 93.90 | 84.50 | 46.63 | 71.80 | 34.92 |
| test_indomain | oracle mean | seeds 0,1 | 1126 | 95.77 | 100.00 | 49.25 | 100.00 | 38.27 |
| test_outdomain | b0_full_ft | single | 500 | 96.08 | 88.40 | 58.17 | 79.94 | 44.57 |
| test_outdomain | b0_full_ft_s1 | single | 500 | 96.06 | 88.27 | 58.11 | 79.92 | 43.98 |
| test_outdomain | b0_cascade_stage2 | predicted | 500 | 96.22 | 89.72 | 59.58 | 82.22 | 45.53 |
| test_outdomain | b0_cascade_stage2 | oracle | 500 | 97.15 | 100.00 | 61.69 | 100.00 | 48.82 |
| test_outdomain | b0_cascade_stage2_s1 | predicted | 500 | 96.24 | 89.72 | 59.88 | 82.22 | 45.41 |
| test_outdomain | b0_cascade_stage2_s1 | oracle | 500 | 97.15 | 100.00 | 61.98 | 100.00 | 48.75 |
| test_outdomain | b0_head_only | single | 500 | 91.66 | 71.05 | 22.86 | 62.10 | 15.94 |
| test_outdomain | b0_hands_only | single | 500 | 98.94 | 89.72 | N/A | 82.22 | N/A |
| test_outdomain | b0_hands_only_s1 | single | 500 | 98.93 | 89.63 | N/A | 82.23 | N/A |
| test_outdomain | baseline mean | seeds 0,1 | 500 | 96.07 | 88.33 | 58.14 | 79.93 | 44.28 |
| test_outdomain | predicted cascade mean | seeds 0,1 | 500 | 96.23 | 89.72 | 59.73 | 82.22 | 45.47 |
| test_outdomain | oracle mean | seeds 0,1 | 500 | 97.15 | 100.00 | 61.84 | 100.00 | 48.78 |

All scores are ×100. Hands-only background includes objects; compare its hand scores only. Oracle changes both hand input and pasted hands; perfect hand scores are by construction. Means average two final-model metrics, not an ensemble.

## Signed split differences

| system | metric | class | in-domain | YouTube | YouTube − in-domain (points) |
|---|---|---|---|---|---|
| b0_cascade_stage2 (oracle) | iou | hand | 100.00 | 100.00 | +0.00 |
| b0_cascade_stage2 (oracle) | iou | interacting_object | 49.32 | 61.69 | +12.37 |
| b0_cascade_stage2 (oracle) | boundary_iou | hand | 100.00 | 100.00 | +0.00 |
| b0_cascade_stage2 (oracle) | boundary_iou | interacting_object | 38.17 | 48.82 | +10.65 |
| b0_cascade_stage2 (predicted) | iou | hand | 84.50 | 89.72 | +5.22 |
| b0_cascade_stage2 (predicted) | iou | interacting_object | 46.66 | 59.58 | +12.92 |
| b0_cascade_stage2 (predicted) | boundary_iou | hand | 71.80 | 82.22 | +10.42 |
| b0_cascade_stage2 (predicted) | boundary_iou | interacting_object | 34.89 | 45.53 | +10.64 |
| b0_cascade_stage2_s1 (oracle) | iou | hand | 100.00 | 100.00 | +0.00 |
| b0_cascade_stage2_s1 (oracle) | iou | interacting_object | 49.18 | 61.98 | +12.79 |
| b0_cascade_stage2_s1 (oracle) | boundary_iou | hand | 100.00 | 100.00 | +0.00 |
| b0_cascade_stage2_s1 (oracle) | boundary_iou | interacting_object | 38.37 | 48.75 | +10.38 |
| b0_cascade_stage2_s1 (predicted) | iou | hand | 84.50 | 89.72 | +5.22 |
| b0_cascade_stage2_s1 (predicted) | iou | interacting_object | 46.60 | 59.88 | +13.28 |
| b0_cascade_stage2_s1 (predicted) | boundary_iou | hand | 71.80 | 82.22 | +10.42 |
| b0_cascade_stage2_s1 (predicted) | boundary_iou | interacting_object | 34.96 | 45.41 | +10.45 |
| b0_full_ft | iou | hand | 82.40 | 88.40 | +6.00 |
| b0_full_ft | iou | interacting_object | 45.85 | 58.17 | +12.32 |
| b0_full_ft | boundary_iou | hand | 68.48 | 79.94 | +11.46 |
| b0_full_ft | boundary_iou | interacting_object | 34.30 | 44.57 | +10.28 |
| b0_full_ft_s1 | iou | hand | 82.33 | 88.27 | +5.93 |
| b0_full_ft_s1 | iou | interacting_object | 46.52 | 58.11 | +11.59 |
| b0_full_ft_s1 | boundary_iou | hand | 68.29 | 79.92 | +11.63 |
| b0_full_ft_s1 | boundary_iou | interacting_object | 33.68 | 43.98 | +10.30 |
| b0_hands_only | iou | hand | 84.50 | 89.72 | +5.22 |
| b0_hands_only | boundary_iou | hand | 71.80 | 82.22 | +10.42 |
| b0_hands_only_s1 | iou | hand | 84.64 | 89.63 | +4.99 |
| b0_hands_only_s1 | boundary_iou | hand | 71.94 | 82.23 | +10.29 |
| b0_head_only | iou | hand | 67.03 | 71.05 | +4.02 |
| b0_head_only | iou | interacting_object | 17.31 | 22.86 | +5.55 |
| b0_head_only | boundary_iou | hand | 52.68 | 62.10 | +9.43 |
| b0_head_only | boundary_iou | interacting_object | 15.84 | 15.94 | +0.11 |
| baseline mean | iou | hand | 82.37 | 88.33 | +5.97 |
| baseline mean | iou | interacting_object | 46.18 | 58.14 | +11.96 |
| baseline mean | boundary_iou | hand | 68.39 | 79.93 | +11.54 |
| baseline mean | boundary_iou | interacting_object | 33.99 | 44.28 | +10.29 |
| predicted cascade mean | iou | hand | 84.50 | 89.72 | +5.22 |
| predicted cascade mean | iou | interacting_object | 46.63 | 59.73 | +13.10 |
| predicted cascade mean | boundary_iou | hand | 71.80 | 82.22 | +10.42 |
| predicted cascade mean | boundary_iou | interacting_object | 34.92 | 45.47 | +10.54 |
| oracle mean | iou | hand | 100.00 | 100.00 | +0.00 |
| oracle mean | iou | interacting_object | 49.25 | 61.84 | +12.58 |
| oracle mean | boundary_iou | hand | 100.00 | 100.00 | +0.00 |
| oracle mean | boundary_iou | interacting_object | 38.27 | 48.78 | +10.51 |

Object IoU is higher on YouTube for every object-predicting system; oracle hands remain perfect by construction. These different source/content mixtures do not isolate a causal domain effect or demonstrate a closed domain gap. The small in-domain object-IoU gain is smaller than the observed baseline seed spread (about 0.7 points). Two seeds do not establish significance.

## Source-level per-image summaries

These are unweighted image summaries, not pooled source IoU. Image dimensions cannot reconstruct the missing intersections/unions. No per-source/per-image Boundary IoU is invented. Each per-run row uses that run's defined scores; its valid count can differ. A null IoU means absent in both GT and prediction; zero remains zero and can include predictions on GT-absent images.

| split | source | images | raw filename groups | conservative diversity groups |
|---|---|---|---|---|
| test_indomain | ego4d | 774 | 455 | 455 |
| test_indomain | epic | 217 | 185 | 185 |
| test_indomain | own | 47 | 14 | 14 |
| test_indomain | thu | 88 | 88 | 88 |
| test_outdomain | youtube | 500 | 27 | 24 |

Grouping removes explicit download-copy suffixes only when a matching embedded YouTube ID and original filename group exist. Raw groups remain in provenance. Neither grouping proves video independence or duplicate-content removal.

### hand

| source/split | run | valid / total | undefined | mean | median | Q25 | Q75 |
|---|---|---|---|---|---|---|---|
| ego4d / test_indomain | b0_full_ft | 774/774 | 0 | 81.18 | 84.21 | 77.34 | 88.99 |
| ego4d / test_indomain | b0_full_ft_s1 | 774/774 | 0 | 81.04 | 84.16 | 77.11 | 89.05 |
| ego4d / test_indomain | b0_cascade_stage2 | 774/774 | 0 | 83.02 | 86.19 | 79.42 | 90.30 |
| ego4d / test_indomain | b0_cascade_stage2_s1 | 774/774 | 0 | 83.02 | 86.19 | 79.42 | 90.30 |
| epic / test_indomain | b0_full_ft | 217/217 | 0 | 88.37 | 90.26 | 87.08 | 92.34 |
| epic / test_indomain | b0_full_ft_s1 | 217/217 | 0 | 88.17 | 90.36 | 87.02 | 92.28 |
| epic / test_indomain | b0_cascade_stage2 | 217/217 | 0 | 89.06 | 90.95 | 88.37 | 93.13 |
| epic / test_indomain | b0_cascade_stage2_s1 | 217/217 | 0 | 89.06 | 90.95 | 88.37 | 93.13 |
| own / test_indomain | b0_full_ft | 47/47 | 0 | 81.65 | 85.72 | 77.53 | 89.73 |
| own / test_indomain | b0_full_ft_s1 | 47/47 | 0 | 81.07 | 85.01 | 77.70 | 88.88 |
| own / test_indomain | b0_cascade_stage2 | 47/47 | 0 | 84.58 | 87.72 | 82.48 | 91.98 |
| own / test_indomain | b0_cascade_stage2_s1 | 47/47 | 0 | 84.58 | 87.72 | 82.48 | 91.98 |
| thu / test_indomain | b0_full_ft | 88/88 | 0 | 81.50 | 84.30 | 78.34 | 87.82 |
| thu / test_indomain | b0_full_ft_s1 | 88/88 | 0 | 81.51 | 84.31 | 77.90 | 87.34 |
| thu / test_indomain | b0_cascade_stage2 | 88/88 | 0 | 83.80 | 85.74 | 80.31 | 88.63 |
| thu / test_indomain | b0_cascade_stage2_s1 | 88/88 | 0 | 83.80 | 85.74 | 80.31 | 88.63 |
| youtube / test_outdomain | b0_full_ft | 500/500 | 0 | 88.29 | 90.73 | 87.18 | 92.87 |
| youtube / test_outdomain | b0_full_ft_s1 | 500/500 | 0 | 88.12 | 90.78 | 87.42 | 92.91 |
| youtube / test_outdomain | b0_cascade_stage2 | 500/500 | 0 | 89.68 | 91.81 | 89.03 | 93.63 |
| youtube / test_outdomain | b0_cascade_stage2_s1 | 500/500 | 0 | 89.68 | 91.81 | 89.03 | 93.63 |

Paired comparisons use the intersection where all four runs have a defined score. For each image, average the two same-seed cascade-minus-baseline differences, then average images equally. Seed-specific deltas below use exactly this shared cohort; same seed numbers do not imply matched training randomness.

| source | eligible | excluded | baseline mean | cascade mean | mean Δ (points) | seed-0 Δ | seed-1 Δ |
|---|---|---|---|---|---|---|---|
| ego4d | 774 | 0 | 81.11 | 83.02 | +1.91 | +1.84 | +1.98 |
| epic | 217 | 0 | 88.27 | 89.06 | +0.79 | +0.70 | +0.89 |
| own | 47 | 0 | 81.36 | 84.58 | +3.22 | +2.93 | +3.51 |
| thu | 88 | 0 | 81.51 | 83.80 | +2.29 | +2.30 | +2.29 |
| youtube | 500 | 0 | 88.21 | 89.68 | +1.48 | +1.39 | +1.56 |

### interacting_object

| source/split | run | valid / total | undefined | mean | median | Q25 | Q75 |
|---|---|---|---|---|---|---|---|
| ego4d / test_indomain | b0_full_ft | 759/774 | 15 | 40.02 | 40.54 | 12.29 | 65.54 |
| ego4d / test_indomain | b0_full_ft_s1 | 761/774 | 13 | 39.58 | 41.48 | 12.21 | 65.62 |
| ego4d / test_indomain | b0_cascade_stage2 | 761/774 | 13 | 41.38 | 41.72 | 14.56 | 66.48 |
| ego4d / test_indomain | b0_cascade_stage2_s1 | 762/774 | 12 | 41.09 | 42.02 | 13.41 | 67.40 |
| epic / test_indomain | b0_full_ft | 215/217 | 2 | 44.97 | 44.09 | 17.88 | 72.50 |
| epic / test_indomain | b0_full_ft_s1 | 214/217 | 3 | 45.60 | 45.19 | 17.69 | 73.38 |
| epic / test_indomain | b0_cascade_stage2 | 216/217 | 1 | 46.22 | 48.06 | 21.75 | 70.44 |
| epic / test_indomain | b0_cascade_stage2_s1 | 216/217 | 1 | 46.61 | 51.40 | 18.42 | 74.85 |
| own / test_indomain | b0_full_ft | 45/47 | 2 | 61.70 | 74.34 | 41.28 | 83.11 |
| own / test_indomain | b0_full_ft_s1 | 46/47 | 1 | 61.51 | 72.15 | 44.03 | 84.34 |
| own / test_indomain | b0_cascade_stage2 | 44/47 | 3 | 62.32 | 71.97 | 47.70 | 85.51 |
| own / test_indomain | b0_cascade_stage2_s1 | 44/47 | 3 | 64.85 | 77.06 | 55.39 | 84.64 |
| thu / test_indomain | b0_full_ft | 85/88 | 3 | 57.63 | 67.94 | 35.53 | 85.52 |
| thu / test_indomain | b0_full_ft_s1 | 85/88 | 3 | 57.08 | 66.76 | 30.54 | 85.84 |
| thu / test_indomain | b0_cascade_stage2 | 86/88 | 2 | 59.51 | 71.95 | 37.30 | 86.09 |
| thu / test_indomain | b0_cascade_stage2_s1 | 86/88 | 2 | 58.97 | 71.41 | 36.89 | 86.12 |
| youtube / test_outdomain | b0_full_ft | 498/500 | 2 | 57.50 | 67.36 | 35.70 | 81.29 |
| youtube / test_outdomain | b0_full_ft_s1 | 499/500 | 1 | 56.97 | 64.94 | 36.41 | 80.41 |
| youtube / test_outdomain | b0_cascade_stage2 | 498/500 | 2 | 59.32 | 67.47 | 39.34 | 82.84 |
| youtube / test_outdomain | b0_cascade_stage2_s1 | 498/500 | 2 | 59.30 | 66.96 | 40.12 | 83.00 |

Paired comparisons use the intersection where all four runs have a defined score. For each image, average the two same-seed cascade-minus-baseline differences, then average images equally. Seed-specific deltas below use exactly this shared cohort; same seed numbers do not imply matched training randomness.

| source | eligible | excluded | baseline mean | cascade mean | mean Δ (points) | seed-0 Δ | seed-1 Δ |
|---|---|---|---|---|---|---|---|
| ego4d | 754 | 20 | 40.12 | 41.64 | +1.53 | +1.47 | +1.58 |
| epic | 212 | 5 | 45.82 | 47.29 | +1.47 | +1.48 | +1.45 |
| own | 44 | 3 | 63.71 | 63.59 | -0.12 | -0.78 | +0.55 |
| thu | 85 | 3 | 57.35 | 59.94 | +2.59 | +2.59 | +2.59 |
| youtube | 496 | 4 | 57.52 | 59.55 | +2.03 | +1.83 | +2.23 |

## Pixel precision/recall diagnostic

Computed from the saved three-class split confusion matrices (GT rows, prediction columns), with 255 excluded. These pixel metrics are distinct from the fixed SAM-region majority FP measure and do not establish a causal explanation.

| split | run | object pixel precision | object pixel recall |
|---|---|---|---|
| test_indomain | b0_full_ft | 69.94 | 57.11 |
| test_outdomain | b0_full_ft | 82.11 | 66.62 |
| test_indomain | b0_full_ft_s1 | 70.86 | 57.52 |
| test_outdomain | b0_full_ft_s1 | 82.44 | 66.32 |
| test_indomain | b0_cascade_stage2 | 67.67 | 60.05 |
| test_outdomain | b0_cascade_stage2 | 81.65 | 68.79 |
| test_indomain | b0_cascade_stage2_s1 | 70.39 | 57.96 |
| test_outdomain | b0_cascade_stage2_s1 | 81.36 | 69.40 |

## Frozen Phase-6 context

FP means strictly more than half of a retained eligible background proposal is predicted as composed object class 2. The candidate denominator is shared by all models. Proposals include correlated/nested regions, surfaces, clothing and object parts, not independently labelled untouched objects.

| split | images | regions | baseline mean | cascade mean | cascade − baseline (pp) |
|---|---|---|---|---|---|
| test_indomain | 1126 | 68160 | 1.15% | 1.26% | +0.11 |
| test_outdomain | 500 | 34614 | 0.65% | 0.75% | +0.10 |

This fixed proxy does not support reduced cascade clutter FP. Its higher FP rate can coexist with higher object IoU. Thresholds, filters, seeds and completed results are unchanged.

## Selected qualitative gallery

The shortlist is deterministic exploratory diagnosis after the frozen evaluation: source support, low object IoU, paired improvements/regressions, GT small components, original ID 5 and both-seed-consistent SAM disagreements. At most two frames per conservative video group. Small means an 8-connected GT object component at most 1% of original frame area; components are not independent object instances. ID 5 suggests two-hand interaction, with visual review still required. Appearance categories are manually reviewed and may be unsupported.

Case review/selection is pending. No failure category has been confirmed.

Gallery computation: pending.

Panels: original RGB / baseline / predicted cascade over original GT / baseline object errors / cascade object errors. Blue=hand, yellow=object, grey hatch=ignore; red=object FP, cyan=object FN. Object FP includes hand/object confusion; SAM outlines identify specifically eligible GT-background regions. Full-frame and matched close-up views share coordinates. Display uses seed 0 for both systems; both-seed scores/consistency are recorded. Captions describe evidence; mechanisms remain hypotheses.

## Provenance and limits

Machine-readable files: `phase7/phase7_v1/manifest.json`, `source_metrics.json`, `request.json`, and, after the CPU scan, `gt_stats.json`, `shortlist.json` and the editable `review.json`. Selected cases are in `case_selection.json`; panels and per-case records are fetched separately.

Manifest hashes bind saved configs/evaluations, all four final checkpoints and shared stage 1, original image/label identities, Phase-6 measurements and both analysis/proposal fingerprints through its sealed manifest, and new Phase-7 code. Dense predictions and SAM bitmaps are not reconstructed from summary JSONs. Only selected predictions are recomputed; completed run directories remain untouched.

Compared systems differ in task/head and compute. No binary RGB ablation isolates the hand channel. No confidence intervals, significance, category prevalence or video independence is claimed. Final user verification is pending.
