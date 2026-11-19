# Phase 5: hand-guided two-stage cascade

IoU × 100. Same Phase-2 original-resolution scoring, 255 excluded, Boundary IoU d=2% of diagonal.
Final weights, no checkpoint selection. Baseline JSONs are reused without inference.
Stage 2: ImageNet MiT-B0, RGB + binary hand input, fresh binary object head; hands are background in its loss.
Oracle uses GT for both the input channel and pasted hands. All cascade seeds share frozen seed-0 stage 1.
This compares complete systems; task/head and compute also differ from the single-stage baseline.

## val

| run | IoU bg | IoU hand | IoU object | mIoU | mIoU_fg | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|
| b0_full_ft (single-stage) | 94.0 | 82.3 | 44.7 | 73.7 | 63.5 | 69.0 | 33.6 |
| b0_full_ft_s1 (single-stage) | 94.1 | 82.1 | 44.2 | 73.5 | 63.2 | 68.8 | 33.7 |
| **single-stage mean** | **94.1** | **82.2** | **44.4** | **73.6** | **63.3** | **68.9** | **33.7** |
| b0_cascade_stage2 (predicted) | 94.3 | 84.6 | 46.6 | 75.2 | 65.6 | 72.0 | 35.2 |
| b0_cascade_stage2_s1 (predicted) | 94.4 | 84.6 | 46.6 | 75.2 | 65.6 | 72.0 | 35.0 |
| **predicted mean** | **94.4** | **84.6** | **46.6** | **75.2** | **65.6** | **72.0** | **35.1** |
| b0_cascade_stage2 (oracle) | 96.0 | 100.0 | 48.6 | 81.5 | 74.3 | 100.0 | 38.4 |
| b0_cascade_stage2_s1 (oracle) | 96.0 | 100.0 | 48.5 | 81.5 | 74.2 | 100.0 | 37.9 |
| **oracle mean** | **96.0** | **100.0** | **48.5** | **81.5** | **74.3** | **100.0** | **38.2** |

## test in-domain

| run | IoU bg | IoU hand | IoU object | mIoU | mIoU_fg | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|
| b0_full_ft (single-stage) | 93.7 | 82.4 | 45.9 | 74.0 | 64.1 | 68.5 | 34.3 |
| b0_full_ft_s1 (single-stage) | 93.8 | 82.3 | 46.5 | 74.2 | 64.4 | 68.3 | 33.7 |
| **single-stage mean** | **93.8** | **82.4** | **46.2** | **74.1** | **64.3** | **68.4** | **34.0** |
| b0_cascade_stage2 (predicted) | 93.8 | 84.5 | 46.7 | 75.0 | 65.6 | 71.8 | 34.9 |
| b0_cascade_stage2_s1 (predicted) | 94.0 | 84.5 | 46.6 | 75.0 | 65.5 | 71.8 | 35.0 |
| **predicted mean** | **93.9** | **84.5** | **46.6** | **75.0** | **65.6** | **71.8** | **34.9** |
| b0_cascade_stage2 (oracle) | 95.7 | 100.0 | 49.3 | 81.7 | 74.7 | 100.0 | 38.2 |
| b0_cascade_stage2_s1 (oracle) | 95.8 | 100.0 | 49.2 | 81.7 | 74.6 | 100.0 | 38.4 |
| **oracle mean** | **95.8** | **100.0** | **49.3** | **81.7** | **74.6** | **100.0** | **38.3** |

## test out-of-domain (YouTube)

| run | IoU bg | IoU hand | IoU object | mIoU | mIoU_fg | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|
| b0_full_ft (single-stage) | 96.1 | 88.4 | 58.2 | 80.9 | 73.3 | 79.9 | 44.6 |
| b0_full_ft_s1 (single-stage) | 96.1 | 88.3 | 58.1 | 80.8 | 73.2 | 79.9 | 44.0 |
| **single-stage mean** | **96.1** | **88.3** | **58.1** | **80.8** | **73.2** | **79.9** | **44.3** |
| b0_cascade_stage2 (predicted) | 96.2 | 89.7 | 59.6 | 81.8 | 74.7 | 82.2 | 45.5 |
| b0_cascade_stage2_s1 (predicted) | 96.2 | 89.7 | 59.9 | 81.9 | 74.8 | 82.2 | 45.4 |
| **predicted mean** | **96.2** | **89.7** | **59.7** | **81.9** | **74.7** | **82.2** | **45.5** |
| b0_cascade_stage2 (oracle) | 97.1 | 100.0 | 61.7 | 86.3 | 80.8 | 100.0 | 48.8 |
| b0_cascade_stage2_s1 (oracle) | 97.2 | 100.0 | 62.0 | 86.4 | 81.0 | 100.0 | 48.7 |
| **oracle mean** | **97.2** | **100.0** | **61.8** | **86.3** | **80.9** | **100.0** | **48.8** |

## Three object-IoU numbers and differences

| split | single-stage | predicted cascade | oracle cascade | predicted − baseline | oracle − predicted |
|---|---|---|---|---|---|
| val | 44.4 | 46.6 | 48.5 | +2.2 | +1.9 |
| test_indomain | 46.2 | 46.6 | 49.3 | +0.4 | +2.6 |
| test_outdomain | 58.1 | 59.7 | 61.8 | +1.6 | +2.1 |

## Observed seed spreads

| split | system | seeds | min object IoU | max | spread |
|---|---|---|---|---|---|
| val | single-stage | 2 | 44.2 | 44.7 | 0.5 |
| val | predicted | 2 | 46.6 | 46.6 | 0.0 |
| val | oracle | 2 | 48.5 | 48.6 | 0.1 |
| test_indomain | single-stage | 2 | 45.9 | 46.5 | 0.7 |
| test_indomain | predicted | 2 | 46.6 | 46.7 | 0.1 |
| test_indomain | oracle | 2 | 49.2 | 49.3 | 0.1 |
| test_outdomain | single-stage | 2 | 58.1 | 58.2 | 0.1 |
| test_outdomain | predicted | 2 | 59.6 | 59.9 | 0.3 |
| test_outdomain | oracle | 2 | 61.7 | 62.0 | 0.3 |

Two seeds show observed variability; their spread is not a bound on future seed noise.
Compare systems within each split. Higher YouTube scores are not evidence of a domain-gap improvement.

## Correctness checks and oracle diagnostic

- PASS: val, b0_cascade_stage2: hand iou equals b0_hands_only exactly.
- PASS: val, b0_cascade_stage2: hand boundary_iou equals b0_hands_only exactly.
- PASS: val, b0_cascade_stage2_s1: hand iou equals b0_hands_only exactly.
- PASS: val, b0_cascade_stage2_s1: hand boundary_iou equals b0_hands_only exactly.
- PASS: val, b0_cascade_stage2: oracle hand IoU is 100%.
- Diagnostic: val, b0_cascade_stage2: oracle − predicted object IoU +1.9 points.
- PASS: val, b0_cascade_stage2_s1: oracle hand IoU is 100%.
- Diagnostic: val, b0_cascade_stage2_s1: oracle − predicted object IoU +1.9 points.
- PASS: test_indomain, b0_cascade_stage2: hand iou equals b0_hands_only exactly.
- PASS: test_indomain, b0_cascade_stage2: hand boundary_iou equals b0_hands_only exactly.
- PASS: test_indomain, b0_cascade_stage2_s1: hand iou equals b0_hands_only exactly.
- PASS: test_indomain, b0_cascade_stage2_s1: hand boundary_iou equals b0_hands_only exactly.
- PASS: test_indomain, b0_cascade_stage2: oracle hand IoU is 100%.
- Diagnostic: test_indomain, b0_cascade_stage2: oracle − predicted object IoU +2.7 points.
- PASS: test_indomain, b0_cascade_stage2_s1: oracle hand IoU is 100%.
- Diagnostic: test_indomain, b0_cascade_stage2_s1: oracle − predicted object IoU +2.6 points.
- PASS: test_outdomain, b0_cascade_stage2: hand iou equals b0_hands_only exactly.
- PASS: test_outdomain, b0_cascade_stage2: hand boundary_iou equals b0_hands_only exactly.
- PASS: test_outdomain, b0_cascade_stage2_s1: hand iou equals b0_hands_only exactly.
- PASS: test_outdomain, b0_cascade_stage2_s1: hand boundary_iou equals b0_hands_only exactly.
- PASS: test_outdomain, b0_cascade_stage2: oracle hand IoU is 100%.
- Diagnostic: test_outdomain, b0_cascade_stage2: oracle − predicted object IoU +2.1 points.
- PASS: test_outdomain, b0_cascade_stage2_s1: oracle hand IoU is 100%.
- Diagnostic: test_outdomain, b0_cascade_stage2_s1: oracle − predicted object IoU +2.1 points.

Oracle ≥ predicted object IoU is an expectation, not a mathematical guarantee for a fixed model.
A reversal needs investigation; it is not an automatic correctness failure.
