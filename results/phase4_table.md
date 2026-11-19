# Phase 4: single-stage SegFormer-B0 on the full EgoHOS training set

Built by `scripts/phase4_table.py` from `results/runs/<run>/eval/<split>.json`. All numbers are IoU x 100.
Protocol (Phase 2): input shorter side 512 px, logits upsampled bilinearly to the original resolution, scored
against the original label; 255 (2nd-order objects) excluded. Dataset-level confusion matrix. Boundary IoU
(BIoU) with d = 2% of the image diagonal. The final model of each run is reported (no checkpoint selection).

## Runs

| run | model | classes | encoder | seed | iterations | train min | img/s | data wait | trainable |
|---|---|---|---|---|---|---|---|---|---|
| `b0_head_only` | head-only | bg, hand, object | frozen | 0 | 11240 | 10.2 | 401.0 | 4.6% | 0.4M |
| `b0_full_ft` | full fine-tune | bg, hand, object | trained | 0 | 11240 | 13.1 | 285.1 | 3.7% | 3.71M |
| `b0_full_ft_s1` | full fine-tune | bg, hand, object | trained | 1 | 11240 | 13.0 | 285.2 | 3.6% | 3.71M |
| `b0_hands_only` | hands-only | bg, hand | trained | 0 | 11240 | 16.4 | 283.8 | 4.8% | 3.71M |
| `b0_hands_only_s1` | hands-only | bg, hand | trained | 1 | 11240 | 14.4 | 269.0 | 4.5% | 3.71M |

Hands-only maps interacting object -> background (`label_map: {2: 0}`); 2nd-order objects stay ignored,
so its hand IoU is scored on exactly the same pixels as the 3-class runs.

## Results per split

### val (1,124 images)

| run | IoU bg | IoU hand | IoU object | mIoU | mIoU_fg | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|
| `b0_head_only` | 90.6 | 66.1 | 19.0 | 58.6 | 42.6 | 52.5 | 16.4 |
| `b0_full_ft` | 94.0 | 82.3 | 44.7 | 73.7 | 63.5 | 69.0 | 33.6 |
| `b0_full_ft_s1` | 94.1 | 82.1 | 44.2 | 73.5 | 63.2 | 68.8 | 33.7 |
| **b0_full_ft mean (seeds 0, 1)** | **94.1** | **82.2** | **44.4** | **73.6** | **63.3** | **68.9** | **33.7** |
| `b0_hands_only` | 98.1 | 84.6 | - | 91.3 | 84.6 | 72.0 | - |
| `b0_hands_only_s1` | 98.1 | 84.7 | - | 91.4 | 84.7 | 72.2 | - |
| **b0_hands_only mean (seeds 0, 1)** | **98.1** | **84.7** | **-** | **91.4** | **84.7** | **72.1** | **-** |

### test in-domain (1,126 images)

| run | IoU bg | IoU hand | IoU object | mIoU | mIoU_fg | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|
| `b0_head_only` | 90.1 | 67.0 | 17.3 | 58.2 | 42.2 | 52.7 | 15.8 |
| `b0_full_ft` | 93.7 | 82.4 | 45.9 | 74.0 | 64.1 | 68.5 | 34.3 |
| `b0_full_ft_s1` | 93.8 | 82.3 | 46.5 | 74.2 | 64.4 | 68.3 | 33.7 |
| **b0_full_ft mean (seeds 0, 1)** | **93.8** | **82.4** | **46.2** | **74.1** | **64.3** | **68.4** | **34.0** |
| `b0_hands_only` | 98.0 | 84.5 | - | 91.2 | 84.5 | 71.8 | - |
| `b0_hands_only_s1` | 98.0 | 84.6 | - | 91.3 | 84.6 | 71.9 | - |
| **b0_hands_only mean (seeds 0, 1)** | **98.0** | **84.6** | **-** | **91.3** | **84.6** | **71.9** | **-** |

### test out-of-domain (YouTube) (500 images)

| run | IoU bg | IoU hand | IoU object | mIoU | mIoU_fg | BIoU hand | BIoU object |
|---|---|---|---|---|---|---|---|
| `b0_head_only` | 91.7 | 71.1 | 22.9 | 61.9 | 47.0 | 62.1 | 15.9 |
| `b0_full_ft` | 96.1 | 88.4 | 58.2 | 80.9 | 73.3 | 79.9 | 44.6 |
| `b0_full_ft_s1` | 96.1 | 88.3 | 58.1 | 80.8 | 73.2 | 79.9 | 44.0 |
| **b0_full_ft mean (seeds 0, 1)** | **96.1** | **88.3** | **58.1** | **80.8** | **73.2** | **79.9** | **44.3** |
| `b0_hands_only` | 98.9 | 89.7 | - | 94.3 | 89.7 | 82.2 | - |
| `b0_hands_only_s1` | 98.9 | 89.6 | - | 94.3 | 89.6 | 82.2 | - |
| **b0_hands_only mean (seeds 0, 1)** | **98.9** | **89.7** | **-** | **94.3** | **89.7** | **82.2** | **-** |

mIoU and mIoU_fg average over the run's own classes: for hands-only that is (bg, hand) and hand only,
so compare hands-only through its hand columns, not its mIoU.

## Seed noise

| group | split | metric | seed 0 | seed 1 | max - min |
|---|---|---|---|---|---|
| b0_full_ft | val | IoU hand | 82.3 | 82.1 | 0.1 |
| b0_full_ft | val | IoU object | 44.7 | 44.2 | 0.5 |
| b0_full_ft | val | mIoU_fg | 63.5 | 63.2 | 0.3 |
| b0_full_ft | val | BIoU hand | 69.0 | 68.8 | 0.1 |
| b0_full_ft | val | BIoU object | 33.6 | 33.7 | 0.1 |
| b0_full_ft | test_indomain | IoU hand | 82.4 | 82.3 | 0.1 |
| b0_full_ft | test_indomain | IoU object | 45.9 | 46.5 | 0.7 |
| b0_full_ft | test_indomain | mIoU_fg | 64.1 | 64.4 | 0.3 |
| b0_full_ft | test_indomain | BIoU hand | 68.5 | 68.3 | 0.2 |
| b0_full_ft | test_indomain | BIoU object | 34.3 | 33.7 | 0.6 |
| b0_full_ft | test_outdomain | IoU hand | 88.4 | 88.3 | 0.1 |
| b0_full_ft | test_outdomain | IoU object | 58.2 | 58.1 | 0.1 |
| b0_full_ft | test_outdomain | mIoU_fg | 73.3 | 73.2 | 0.1 |
| b0_full_ft | test_outdomain | BIoU hand | 79.9 | 79.9 | 0.0 |
| b0_full_ft | test_outdomain | BIoU object | 44.6 | 44.0 | 0.6 |
| b0_hands_only | val | IoU hand | 84.6 | 84.7 | 0.2 |
| b0_hands_only | val | BIoU hand | 72.0 | 72.2 | 0.2 |
| b0_hands_only | test_indomain | IoU hand | 84.5 | 84.6 | 0.1 |
| b0_hands_only | test_indomain | BIoU hand | 71.8 | 71.9 | 0.1 |
| b0_hands_only | test_outdomain | IoU hand | 89.7 | 89.6 | 0.1 |
| b0_hands_only | test_outdomain | BIoU hand | 82.2 | 82.2 | 0.0 |

Largest seed spread: 0.7 points. Differences between models smaller than this are noise.

## Stage-1 question: hand from the hands-only model (b0_hands_only) vs the 3-class model (b0_full_ft)

| split | IoU hand b0_full_ft | IoU hand b0_hands_only | diff (c - b) | BIoU hand b0_full_ft | BIoU hand b0_hands_only | diff (c - b) |
|---|---|---|---|---|---|---|
| val | 82.2 (82.3 / 82.1) | 84.7 (84.6 / 84.7) | +2.4 | 68.9 (69.0 / 68.8) | 72.1 (72.0 / 72.2) | +3.2 |
| test_indomain | 82.4 (82.4 / 82.3) | 84.6 (84.5 / 84.6) | +2.2 | 68.4 (68.5 / 68.3) | 71.9 (71.8 / 71.9) | +3.5 |
| test_outdomain | 88.3 (88.4 / 88.3) | 89.7 (89.7 / 89.6) | +1.3 | 79.9 (79.9 / 79.9) | 82.2 (82.2 / 82.2) | +2.3 |

Means over seeds, per-seed values in brackets. Compare the difference with the seed spread above.

## Domain gap: test in-domain -> test out-of-domain

| run | hand in / out / drop | object in / out / drop | mIoU_fg in / out / drop | BIoU hand in / out / drop | BIoU object in / out / drop |
|---|---|---|---|---|---|
| `b0_head_only` | 67.0 / 71.1 / -4.0 (-6%) | 17.3 / 22.9 / -5.5 (-32%) | 42.2 / 47.0 / -4.8 (-11%) | 52.7 / 62.1 / -9.4 (-18%) | 15.8 / 15.9 / -0.1 (-1%) |
| `b0_full_ft` | 82.4 / 88.4 / -6.0 (-7%) | 45.9 / 58.2 / -12.3 (-27%) | 64.1 / 73.3 / -9.2 (-14%) | 68.5 / 79.9 / -11.5 (-17%) | 34.3 / 44.6 / -10.3 (-30%) |
| `b0_full_ft_s1` | 82.3 / 88.3 / -5.9 (-7%) | 46.5 / 58.1 / -11.6 (-25%) | 64.4 / 73.2 / -8.8 (-14%) | 68.3 / 79.9 / -11.6 (-17%) | 33.7 / 44.0 / -10.3 (-31%) |
| **b0_full_ft mean** | 82.4 / 88.3 / -6.0 (-7%) | 46.2 / 58.1 / -12.0 (-26%) | 64.3 / 73.2 / -9.0 (-14%) | 68.4 / 79.9 / -11.5 (-17%) | 34.0 / 44.3 / -10.3 (-30%) |
| `b0_hands_only` | 84.5 / 89.7 / -5.2 (-6%) | - | - | 71.8 / 82.2 / -10.4 (-15%) | - |
| `b0_hands_only_s1` | 84.6 / 89.6 / -5.0 (-6%) | - | - | 71.9 / 82.2 / -10.3 (-14%) | - |
| **b0_hands_only mean** | 84.6 / 89.7 / -5.1 (-6%) | - | - | 71.9 / 82.2 / -10.4 (-14%) | - |

drop = in-domain - out-of-domain in IoU points (relative drop in brackets).

## Reference: EgoHOS paper (arXiv 2208.03826)

The paper reports on the YouTube test set (our test_outdomain) unless stated otherwise, with the best
checkpoint chosen on val. Its labels keep left/right hands and left/right/two-hand objects, and its models
are much larger (ResNet-18 + HRNet, Swin-L + UperNet), so the numbers are a ballpark, not a comparison.

| source | IoU |
|---|---|
| Table 5, Para. Decode: left hand | 69.1 |
| Table 5, Para. Decode: right hand | 73.5 |
| Table 5, Para. Decode: left-hand object | 48.7 |
| Table 5, Para. Decode: right-hand object | 36.2 |
| Table 5, Para. Decode: two-hand object | 37.5 |
| Table 5, Para. Decode: mean of the two hands | 71.3 |
| Table 5, Para. Decode: mean of the three objects | 40.8 |
| Table 3, binary hand, ResNet-18 + HRNet ("mIoU"; unclear if bg is averaged in) | 83.2 |
| ours, `b0_full_ft` (full fine-tune), test_outdomain IoU hand | 88.3 |
| ours, `b0_full_ft` (full fine-tune), test_outdomain IoU object | 58.1 |
| ours, `b0_hands_only` (hands-only), test_outdomain IoU hand | 89.7 |
| ours, `b0_head_only` (head-only), test_outdomain IoU hand | 71.1 |
| ours, `b0_head_only` (head-only), test_outdomain IoU object | 22.9 |

Our classes merge left and right (easier: no left/right confusion) and our model is B0 with 3.7M
parameters (harder). Ours is one merged-class IoU; the paper means above average per-class IoUs.

## Automatic consistency checks

Group means. These are sanity checks, not the verification (look at the numbers and the overlays).

- PASS: val: full fine-tune > head-only on hand IoU
- PASS: val: full fine-tune > head-only on object IoU
- PASS: val: full fine-tune > head-only on mIoU_fg
- PASS: test_indomain: full fine-tune > head-only on hand IoU
- PASS: test_indomain: full fine-tune > head-only on object IoU
- PASS: test_indomain: full fine-tune > head-only on mIoU_fg
- PASS: test_outdomain: full fine-tune > head-only on hand IoU
- PASS: test_outdomain: full fine-tune > head-only on object IoU
- PASS: test_outdomain: full fine-tune > head-only on mIoU_fg
- PASS: val: full fine-tune hand IoU > object IoU
- PASS: test_indomain: full fine-tune hand IoU > object IoU
- PASS: test_outdomain: full fine-tune hand IoU > object IoU
- PASS: val: head-only hand IoU > object IoU
- PASS: test_indomain: head-only hand IoU > object IoU
- PASS: test_outdomain: head-only hand IoU > object IoU
- FAIL: full fine-tune: out-of-domain hand IoU < in-domain
- FAIL: full fine-tune: out-of-domain object IoU < in-domain
- FAIL: head-only: out-of-domain hand IoU < in-domain
- FAIL: head-only: out-of-domain object IoU < in-domain
- FAIL: hands-only: out-of-domain hand IoU < in-domain
- FAIL: full fine-tune out-of-domain hand IoU within 15 points of the paper's 71.3
- FAIL: full fine-tune out-of-domain object IoU within 15 points of the paper's 40.8
