# Project phases

Full project context: `skild_project.md`. We build one phase at a time, verify it, and only then move on.
Each phase lists **what we build**, **what it produces**, and **how we verify it** (the gate to the next phase).

## Decisions so far

| Topic | Decision |
|---|---|
| Local env | Plain venv at `.venv` from Homebrew `python3.12` (3.12.13), `modal` installed |
| Repo | `inHand/` is the repo root |
| Compute | Modal, workspace `whitelovemohan123`, GPU `L40S` |
| Classes | 0 background, 1 hand, 2 interacting object; 2nd-order objects -> 255 (ignore) |
| Cascade stage 2 | Predicts background vs interacting object; final mask = stage-1 hands + stage-2 objects |
| Clutter masks | Original SAM ViT-H (`segment-anything`) automatic mask generator |
| Logging | Metrics JSON + checkpoints on the Modal Volume (no wandb) |
| Loss | Plain cross-entropy (`ignore_index=255`) first; add Dice only if object IoU is weak |

## Current state (for a new chat)

- **Done:** Phases 0, 1, 2, 3, 4, 5 and 6 (all verified).
  - Phase 2 added `src/data.py`, `src/preresize.py`, `src/sanity.py` and the `preresize` / `dev_subset` / `sanity` entrypoints in `modal_data.py`.
  - Phase 3 added `src/config.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, `configs/{smoke_dev500,overfit8}.yaml`, `scripts/check_phase3_local.py`, and the `train` / `evaluate` entrypoints in `modal_app.py`. Runs `overfit8` and `smoke_dev500` are on the Volume and in `results/runs/`.
  - Phase 4 added `src/labels.py` (per-run label maps), config fields `data.label_map` and `eval.val_every`, the `train_many` / `fetch` entrypoints, `configs/b0_*.yaml`, `scripts/phase4_table.py`, `scripts/check_phase4_local.py`. Runs `b0_head_only`, `b0_full_ft(_s1)`, `b0_hands_only(_s1)` are on the Volume (`model_final.pt`) and in `results/runs/`; table in `results/phase4_table.md`.
- **Phase 5 complete (verified by the user):** implementation, preflight checks, both Modal training runs, numeric result review and final prediction-panel review are complete.
  - Stage 1 = `/data/runs/b0_hands_only/model_final.pt` (2 classes, `label_map {2: 0}` in its saved config).
  - Single-stage baseline = `b0_full_ft` (+ `_s1`).
  - Seed noise from Phase 4 is ≤ 0.7 IoU points.
  - Stage 2: `configs/b0_cascade_stage2{,_s1}.yaml`, two seeds, binary head, hand target -> background. Both runs share the frozen seed-0 stage 1. Both finished 11,240 iterations on all 8,993 train images; outputs are on the Volume and in `results/runs/b0_cascade_stage2{,_s1}/`.
  - `src/cascade_data.py` wraps the existing RGB transform, derives the aligned hand mask, corrupts only that input and maps the clean target. `src/data.py` is unchanged.
  - `src/cascade.py` provides composed predicted/oracle inference and evaluation; `scripts/phase5_table.py` reuses the existing baseline JSONs.
  - Preflight passed: user reported the init check **5/5** and accepted the real-image corruption previews (21 panels).
  - Local development checks pass: Phase 3 **13/13**, Phase 4 **28/28**, Phase 5 **27/27**. Both real runs report **23/23** checks passed, with full split counts and exact stage-1 hand-score identity.
  - `results/phase5_table.md`: mean object IoU baseline / predicted / oracle = **44.4 / 46.6 / 48.5** (val), **46.2 / 46.6 / 49.3** (test_indomain), **58.1 / 59.7 / 61.8** (test_outdomain). Gains are +2.2 / +0.4 / +1.6 points; the in-domain gain is smaller than the baseline's observed seed spread and needs cautious interpretation.
- **Phase 6 complete (verified by the user, 2026-11-19):** isolated `modal_clutter.py` app, validated `configs/phase6/clutter_sam_v1.yaml`, CPU filtering/cache arithmetic, candidate-only pilot previews, streaming four-model inference, resumable shards, compact fetch and a local report/plot script. Local Phase-6 checks pass **59/59**; Phase 3 **13/13**, Phase 4 **28/28** and Phase 5 **27/27** also pass. See Phase 6 below and `docs/phase6.md`.
  - **Preparation and the real ten-image pilot completed by the user:** original SAM ViT-H works on L40S with the pinned stack. The pilot produced 996 proposals and 670 retained regions; median/max SAM generation 2.60/7.03 s per image, peak allocated/reserved GPU memory 8.45/26.41 GB. Source and preview checksums match the prepared manifest. Details: `results/phase6/clutter_sam_v1/pilot_review.md`.
  - **Full run finished and fetched:** user persisted pilot approval and completed all four L40S shards (407/407/406/406 images, about 38.4 minutes per worker). Verified 1,626 unique images and 102,774 shared candidate regions; local aggregation reproduces the merged summary. Plot and report: `results/clutter_fp_vs_distance.png`, `.json`, `results/phase6_report.md`.
  - **Observed result:** the predicted cascade has higher mean candidate FP rates: baseline -> cascade **1.15% -> 1.26%** in-domain and **0.65% -> 0.75%** on YouTube. Same-seed comparisons have the same direction for both seeds. This does not support reduced clutter FP under the fixed SAM background-region proxy, despite Phase 5's higher object IoU. The user explicitly confirmed review and acceptance of the final curves/counts in this chat on 2026-11-19.
- **Phase 7 quantitative analysis, CPU survey and four actual previews verified by the assistant; user preview acceptance/full gallery pending (2026-11-19):** the user completed survey/fetch/build and the four-case preview/fetch. All 1,626 label records, source metrics and survey checksums reconcile. The assistant selected 16 RGB/GT-reviewed cases across five sources, including four SAM disagreements, then checked all eight actual preview panels. All 16 model/image records exactly reproduce the saved IoUs and FP/FN arithmetic; both selected SAM cases reproduce Phase-6 fractions/metadata, and all 13 preview file hashes validate. Source code/provenance remain frozen. **Next:** the user personally inspects `gallery/preview.html`, accepts the four previews, runs `approve-preview`, then launches full rendering (four cases can be reused; 12 remain). No user preview acceptance or full-gallery completion has been recorded. Reviews: `results/phase7/phase7_v1/{verification,preview_review}.md`; report: `results/phase7_report.md`.
- **Phase-8 handoff requested (2026-11-19):** the user asked to proceed to the write-up and prepare its prompt. `phase8_prompt.md` carries the verified numbers, available four-case visual evidence, documentation/resume deliverables and publication constraints. It explicitly records the absent full-gallery/approval artifacts and permits useful write-up drafting without inventing their completion. The README and other Phase-8 deliverables have not yet been created in this chat; the 16-case gallery scope has not been explicitly reduced or marked complete.
- **Findings to carry forward:** test_outdomain scores *higher* than test_indomain for every model (Ego4D dominates in-domain and is the hardest source); see Phase 4 Result.
- **Working rules:**
  - Environment/install commands are given to the user to run.
  - The user verifies each phase themselves before moving on.
  - The AI may run other commands (e.g. `modal volume ls`).
- **Local:** `.venv` (Python 3.12.13), `requirements.txt` installed, git initialised (nothing committed yet).
- **Modal workspace:** `whitelovemohan123`. Volume **`egohos`**, mounted at `/data` in containers:
  - `/data/raw/egohos_dataset.zip`: the original 4.5 GB zip (backup)
  - `/data/egohos/{train,val,test_indomain,test_outdomain}/{image,label}`: `.jpg` images, `.png` labels
  - `/data/egohos_512/train/{image,label}`: pre-resized training copy (shorter side 512, never upscaled, labels keep original 0–8 ids); written by Phase 2
  - `/data/sanity/`: Phase 2 overlay PNGs
  - `/data/runs/<run_name>/`: Phase 3+ training runs (config, logs, `model_final.pt`, `eval/`); see Phase 3, item 7
- **`modal_data.py`:** CPU-only app `egohos-data` with its own lighter image (no torch) for data jobs. Entrypoints: `download`, `stats`, `preresize`, `dev_subset`, `sanity`.
- **`src/data.py`** is shared by both apps and must stay torch-free. Training reads `EgoHOSDataset("train", "/data/egohos_512", TrainTransform(), file_list=...)`; evaluation reads the original splits with `EgoHOSDataset(split, "/data/egohos", EvalTransform())` (variable sizes, so batch size 1 or grouping by size).
- **Dev subset:** `configs/dev_subset_500.txt` (file stems) lives in the repo. The GPU image adds `configs/` at `/root/configs`; relative paths in configs resolve against `/root`.
- **Local multiprocessing note:** on macOS, `ProcessPoolExecutor` code must run from a script file with an `if __name__ == "__main__":` guard (spawn start method); Modal's Linux containers use fork.
- **`modal_app.py`:** GPU app `egohos-segformer` with the torch image. Entrypoints: `check`, `train --config <yaml> [--overwrite|--resume]`, `train_many --configs <a.yaml,b.yaml,...>` (parallel, one container each; launch with `modal run --detach`), `evaluate --run-name <name> --split <split>`, `fetch --run-names <a,b,...>` (copies a run's small files from the Volume to `results/runs/`). The GPU type comes from the `EGOHOS_GPU` env var (default `L40S`).
- **GPU:** payment method added; L40S works (Phase 0 check passed).
- **Local env:** `requirements.txt` includes torch/torchvision/transformers/albumentations, pinned to the same versions as the Modal image, for local debugging (CPU/MPS) and later ONNX work. Still prefer returning plain Python types from Modal functions.
- **Pinned GPU image:** torch 2.14.1, torchvision 0.29.2, transformers 5.22.2, albumentations 2.0.8, opencv-python-headless 4.14.0.94, numpy 2.3.5.

---

## Phase 0: Environment and repo skeleton

**Build**
- [x] Python 3.12 venv, `modal` installed, Modal login verified (`modal volume list` works)
- [x] `git init`, `.gitignore` (`.venv/`, `data/`, checkpoints, large outputs)
- [x] `requirements.txt` (local) and folder skeleton: `src/`, `configs/`, `scripts/`, `results/`
- [x] `modal_app.py` with the pinned GPU image (PyTorch, transformers, albumentations, opencv-headless, gdown) and the `egohos` Volume
- [x] Local requirements installed into `.venv`

**Verify**
- A trivial Modal GPU function runs and prints the torch version, CUDA availability and GPU name.
- [x] **Passed on L40S:**
  - torch 2.14.1+cu130, CUDA available, bf16 supported
  - SegFormer-B0 has 3.71M params; logits `[2, 3, 128, 128]` for a 512x512 input
  - Volume shows `['egohos', 'raw']`
  - Note: values returned from Modal functions must be plain Python types (e.g. `str(torch.__version__)`), because the local env has no torch.

---

## Phase 1: Dataset onto the Modal Volume

**Build**
- [x] `modal_data.py`: a separate CPU-only Modal app (lighter image, no GPU functions, so no payment method needed).
- [x] `download_dataset`: `gdown` -> `raw/egohos_dataset.zip` on the Volume (kept as a backup) -> unzip -> `egohos/` (fallback: browser download + `modal volume put egohos <zip> raw/egohos_dataset.zip`, then rerun).
- [x] `src/dataset_stats.py` + `dataset_stats`: counts, image/label matching, image sizes, label PNG modes and values, per-class pixel share, contact mask values.

**Run**
```bash
modal run modal_data.py::download
modal run modal_data.py::stats
```

**Produces**
- `/data/egohos/{train,val,test_indomain,test_outdomain}/{image,label,contact}` on the Volume.
- `results/dataset_stats.json`: file counts, image sizes, unique label values, per-class pixel share.

**Verify**
- [x] Counts match the paper: train 8,993 / val 1,124 / test_indomain 1,126 / test_outdomain 500.
- [x] Every image has a matching label file.
- [x] Label PNGs contain only values 0–8.

**Result (passed):** the dataset was downloaded in the browser (gdown hit the Drive quota) and uploaded with `modal volume put`.
- Labels are 8-bit grayscale PNGs (`L` mode) with the same size as their images.
- The zip has **no `contact/` folders**. This only affects stretch S4.
- Image sizes vary from 456x256 (EPIC) to 3840x2160. Out-of-domain images are all 1920x1080.
- Train pixel share after our remap: background 80.9%, hand 10.4%, interacting object 5.8%, ignore (2nd order) 2.9%.

---

## Phase 2: Data pipeline and sanity check

**Agreed choices**
1. **Label remap** (lookup table on the `L`-mode PNG): `0 -> 0` background, `1,2 -> 1` hand, `3,4,5 -> 2` interacting object, `6,7,8 -> 255` ignore.
2. **Pre-resized training copy** on the Volume at `/data/egohos_512/train/{image,label}`:
   - shorter side resized to **512 px**, aspect ratio kept
   - images already smaller (e.g. 456x256 EPIC frames) are **not upscaled**
   - images: bilinear/area interpolation, saved as JPEG (quality ~95)
   - labels: **nearest-neighbour**, saved as PNG with the **original 0–8 ids**, so the remap stays a single function used everywhere
   - val/test splits are not pre-resized
3. **Training augmentation** (albumentations, applied to the pre-resized copy):
   - random rescale **0.5–2.0x relative to the 512 px eval scale**: the shorter side is resized to a random value in 256–1024 px (step 8), whatever the stored size. This way small EPIC frames (stored at 256 px) are seen at the same scales as everything else and match evaluation (which upscales them to 512).
   - **random 512x512 crop**; if the image is smaller than 512, pad it (image with 0, label with **255 = ignore** so padding never counts in the loss)
   - horizontal flip (p=0.5), safe because left/right hands are merged
   - colour jitter (brightness/contrast/saturation 0.25, hue only 0.02 so skin keeps a plausible colour, p=0.8)
   - normalise with ImageNet mean/std
4. **Evaluation protocol** (used from Phase 3 on):
   - resize the input so its **shorter side is 512 px** and predict
   - **upsample the logits to the original image size** (bilinear), then argmax
   - score against the **original-resolution label**: full-resolution metrics without running the model on 4K inputs
5. **Dev subset:** 500 training images, fixed seed, **sampled in proportion to source** (`ego4d_*`, `epic_*`, `thu_*`, and the authors' own videos = any other prefix). Saved as a file list: `configs/dev_subset_500.txt`.
   - Actual train prefixes: `ego4d` 6,227, `epic` 1,685, `thu` 637, `escape` 444 (the only "other" prefix = authors' own videos, called `own` in code). Expected subset: 346 / 94 / 35 / 25.
6. **`src/data.py` is torch-free** (the `egohos-data` image has no torch). `EgoHOSDataset` is a plain map-style class (`__len__`/`__getitem__`, which `DataLoader` accepts) returning numpy arrays: float32 CHW image and int64 HW mask. `DataLoader`'s default collate turns them into tensors.
7. Small details: images with shorter side ≤ 512 are **copied byte-for-byte** into the pre-resized copy (no JPEG re-encode); labels are resized with `INTER_NEAREST_EXACT` (no half-pixel shift vs the image); images are read with EXIF orientation ignored so they always match their labels.

**Build**
- [x] `src/data.py`:
  - `REMAP_LUT` + `remap_label()`
  - `EgoHOSDataset(split, root, transform=None, file_list=None)` returning (image, remapped mask) as numpy; `.load(i)` gives the raw RGB image + original 0–8 label; `.names` maps index -> file stem
  - `TrainTransform(seed=None)` / `EvalTransform()`: callables `(image, mask) -> (image, mask)`; the eval transform resizes only the image and leaves the mask at original resolution
  - `source_of()`, `make_dev_subset()`, `read_file_list()`, `denormalize()`; path constants `DATA_ROOT`, `PRERESIZED_ROOT`
- [x] `src/preresize.py`: `preresize_split()` (multiprocessing, skips pairs already written), then re-reads every output file and checks counts, sizes, label values
- [x] `src/sanity.py`: `build_sanity()` renders the panels and returns PNG bytes + a summary with checks
- [x] `modal_data.py` entrypoints:
  - `preresize`: writes `/data/egohos_512/train`, saves `results/preresize_train.json`
  - `sanity`: writes overlays to `/data/sanity/` on the Volume and to local `results/sanity/` (+ `summary.json`)
  - `dev_subset`: writes `configs/dev_subset_500.txt`, prints source proportions
- [x] Overlay colours: hand = blue, interacting object = yellow (tint + contour), ignore = grey hatched, background = no tint. The original 0–8 label panel uses one colour per id with a legend.

**Run** (in this order; `sanity` needs the pre-resized copy)
```bash
modal run modal_data.py::preresize
modal run modal_data.py::dev_subset
modal run modal_data.py::sanity
```

**Produces**
- `/data/egohos_512/train/` on the Volume (8,993 images + labels).
- `configs/dev_subset_500.txt`.
- `results/preresize_train.json`.
- `results/sanity/`:
  - `label_00..19_*.png`: 20 random training samples (at least 2 per source, original files), each a panel of image | original 0–8 label (colour per id) | remapped 3-class overlay
  - `aug_0..4_*.png`: 5 pre-resized samples (1 EPIC, 1 THU, 1 own, 2 Ego4D), each stored image + 3 training-augmentation draws
  - `eval_0..2_*.png`: the smallest and largest val images + 1 random, eval input next to the original-resolution mask
  - `summary.json`

**Verify**
- [x] **You look at the overlays:** hands, 1st-order objects and ignored 2nd-order objects are coloured correctly, and image/mask stay aligned after augmentation.
- [x] Remapped masks contain only {0, 1, 2, 255}.
- [x] The pre-resized copy has 8,993 image/label pairs, shorter side ≤ 512, and its labels still contain only 0–8.
- [x] Dev subset: 500 unique names, all in train, source proportions printed next to the full train set's.

**Result (passed):** verified by the user; overlays in `results/sanity/`, pre-resize report in `results/preresize_train.json`, dev subset in `configs/dev_subset_500.txt`.

---

## Phase 3: Model, training loop, evaluation (smoke test)

**Agreed choices**
1. **Optimizer:** AdamW. The encoder (`segformer.*`) uses lr 6e-5; the whole `decode_head.*` (new, randomly initialised) uses 10x. Weight decay is 0.01, except on 1-D params (norms and biases), as in the mmseg SegFormer recipe.
2. **Schedule:** iteration-based (infinite seeded sampler, no epochs). Linear warmup, then polynomial decay (power 1.0) to 0.
3. **Loss:** computed in `train.py`, not by Hugging Face. Logits (1/4 resolution) are upsampled bilinearly to the crop size in fp32, then cross-entropy with `ignore_index=255`. bf16 autocast with no GradScaler.
4. **Worker seeding fix:** albumentations 2.0.8 creates its generator once in the main process, so every DataLoader worker would draw identical augmentations. `train.seed_worker` reseeds `dataset.transform.aug` from torch's per-worker seed. `src/data.py` is unchanged.
5. **Evaluation:** the Phase 2 protocol at batch size 1, with fp32 logits upsampled to the original size. The confusion matrix is accumulated over the dataset, ignoring 255. Reported: per-class IoU, **mIoU over 3 classes**, **`miou_fg` over hand + object**, pixel accuracy, per-image IoU (for later failure picking).
6. **Boundary IoU** (Cheng et al. 2021, official definition):
   - The band is the mask minus its erosion. d = **2% of the image diagonal** at original resolution: 10 px for 456x256, 36 px for 1440x1080, 88 px for 4K.
   - The image border counts as outside the mask, matching the official `boundary-iou-api`. It's computed on the GPU with a separable max-pool, and checked against the cv2 implementation locally.
   - Hand and object only. Ignore pixels are removed from the predicted band. Intersections and unions are summed over the dataset.
7. **Run layout** on the Volume, `/data/runs/<run_name>/` (`run_name` = config file stem):
   - `config.yaml`, `train_names.txt`, `train_log.jsonl` (loss, both lrs, img/s, data-wait fraction, GPU memory per log step), `loss_curve.png`, `summary.json`
   - `checkpoint_last.pt`: model, optimizer, scheduler and step every `ckpt_every`; deleted when the run finishes
   - `model_final.pt`: weights + HF config + run config; rebuildable without the hub
   - `eval/<split>.json`, `eval/<split>_vis/*.png`

   Small files are copied to local `results/runs/<run_name>/`. An existing run name stops with an error unless `--overwrite` (start fresh) or `--resume` (continue from `checkpoint_last.pt`) is passed.
8. **Configs:** YAML in `configs/`, merged over `DEFAULTS` in `src/config.py` (unknown keys are rejected). Sections:
   - `data`: `train_root`, `train_file_list`, `subset`, `augment`, `num_workers`
   - `model`: `pretrained`, `class_names` (sets the number of classes), `freeze_encoder`
   - `train`: `max_iters`, `batch_size`, `warmup_iters`, `lr`, `head_lr_mult`, `weight_decay`, `poly_power`, `bf16`, `log_every`, `ckpt_every`
   - `eval`: `splits`, `boundary_dilation_ratio`, `n_vis`, `min_fg_iou`
9. **Smoke test** (`configs/smoke_dev500.yaml`): the 500-image dev subset, normal augmentation, batch size 16, 1,000 iterations (about 32 passes), 100 warmup, then full val.
10. **Overfit test** (`configs/overfit8.yaml`):
    - Images: 8 from the dev subset, 2 per source, each with at least 1% hand and 1% object pixels; seed 0, saved in `train_names.txt`.
    - **No augmentation:** `FixedTransform` resizes to a 512 px shorter side like eval, then pads to 512x912 (image 0, mask 255).
    - Batch size 8, 500 iterations, 25 warmup.
    - Scored on the same 8 images at original resolution (split `train` = the run's training images). PASS if hand and object IoU are both ≥ 0.90.
11. **Visualisations:** each panel is image | ground truth | prediction | errors (wrong pixels red, ignore hatched), with per-image IoU in the title. For val, the 6 fixed picks are the smallest and largest image plus one per source, the same for every run. The overfit run shows all 8 images.
12. **Modal:** `train_run` / `eval_run` (GPU, `cpu=8`, 32 GB). The image adds `configs/` at `/root/configs`. Entrypoints: `train --config <yaml> [--overwrite|--resume]` (trains, then evaluates `eval.splits` in the same container) and `evaluate --run-name <name> --split <split>`.

**Build**
- [x] `src/model.py`: `build_model(class_names, pretrained)`, `freeze_encoder` / `set_train_mode` (a frozen encoder stays in eval mode), `param_groups`, `save_model` / `load_model`.
- [x] `src/train.py`: training loop, `FixedTransform`, `InfiniteSampler`, `seed_worker`, subset picker, run orchestration (train -> eval -> files).
- [x] `src/evaluate.py`: `evaluate` (confusion matrix + Boundary IoU at full resolution), `mask_boundary`, prediction panels, `evaluate_split`, `evaluate_run`.
- [x] `src/config.py`, `configs/smoke_dev500.yaml`, `configs/overfit8.yaml`, `modal_app.py` functions + entrypoints.
- [x] `scripts/check_phase3_local.py`: local CPU check on synthetic data. It covers worker seeding (reproduces the bug, then the fix), `FixedTransform` size, Boundary IoU against the official cv2 code, IoU arithmetic, and a tiny end-to-end run: interrupt, refused rerun, resume, eval, and reloading `model_final.pt` reproduces the metrics.

**Run**
```bash
modal run modal_app.py::train --config configs/overfit8.yaml
modal run modal_app.py::train --config configs/smoke_dev500.yaml
```

**Produces**
- `/data/runs/{overfit8,smoke_dev500}/` on the Volume.
- `results/runs/{overfit8,smoke_dev500}/`: `loss_curve.png`, `train_log.jsonl`, `summary.json`, `config.yaml`, `train_names.txt`, `eval/<split>.json`, `eval/<split>_vis/*.png`.

**Verify** (short run on the 500-image dev subset, about 2–3 min on GPU)
- [x] Loss goes down (`smoke_dev500` loss curve; `loss_decreased_by_half` PASS).
- [x] The model can overfit a handful of images (`overfit8`: hand and object IoU ≥ 0.90 on its 8 training images).
- [x] Evaluation runs on the full val set (1,124 images) and writes `eval/val.json`.
- [x] Predicted masks visualised for a few val images (`results/runs/smoke_dev500/eval/val_vis/`).

**Result (passed, verified by the user), on L40S:**
- **`overfit8`:** 500 iterations in 30 s, loss 0.964 -> 0.017. On its 8 training images at original resolution:
  - IoU: background 0.993, hand **0.936**, object **0.957**
  - Boundary IoU: hand 0.884, object 0.900
- **`smoke_dev500`:** 1,000 iterations in 67 s, loss 1.048 -> 0.183. Full val (1,124 images) took 21 s:
  - IoU: background 0.910, hand 0.692, object 0.248
  - mIoU 0.616, `miou_fg` 0.470
  - Boundary IoU: hand 0.539, object 0.186
  - These numbers come from only 500 images; they're not a target.
- **Throughput:** about **282 img/s** at batch size 16 (about 0.057 s per iteration). Data wait is about 0.5% of step time and peak GPU memory 4.75 GB, so 8 CPUs are enough and the GPU is the bottleneck.
  - The first log window is slower (worker start-up).
  - Each checkpoint save + Volume commit costs about 4 s: the step-520 window dropped to 83 img/s.
- **Implication for Phase 4:** a full run of about 11k iterations (about 20 epochs at batch size 16) should take about **11 min of training** plus about 1 min of evaluation, much less than the 20–40 min estimated earlier.

---

## Phase 4: Single-stage experiments (full data, run in parallel)

**Runs** (5 L40S containers at once, about 13–17 min each)
- (a) Head-only (frozen encoder), 3 classes: `b0_head_only`.
- (b) Full fine-tune, 3 classes, the main single-stage model: `b0_full_ft` + `b0_full_ft_s1`.
- (c) Hands-only model (background vs hand), stage 1 of the cascade: `b0_hands_only` + `b0_hands_only_s1`.

**Agreed choices**
1. **Schedule:** 11,240 iterations (20 epochs at batch 16 on 8,993 images), 1,500 warmup, poly decay (power 1.0) to 0, the same for every run. Learning rates as in Phase 3 (encoder 6e-5, head 6e-4); head-only keeps head lr 6e-4, so the encoder is the only difference between (a) and (b).
2. **Hands-only label map:** new config field `data.label_map` (remapped id -> training id), `{2: 0}` for hands-only.
   - 2nd-order objects stay 255. Hand IoU of (c) and (b) is then computed on exactly the same pixels: merging object into background does not change the hand row or column of the confusion matrix.
   - `src/labels.py` (torch-free): `label_lut()` validates the map against `class_names` (also at config load); `LabelMapTransform` / `with_label_map()` apply it to the remapped mask **before** the transform (after it, 255 also marks crop padding). It exposes `.aug`, so `seed_worker` still works.
   - The same map is used in training, evaluation, the val curve and the panels. It is saved in `config.yaml` and in `model_final.pt` (`ckpt["config"]["data"]["label_map"]`), so `evaluate_run` and Phase 5 reproduce it. `src/data.py` is unchanged.
3. **Seeds:** 2 seeds for (b) and (c), 1 for (a). Seed-1 configs differ only in `seed`. The new decode head's initialisation is now seeded (`torch.manual_seed(cfg.seed)` before `build_model`).
4. **Checkpoints:** `ckpt_every: 2000`.
5. **Evaluation:** `eval.splits: [val, test_indomain, test_outdomain]`, right after training.
6. **Visualisations:** 8 per split, always picked with seed 0 (`VIS_SEED`), the same for every run.
   - val / test_indomain: smallest and largest image, one per source, 2 random.
   - test_outdomain: one frame from each of 8 different videos (names are `youtube_<video title>_<frame>`, 27 video stems; `source_of` says `own`, titles show `youtube`).
   - PNG names are sanitised (`vis_file_stem`: no spaces or brackets, ≤ 80 characters, frame number kept).
7. **Val curve:** new config field `eval.val_every: 2000`: full val -> `val_log.jsonl` + `val_curve.png`; the last point is the final val eval. A curve only: the reported model is always the final one.
8. **Launch:** `modal run --detach modal_app.py::train_many --configs a.yaml,b.yaml,...` spawns one container per config; each log line is prefixed with `[run_name]`. `fetch --run-names ...` (CPU-only `fetch_run`) copies everything except `*.pt` from the Volume into `results/runs/`.
9. **Table:** `scripts/phase4_table.py` (local, no torch) reads `results/runs/<run>/{config.yaml,summary.json,eval/*.json}` and writes `results/phase4_table.md`: runs, per-split results, seed noise, the stage-1 question, domain gap, the EgoHOS reference, consistency checks. Runs named `<group>_s<k>` are seeds of `<group>` and get mean rows.
10. **EgoHOS reference:** the paper evaluates on the YouTube test set (= our test_outdomain), best checkpoint on val. Quoted: Table 5 "Para. Decode" (left hand 69.08, right hand 73.50, left/right/two-hand object 48.67 / 36.21 / 37.46) and Table 3 binary hand "mIoU" 83.18 (ResNet-18 + HRNet; unclear whether background is averaged in).

**Build**
- [x] `src/labels.py`; `src/config.py`: `data.label_map`, `eval.val_every`, map validated in `load_config`.
- [x] `src/train.py`: label map in training and `select_subset`, seeded head init, val curve (`val_point`, `plot_val_curve`, resume-safe `val_log.jsonl`), `[run_name]` log prefix.
- [x] `src/evaluate.py`: label map in evaluation and panels, seed-independent picks (by video on test_outdomain), `vis_file_stem`, `display_source`, `label_map` in the eval JSON. `src/sanity.py`: `_class_legend(num_classes)`.
- [x] `modal_app.py`: `train_many`, `fetch` + `fetch_run`, shared report printing.
- [x] `configs/b0_{head_only,full_ft,full_ft_s1,hands_only,hands_only_s1}.yaml`.
- [x] `scripts/phase4_table.py`, `scripts/check_phase4_local.py` (28 checks on synthetic data); `scripts/check_phase3_local.py` still passes 13/13.

**Run**
```bash
.venv/bin/python scripts/check_phase4_local.py
modal run --detach modal_app.py::train_many --configs configs/b0_head_only.yaml,configs/b0_full_ft.yaml,configs/b0_full_ft_s1.yaml,configs/b0_hands_only.yaml,configs/b0_hands_only_s1.yaml
modal run modal_app.py::fetch --run-names b0_head_only,b0_full_ft,b0_full_ft_s1,b0_hands_only,b0_hands_only_s1   # only after a disconnect
.venv/bin/python scripts/phase4_table.py
```

**Produces**
- `/data/runs/b0_*/` on the Volume (with `model_final.pt`).
- `results/runs/b0_*/`: `config.yaml`, `train_log.jsonl`, `loss_curve.png`, `val_log.jsonl`, `val_curve.png`, `summary.json`, `eval/{val,test_indomain,test_outdomain}.json` (with `per_image`), `eval/<split>_vis/*.png`.
- `results/phase4_table.md`.

**Verify**
- [x] Full fine-tune beats head-only, on every split and metric, by about 20 points of mIoU_fg.
- [x] The numbers are sensible next to the EgoHOS paper: ours are higher (see Result), which is consistent with our merged left/right classes. Not directly comparable.
- [x] Prediction overlays look right (verified by the user).

**Result (passed, verified by the user), on L40S.** IoU x 100, means over seeds where there are 2:

| run | val hand / object / mIoU_fg | test_indomain hand / object | test_outdomain hand / object | BIoU hand / object (val) |
|---|---|---|---|---|
| (a) `b0_head_only` | 66.1 / 19.0 / 42.6 | 67.0 / 17.3 | 71.1 / 22.9 | 52.5 / 16.4 |
| (b) `b0_full_ft` (2 seeds) | 82.2 / 44.4 / 63.3 | 82.4 / 46.2 | 88.3 / 58.1 | 68.9 / 33.7 |
| (c) `b0_hands_only` (2 seeds) | 84.7 / - / - | 84.6 / - | 89.7 / - | 72.1 / - |

- **Seed noise is small:** the largest seed-0 vs seed-1 spread on any metric is 0.7 points (object IoU); hand IoU differs by ≤ 0.2.
- **Stage-1 question:** the dedicated hands-only model beats the 3-class model's hand by +2.4 (val), +2.2 (test_indomain), +1.3 (test_outdomain) IoU and +2.3 to +3.5 Boundary IoU. That is well above the seed spread, so (c) is the better stage 1.
- **No in-domain -> out-of-domain drop:** every model scores *higher* on test_outdomain (full fine-tune: hand +6.0, object +12.0).
  - It is not a pixel-weighting artefact. Per-image medians for `b0_full_ft` are also higher on YouTube (hand 0.91, object 0.67) than on test_indomain.
  - test_indomain is 87% Ego4D by pixel, and Ego4D is the hardest source: median per-image object IoU 0.41, against 0.44 EPIC, 0.68 THU, 0.74 own.
  - So the YouTube set is easier for these models, not harder. Phase 7 must report this as a finding (per source), not as a "domain gap".
  - The table's automatic checks "out-of-domain < in-domain" and "within 15 points of the paper" FAIL for this reason. They encoded expectations, not correctness.
- **Paper comparison (test_outdomain):** ours are above the paper's Table 5 single-network numbers: hand 88.3 vs 69.1 / 73.5 (left / right), object 58.1 vs 36.2–48.7. That fits our easier label granularity: merged left/right means no left/right confusion, and the paper's numbers come from its own models and protocol.
- **Val curve:** full fine-tune flattens at the end (val mIoU_fg 62.3 at 8k, 63.5 at 10k, 63.5 at 11,240); head-only is still slowly rising (41.6 -> 42.6). 20 epochs is enough for (b).
- **Cost and speed:**
  - Full fine-tune 285 img/s, head-only 401 img/s, peak GPU memory 4.75 GB.
  - Training 13–16 min per run, including about 65 s for the val curve; evaluation 9–29 s per split.
  - Mean data wait 3.6–4.8%. It is much higher (up to about 90%) in the first windows while the full train set is read from the Volume for the first time.

---

## Phase 5: Two-stage cascade

**Agreed choices (2026-11-19)**
1. **Stage 2:** binary head `[background, interacting_object]`; clean target map **`{1: 0, 2: 1}`**, so hand pixels become background and 2nd-order pixels stay 255. The hand input is derived before this map. Corruption never changes the loss target.
2. **Initialisation:** ImageNet `nvidia/mit-b0`, fresh 2-class decode head. Expand only the loaded first patch projection, preserve its RGB weights and bias, zero the new channel, set HF `config.num_channels=4`. No mismatched-size loading and no warm-start from the fine-tuned baseline.
   - Installed Transformers 5.22.2 uses `segformer.stages[0].patch_embeddings.proj`, not the older `segformer.encoder.patch_embeddings[0].proj` path. Loaded shape confirmed: Conv2d 3 -> 32, kernel 7, stride 4, padding 3. Surgery reads these properties from the loaded layer.
   - The init check compares a model and its clone (identical decoder and all other weights), including arbitrary nonbinary hand values, in eval mode/fp32.
3. **Input:** RGB follows the unchanged Phase-2 recipe; the appended channel is float32 0/1, never ImageNet-normalised. Training derives it from the clean mask after the existing geometric transform (equivalent to applying the same nearest-neighbour geometry to a separate binary hand mask). RGB colour jitter and normalisation never touch it. Worker seeds independently reseed augmentation and corruption.
4. **Corruption at the final 512 crop:**
   - Drop with p=0.10: remove one connected component chosen uniformly from the two largest (8-connectivity), retaining the other if present.
   - Then none / dilation / erosion with p=0.40 / 0.30 / 0.30; elliptical kernel uniformly from odd sizes 3–15 px, zero border.
   - Independently with p=0.20, add 1–3 filled circles (uniform count), radius 3–10 px (uniform), centres anywhere in the crop. No object-label-dependent placement.
   - About 29% of draws have no corruption events; additional draws may be unchanged if an operation has no effect. The real-image previews also include explicitly labelled forced-operation examples; these do not represent training frequencies.
5. **Schedule:** 11,240 iterations, batch 16, warmup 1,500, poly power 1, encoder/head lr 6e-5 / 6e-4, weight decay 0.01, bf16, plain cross-entropy ignoring 255, checkpoints every 2,000. Same full training set and RGB recipe as Phase 4.
6. **Two stage-2 seeds:** `b0_cascade_stage2` and `b0_cascade_stage2_s1`. Both use `/data/runs/b0_hands_only/model_final.pt`, frozen and in eval mode. Do not retrain stage 1 or either full-fine-tune baseline. Observed two-seed spreads are descriptive, not a bound on future noise.
7. **Evaluation:** full val / test_indomain / test_outdomain, same Phase-2 confusion-matrix and Boundary-IoU definitions on the original 3-class labels (255 excluded).
   - Predicted input = stage-1 logits upsampled to RGB-input size, then argmax hand. Pasted hands = the same logits independently upsampled to original size, then argmax. Never resize the input-scale hard mask for composition.
   - Oracle input = `(original GT == 1)` resized nearest-exact to input size. Oracle also pastes GT hands at original resolution, so oracle hand IoU is 100%.
   - Stage-2 logits are upsampled to original size before argmax. Paint objects first, hands last. Predicted hand IoU and Boundary IoU must equal the saved **seed-0** hands-only scores exactly (not the two-seed mean).
   - Oracle object IoU >= predicted is an expected diagnostic, **not a mandatory correctness check**; a fixed model can respond unfavourably to cleaner inputs. Report and investigate reversals honestly.
   - This compares complete systems. A separate RGB-only binary ablation would be needed to isolate hand information from task/head and compute differences; it is outside Phase 5.
8. **Val curves:** both composed modes on full val every 2,000 iterations and at the final step; nested records in `val_log.jsonl`, object IoU and Boundary IoU in `val_curve.png`. No checkpoint selection.
9. **Visualisations:** same 8 `VIS_SEED=0` picks per split as Phase 4, with image / GT / actual hand input / composed prediction / errors. Input channel is display-resized only. Predictions are recomputed on demand; no prediction cache.
10. **Launch:** detached two-run training, then fetch. Training/eval write only the stage-2 run directories. CPU previews are separately stored under `/data/cascade_previews/<run_name>/` and `results/cascade_previews/<run_name>/`.

**Build**
- [x] `src/model.py`: first-projection discovery and 3 -> 4 channel surgery; save/load reconstructs four channels offline.
- [x] `src/cascade_data.py`: aligned clean/corrupted inputs and worker-safe corruption; no edits to `src/data.py`.
- [x] `src/cascade.py`: stage-1 -> stage-2 inference, both composed modes, original-resolution metrics and panels, reference hand-score checks. Existing RGB `evaluate()` is unchanged.
- [x] Training integration: two-mode val curves, four-channel resume, final paired eval; defaults keep old configs and checkpoints working.
- [x] Two configs, CPU corruption-preview entrypoint, `scripts/check_phase5_local.py`, `scripts/phase5_table.py`.

**Local development checks:** Phase 5 **27/27**, Phase 3 **13/13**, Phase 4 **28/28**, all on synthetic CPU data. Init logit maximum absolute error was 0 for zeros, ones and arbitrary nonbinary masks. The preview renderer was also exercised on synthetic images. The Phase-4 check had a pre-existing assertion comparing a returned Python integer label-map key with a JSON string key; it now compares with the saved integer-key map. Reloaded metrics matched exactly; evaluation scoring was unchanged. Both real runs are complete; the user has accepted the preflight checks and final prediction panels.

**Run (user runs these; inspect preflight before launching training)**
```bash
# Synthetic CPU check, including init equivalence and end-to-end resume/reload.
.venv/bin/python scripts/check_phase5_local.py
# Optional shorter init-only check:
.venv/bin/python scripts/check_phase5_local.py --init-only

# Real training-image previews, CPU only; uses the agreed seed-0 recipe for both runs.
.venv/bin/modal run modal_app.py::cascade_preview --config configs/b0_cascade_stage2.yaml
```
Inspect `results/cascade_previews/b0_cascade_stage2/`: random crops plus labelled forced stress examples. Hands should align with RGB; dilation/erosion should be modest; drop should remove one component; green/red differences should affect the input, while the binary loss target stays clean (yellow objects, no hand class, grey ignore). Adjust the recipe before training if these look unrealistic.

After the user accepts these previews:
```bash
.venv/bin/modal run --detach modal_app.py::train_many --configs configs/b0_cascade_stage2.yaml,configs/b0_cascade_stage2_s1.yaml

# After both runs finish (also recovers outputs after a client disconnect):
.venv/bin/modal run modal_app.py::fetch --run-names b0_cascade_stage2,b0_cascade_stage2_s1
.venv/bin/python scripts/phase5_table.py

# Recompute a stage-2 split if needed; evaluates both modes from the saved final weights:
.venv/bin/modal run modal_app.py::evaluate --run-name b0_cascade_stage2 --split val
# Resume only an interrupted training run with checkpoint_last.pt:
.venv/bin/modal run --detach modal_app.py::train --config configs/b0_cascade_stage2.yaml --resume
```

**Produces**
- `/data/runs/b0_cascade_stage2{,_s1}/`: run config, names, training logs, curves, `summary.json`, `model_final.pt`; small files fetched to `results/runs/`.
- `eval/<split>_{predicted,oracle}.json` with original-resolution 3-class metrics, Boundary IoU, per-image IoU and provenance; matching `<split>_{predicted,oracle}_vis/` panels.
- `results/phase5_table.md`: three object-IoU numbers per split, all per-class metrics, per-seed rows and means, observed seed spreads, hand-identity checks and oracle diagnostics. Reads the existing `b0_full_ft{,_s1}` and `b0_hands_only` JSONs; never re-evaluates those models.

**Verify (passed, verified by the user)**
- [x] Init check passes for zeros, ones and arbitrary nonbinary hand inputs with identical decoder weights (user reported 5/5 passed).
- [x] User inspects real training corruption overlays and accepts the recipe before training (21 panels; user confirmed it is working fine).
- [x] Both final runs finish; all three splits have 1,124 / 1,126 / 500 images and both composed modes. Matching `per_image` counts and 8 panels per split/mode; all visualisation picks match Phase 4.
- [x] Reported predicted hand IoU and Boundary IoU exactly match `b0_hands_only` seed 0; oracle hand IoU and Boundary IoU are 100%. All 23 run checks pass for each seed.
- [x] Results table built and numeric comparison recorded below. Oracle object IoU exceeds predicted on every split for each seed; no reversals.
- [x] User inspects final prediction panels and accepts the Phase-5 results before Phase 6 (user confirmed they look fine).

**Result (passed, verified by the user), on L40S.** IoU x 100; means over two baseline seeds and two stage-2 seeds. The cascade shares the frozen seed-0 hand model.

| split | single-stage object | predicted cascade object | oracle cascade object | predicted - baseline | oracle - predicted |
|---|---|---|---|---|---|
| val | 44.4 | 46.6 | 48.5 | +2.2 | +1.9 |
| test_indomain | 46.2 | 46.6 | 49.3 | +0.4 | +2.6 |
| test_outdomain (YouTube) | 58.1 | 59.7 | 61.8 | +1.6 | +2.1 |

Deltas are computed before rounding. Full per-class and per-seed results are in `results/phase5_table.md`.

- **Finding:** the complete cascade improves mean object IoU on all three splits, with clearer gains on val and YouTube than in-domain. The +0.4-point in-domain gain is smaller than the baseline's observed 0.7-point seed spread, so two seeds do not establish a robust in-domain improvement. Val (+2.2) and YouTube (+1.6) gains exceed the observed within-system spreads, without establishing statistical significance from only two seeds.
- **Seed consistency:** predicted object IoU is 46.6 / 46.6 on val, 46.7 / 46.6 in-domain, 59.6 / 59.9 on YouTube (seed 0 / seed 1). Largest predicted-cascade object spread is 0.3 points.
- **Object boundaries also improve:** single-stage -> predicted Boundary IoU is 33.7 -> 35.1 (val), 34.0 -> 34.9 (in-domain), 44.3 -> 45.5 (YouTube). Oracle object Boundary IoU is 38.2 / 38.3 / 48.8.
- **Hands are inherited from stage 1:** predicted hand IoU is 84.6 / 84.5 / 89.7 (val / in-domain / YouTube), matching the saved seed-0 scores exactly, including Boundary IoU. This is the expected composition invariant, not a new hand-learning result from stage 2.
- **Oracle:** object IoU is higher for both seeds on every split, with a mean advantage of 1.9 / 2.6 / 2.1 points. The oracle changes both the hand input and pasted hands; this gap does not isolate either contribution. Its perfect hand scores are by construction.
- **Per-split interpretation:** YouTube remains easier than test_indomain for this cascade, consistent with Phase 4. Compare systems within each split; no domain-gap-closing claim follows from these numbers. No RGB-only binary ablation was run, so the gain cannot be attributed solely to the extra hand channel.
- **Val curves:** seed-0 / seed-1 predicted object IoU is 45.8 / 45.1 at 8k, 46.0 / 45.9 at 10k, 46.6 / 46.6 at 11,240. The final interval still adds about 0.6–0.7 points. The agreed fixed budget was retained for comparison with Phase 4; no checkpoint selection or extra training.
- **Speed and outputs:**
  - Training including validation curves: 725.9 s (12.1 min, seed 0) and 907.9 s (15.1 min, seed 1); validation curves account for 163.1 / 240.2 s.
  - Logged throughput 326.6 / 274.8 img/s, mean data wait 0.4% each, peak GPU memory 4.77 GB. These are observed run speeds, not a controlled architecture speed comparison.
  - Final evaluation over all three splits took 81.3 / 116.1 s total for both modes together. Each mode's `eval_seconds` repeats that split's shared two-mode time; do not double-count it.
  - Both seeds trained on 8,993 images for 11,240 iterations. All six eval JSONs per seed contain full per-image results, with 8 matching panels per split/mode (48 per seed).

**Status:** Phase 5 complete and verified by the user. Phase-6 implementation is described below; real SAM work and user verification remain pending.

---

## Phase 6: Clutter analysis with SAM

**Status:** complete, verified by the user on 2026-11-19. Implementation, local synthetic verification, user-run preparation/pilot, user preview acceptance, persisted pilot approval, full computation, checked merge, fetching and local report generation are complete. The assistant verified the real measurements and inspected the plot. The user explicitly confirmed review and acceptance of the final curves/counts, including the higher cascade FP rates, in this chat. Full protocol, output layout, commands and recovery instructions: `docs/phase6.md`.

**Agreed choices (2026-11-19, user accepted the defaults)**

1. Compare both existing baseline seeds against both predicted-cascade seeds, sharing one proposal population. Both cascades use frozen seed-0 stage 1. No oracle comparison or training.
2. Original SAM ViT-H, official revision `dca509fe793f601edb92606367a655c15ac00fdf`, official `sam_vit_h_4b8939.pth` at `/data/checkpoints/sam/`. Record its observed SHA-256 and validate tensor structure. Isolate SAM in app `egohos-clutter`, preserving the existing pinned torch/torchvision stack and apps. Real CUDA compatibility is a pilot check.
3. SAM fp32, TF32 disabled; RGB uint8 HWC from original files and SAM's own preprocessing. Grid 32, prompt batch 32, predicted-IoU threshold 0.88, stability 0.95/offset 1, box/crop NMS 0.7, no crop layers or region cleanup; uncompressed RLE compressed on disk. Retain original-resolution coordinates throughout.
4. Filter raw empty masks and areas outside inclusive 0.1%–30%. Reject any raw overlap with second-order/255 pixels. Let F be joint hand/direct-object foreground: reject `|M & F| / |M| > 0.5` or IoU `> 0.3` against F or any separate 8-connected hand/object component. Components prevent a large union from hiding small-object overlap.
5. Clip surviving proposals to labelled background: `C = M & (original three-class GT == 0)`. Reject empty C and reapply the same area bounds. Do not split disconnected remnants into new instances. Area, FP fractions and distances all refer to C.
6. Additional greedy mask NMS on C at IoU `> 0.9`, descending SAM predicted IoU/stability and then stable mask ID. Other nested parts/regions remain. Interpret the denominator as candidate proposal regions, not independent object instances.
7. Minimum Euclidean pixel-centre distance to GT hands, divided by `sqrt(H² + W²)`. Adjacent pixels are 1 pixel apart; overlapping pixels are zero. No-hand images retain null distances and separate counts. Edges `[0, .01, .02, .05, .10, .20, 1]`, left-inclusive/right-exclusive except final endpoint included.
8. Object fraction `count(C & (P == 2)) / count(C)`; FP only if strictly `> 0.5`. Use final composed cascade class 2, explicitly `gt=None, modes=("predicted",)`, and the established SegFormer 512-shorter-side/saved-bf16/logit-upsampling protocol. Stream predictions, with no dense disk cache or writes to completed model runs.
9. Primary rates are candidate-level, split-separated, per seed plus system means. Show FP numerators, region denominators, contributing-image/source counts and descriptive seed ranges. Secondary per-image bin averages are recorded. Empty bins are undefined; no proposal-independent confidence interval is used.
10. Expensive RLE caches live at `/data/clutter_cache/sam/<proposal_hash>/`; analysis outputs at `/data/analysis/clutter_sam_v1/`. Fingerprints bind images, labels where relevant, checkpoints, package/settings and source provenance. Atomic checksummed completed-empty records support resume. Sorted modulo shards have unique ownership; merge checks the exact 1,626-image inventory. A shared atomic Dict lease excludes concurrent writers. Fetch only allowlisted small outputs into `results/phase6/clutter_sam_v1/`.
11. Fixed pilot: six in-domain images covering all four sources and area extremes, four distinct YouTube videos, seed 0; names frozen during preparation before SAM. Candidate-only overview and paginated retained/rejected sheets show IDs, GT, clipping, reasons and legends. Measure actual speed/memory before choosing the full timeout. Provisionally four L40S workers, each bounded to at most two hours; user review and explicit pilot approval must precede the full launch.

**Build**
- [x] `src/clutter_config.py`, `configs/phase6/clutter_sam_v1.yaml`: separate validated schema, outside the existing training-config glob.
- [x] `src/clutter_sam.py`: original-resolution filter arithmetic, compact RLE, exact distances/bins, continuous scoring, cache validation, ownership/coverage and aggregation.
- [x] `src/clutter_workflow.py`, `src/clutter_lock.py`: checkpoint/data provenance, SAM preparation, pilot, streaming final-weight inference, restart-safe image records, exclusive writers, checked merge and small-output allowlist.
- [x] `src/clutter_preview.py`, `modal_clutter.py`, `scripts/phase6_report.py`, `docs/phase6.md`.
- [x] `scripts/check_phase6_local.py`: **59/59** synthetic CPU checks, including two-phase interruption/resume, valid-empty work, strict thresholds, ignore handling, component overlap, clipping, exact distances, no hands, NMS, cache corruption/identity, 1,626-image shard coverage, approval binding, inference composition and local plot/report reproduction. Synthetic visual layouts inspected.
- [x] Existing checks still pass: Phase 3 **13/13**, Phase 4 **28/28**, Phase 5 **27/27**. macOS DataLoader tests required access outside the sandbox for PyTorch shared memory; downloads were disabled. Existing data/model/evaluation files, training configs, apps and requirements are unchanged.
- [x] User-run preparation: isolated image, validated/downloaded SAM weights, frozen real 1,626-image inventory and ten-image pilot manifest.
- [x] User-run pilot: original SAM ViT-H ran successfully on L40S; 996 proposals, 670 retained regions. SAM median/max time 2.60/7.03 s, peak allocated/reserved GPU memory 8.45/26.41 GB; pilot wall time 316.24 s including rendering and other work. Fetched 271 allowlisted files. Prepared source hashes and every preview checksum verified locally.
- [x] User accepts the preview/default background-proposal recipe and authorizes the final Phase-6 run (2026-11-19). Four L40S workers retain the two-hour per-worker ceiling.
- [x] User ran `approve_pilot`, persisted acceptance bound to the exact pilot record on the Volume and launched the detached full run.
- [x] Full 1,626-image job, exact-inventory merge, allowlisted fetch and local report generation completed. All four workers report complete and the writer lease is clear. Local aggregation matches the saved summary; source/data/model identities, per-record checksums, candidate/audit populations, original-size distance normalisation, integer pixel-fraction arithmetic, bins and FP numerators were checked. The assistant inspected the rendered plot.
- [x] User verifies the real curves/counts and Phase 6 is recorded complete with actual findings (explicit acceptance in this chat, 2026-11-19).

**Preparation/pilot commands (completed by the user)**

```bash
.venv/bin/modal run --detach modal_clutter.py::prepare --config configs/phase6/clutter_sam_v1.yaml
.venv/bin/modal run --detach modal_clutter.py::pilot --config configs/phase6/clutter_sam_v1.yaml
.venv/bin/modal run modal_clutter.py::fetch --config configs/phase6/clutter_sam_v1.yaml
```
The user has accepted `results/phase6/clutter_sam_v1/pilot/index.html` and the preview recipe. Full-run approval/launch, status and interruption recovery commands are in `docs/phase6.md`.

**Full-run launch commands (completed by the user)**

```bash
.venv/bin/modal run modal_clutter.py::approve_pilot --config configs/phase6/clutter_sam_v1.yaml
# Once approval succeeds:
.venv/bin/modal run --detach modal_clutter.py::run --config configs/phase6/clutter_sam_v1.yaml --shards 4
```

Final data have been fetched, and `scripts/phase6_report.py` reproduces the merged summary and renders the plot/report locally. The user confirmed review and acceptance of the final curves/counts on 2026-11-19.

**Pilot finding:** all ten images have labelled hands; distance-bin counts are 120 / 16 / 61 / 73 / 181 / 219. The retained population visibly includes untouched objects plus surface/clothing regions and nested parts. Green masks certify rule acceptance, not semantic object identity. The assistant reviewed all overviews and selected detail sheets; the user subsequently accepted the previews and authorized the default full run. The 1.17–3.17 SAM GPU-hour projections are median-/maximum-sample extrapolations, not full-runtime bounds. See `results/phase6/clutter_sam_v1/pilot_review.md`.

**Full result (computed, audited and verified by the user)**

| split | images | retained regions | baseline FP rate, 2-seed mean | predicted cascade FP rate, 2-seed mean | cascade - baseline |
|---|---:|---:|---:|---:|---:|
| test_indomain | 1,126 | 68,160 | 1.15% | 1.26% | +0.11 percentage points |
| test_outdomain | 500 | 34,614 | 0.65% | 0.75% | +0.10 percentage points |

- FP means object fraction strictly greater than 0.5 on a retained original-resolution background proposal. All models/seeds share the same 102,774 regions. Means/deltas are computed before rounding.
- In-domain per-seed FP counts (seeds 0/1): baseline **807/764**, cascade **855/863**, each out of 68,160. YouTube: baseline **214/238**, cascade **256/265**, each out of 34,614. Both same-seed comparisons have higher cascade FP rates on each split.
- The hypothesis of reduced clutter false positives is **not supported by this fixed proxy**. This is compatible with higher object IoU in Phase 5: the metrics measure different behaviour. No cause or statistical significance is established by two seeds.
- FP rates are generally higher near hands, without strict monotonicity. In the closest `[0, .01)` bin, baseline/cascade means are **2.20%/2.64%** in-domain and **1.77%/2.00%** on YouTube. All four bins below 0.1 have higher cascade means. Some farther bins favour the cascade; do not claim it is worse at every distance.
- SAM produced **167,652** proposals; **102,774** survived the fixed filters. Every image has retained candidates and labelled hands, so there are no empty-image or no-hand exclusions in these results. The candidate population includes surfaces/parts and correlated proposals, as accepted during pilot review.
- Four workers completed in about **38.4 minutes** each at the longest, about **2.5 summed GPU processing hours**; this is processing time, not a billing measurement. Bulk masks and final weights remain on the Volume; compact fetched measurements are about 14 MiB.
- Exact bin counts, per-seed rates, source counts, filter reasons, per-image-rate diagnostics and model/config provenance are in `results/clutter_fp_vs_distance.json` and `results/phase6_report.md`.

**Produces**
- `results/clutter_fp_vs_distance.png` and `.json`, plus `results/phase6_report.md`.
- Compact continuous fractions/distances and provenance at `results/phase6/clutter_sam_v1/`; bulk masks and weights stay on the Volume.

**Verify**
- **The user inspects ten pilot images** and agrees that retained candidates are plausible untouched regions, with understood exclusions and proposal-level limitations.
- Check exact image coverage, filtering/bin counts, shared model populations and provenance, then inspect the measured split-separated curves.
- Lower cascade FP rates and higher FP rates near hands are hypotheses, **not correctness conditions**. Report negative or ambiguous findings honestly. No significance or domain-gap-closing claim follows from this proxy or two seeds alone.

---

## Phase 7: Source analysis and failure gallery

**Status:** implementation, saved-data analysis, CPU survey and four actual prediction previews are complete and checked by the assistant. The 16-case selection covers five sources. User acceptance of the preview layout/captions, the remaining 12 selected cases, full-gallery category review and final user verification remain pending. Phase 7 is not complete. Full commands: `docs/phase7.md`; audits: `results/phase7/phase7_v1/{verification,preview_review}.md`.

**Agreed choices (2026-11-19, explicitly accepted by the user)**

1. Both existing baseline seeds and both predicted-cascade stage-2 seeds on the complete two test splits (1,126 + 500 original images). Both cascades retain frozen seed-0 stage 1. Existing val/head-only/hands-only/oracle metrics are context; no additional model/mode or val inference.
2. Reuse exact saved pooled split IoU and Boundary IoU. Source analysis uses explicitly labelled unweighted image means/medians/quartiles, per-run valid/undefined counts and shared-population comparisons. No exact pooled per-source pass; image sizes cannot recover missing sufficient statistics. Signed split differences are YouTube minus in-domain, including the actual higher YouTube scores.
3. Paired per-image comparisons use the intersection where all four main runs have defined class IoU. Average same-seed deltas within each image, then images equally. Object eligibility is 1,095 in-domain / 496 YouTube images; hand eligibility is all 1,626. Undefined values remain null, not zero. Seed spreads are descriptive, not confidence intervals.
4. One CPU scan verifies original-label hashes and records GT presence/absence, original class counts, source pixel support, 8-connected object components and original ID 5. Small means a GT component at most 1% of original frame area; components are not instances. ID 5 suggests a two-hand-interaction case and still requires visual review. GT-present and GT-absent source summaries are separate from model-dependent valid-score cohorts. Existing split confusion matrices supply optional pixel precision/recall diagnostics, distinct from the SAM-region metric.
5. At most 60 diagnostic RGB/GT survey thumbnails, four real preview cases, then up to 20 manually reviewed gallery cases. Target about two examples per supported appearance category plus improvements, regressions and four SAM disagreement diagnostics; overlapping tags are allowed. Unsupported/uncertain categories remain explicit. Fixed seed-0 display pair; both-seed scores and consistency recorded. Final gallery covers five sources, with at most two images per conservative video group. Raw filename video IDs are retained; three matching-ID YouTube download-copy aliases are grouped for diversity, not treated as independent videos.
6. Six aligned tiles (RGB/baseline/cascade over GT/baseline errors/cascade errors), full-frame context and optional matched close-ups. Blue hands, yellow objects, grey-hatched ignore; red object FP and cyan object FN; magenta selected eligible SAM-region outlines. FP includes valid GT hand/object confusion; SAM diagnostics are specifically GT-background candidates. Captions describe evidence and avoid causal claims.
7. Reuse compact Phase-6 measurements and only selected existing Volume proposal masks. Reconstruct eligible masks with the frozen filters and verify image/label/proposal/eligible-mask identities, mask IDs, area, distance and continuous fractions. No SAM generation fallback. Include surface/part limitations honestly. Preserve the completed negative clutter-FP result and fixed thresholds.
8. Separate validated config and dedicated `/data/analysis/phase7_v1/` / `results/phase7/phase7_v1/` outputs. The new app reuses the working clutter runtime and strict frozen source/runtime checks. One L40S, 30-minute provisional GPU ceiling, serial selected cases and atomic per-image commits; CPU survey is separately bounded. A separate writer lease, safe resume and allowlisted 256-MiB fetch preserve old data/models/experiments. The user launches all remote output-producing commands. Full gallery rendering requires explicit user acceptance of the four actual previews; final phase completion requires a separate user review.

**Build and computed outputs**

- [x] Independent config, saved-result/schema/provenance validation, source summaries, paired deltas, signed split tables, pixel precision/recall and source plot.
- [x] Deterministic shortlist, editable manual review and category/evidence validation; no automatically assigned appearance categories.
- [x] CPU original-label scan, selected predicted-only inference via the frozen predictor, original-coordinate FP/FN panels, selected SAM cache reconstruction, exact saved-score/fraction reconciliation, resumable records and bounded fetch entrypoints implemented.
- [x] Focused synthetic Phase-7 checks **57/57**, covering isolated container startup, missing/undefined classes, pooling distinctions, joins, source aliases, label geometry, paired arithmetic, FP/FN/crop alignment, selection/diversity, provenance/cache rejection, interruption/resume, preview acceptance and safe fetching. Synthetic panel layout inspected.
- [x] Existing Phase-3–6 regressions re-run successfully: **13/13**, **28/28**, **27/27**, **59/59**. Downloads were disabled; macOS DataLoader checks used shared-memory access outside the sandbox. Phase-7 CPU analysis imports remain torch-free; the new Modal app imports locally without invoking remote work.
- [x] Local saved-data report computed from all 1,626 test images; Phase-6 aggregation and frozen hashes rechecked. Five source populations: Ego4D 774, EPIC 217, THU 88, own 47, YouTube 500. Machine-readable provenance/metrics and plot are in `results/phase7/phase7_v1/`.
- [x] First user survey launch failed before function execution because the container could not import the local `modal_clutter` app. The import/image construction is now local-only; the isolated startup check omits that app file and passes. Initial local-only metadata was archived under `results/phase7/phase7_v1_startup_failed_<fingerprint>/` and rebuilt with the corrected code identity. The subsequent user-run CPU survey completed successfully.
- [x] User-run CPU scan and visual survey complete, fetched and incorporated into GT-present/absent comparisons. Independent local reconstruction verifies all 1,626 records, class-pixel totals against all eight primary split confusion matrices, source metrics, shortlist reproducibility and all 61 survey file hashes/60 decodable PNGs. Read-only Volume inspection confirms that remote outputs contain the survey only.
- [x] Assistant reviewed all 60 real RGB/GT survey thumbnails and selected 16 draft cases with captions/evidence, candidate IDs and four preview cases (own 4K, YouTube improvement, Ego4D cascade-only region, EPIC baseline-only region). `review.before_gallery_review.json` preserves the original empty review. All five sources and the video cap validate. Appearance tags and candidate semantic descriptions remain draft interpretations pending actual prediction/eligible-mask panels and user review; transparent-object and motion-blur categories are explicitly uncertain.
- [x] User computed/fetched four actual prediction previews. The assistant inspected all eight full-frame/matched-crop panels and verified 16 exact saved-IoU/GT-count/FP-FN records, both selected SAM-region metadata/fractions and all 13 referenced file checksums. Layout, crop alignment, colours and run/seed labels look consistent; visible evidence supports the paired improvement/regression and opposite SAM-event examples. The latest preview invocation reused all four case records; its null GPU field/zero new-prediction count does not imply the original inference used no GPU.
- [ ] User personally accepts the four actual previews and records `approve-preview` before full rendering.
- [ ] Selected full gallery rendered and fetched, with paired improvements/regressions and honest SAM-region descriptions.
- [ ] User verifies real categories/captions and final source/metric comparisons; actual final findings and limitations recorded before Phase 7 is marked complete.

**Audited quantitative findings (prediction gallery and final user review pending)**

- Existing pooled object IoU remains baseline/predicted cascade **46.2/46.6** in-domain and **58.1/59.7** on YouTube. Source/content mixtures do not isolate a domain effect.
- On the shared four-run-defined image cohort, mean per-image object-IoU cascade deltas are **+1.53** points Ego4D (754 images), **+1.47** EPIC (212), **+2.59** THU (85), **-0.12** own (44) and **+2.03** YouTube (496). These are image averages on a model-dependent cohort, not pooled source IoU or category prevalence.
- The CPU scan identifies **1,037/89** GT object-present/absent in-domain images and **480/20** on YouTube. On the separate, model-independent GT-present cohort, mean paired object-IoU deltas are **+1.61** Ego4D (714 images), **+1.53** EPIC (204), **+2.86** THU (77), **-0.13** own (42), **+2.10** YouTube (480). Ego4D accounts for **87.14%** of original in-domain frame pixels. The scanner's class-pixel totals exactly equal the GT row sums of the saved primary confusion matrices.
- The four real previews show the displayed seed-0 book regression (object IoU **21.99 -> 5.52**, with much more bedsheet FP), tray improvement (**0.47 -> 67.37**), headlamp-part candidate FP (**0% -> 100%** object fraction) and packaging-region reduction (**85.26% -> 12.26%**). These deliberately selected cases do not estimate prevalence or isolate causes. The book difference reverses for seed 1, as recorded. Final category captions and the remaining 12 panels still require review; the two shown SAM regions were visually confirmed as a headlamp part and a packaging-looking background proposal.

---

## Phase 8: Write-up

**Status:** handoff prompt prepared at the user's request: `phase8_prompt.md`. Documentation drafting is the next requested work. Four real reviewed prediction examples are available; the remaining Phase-7 gallery/acceptance state must remain explicit in the write-up. No README/resume/publication completion is claimed yet.

**Produces**
- `README.md`: problem, method, results table, plots, failure gallery.
- Resume bullets filled with the real numbers.
- Pushed to GitHub.

**Verify**
- You can explain every design choice (see the interview questions in `skild_project.md`, section 12).

---

## Stretch phases (only if time remains)

- **S1:** ONNX export + int8 quantization, CPU FPS on the Mac, accuracy drop.
- **S2:** B0 vs B2: accuracy vs speed.
- **S3:** Out-of-domain augmentation, and whether it closes the gap.
- **S4:** Three-stage cascade: hands -> contact boundary -> object.
- **S5:** Data scaling curve (10 / 25 / 50 / 100%).
- **S6:** Zero-shot Grounding DINO + SAM baseline.
- **S7:** Label ablation: 2nd-order objects as ignore vs as background.
- **S8:** Video demo GIF on a YouTube egocentric clip.
