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

- **Done:** Phases 0, 1 and 2 (all verified). Phase 2 added `src/data.py`, `src/preresize.py`, `src/sanity.py` and the `preresize` / `dev_subset` / `sanity` entrypoints in `modal_data.py`.
- **Next:** Phase 3 (model, training loop, evaluation; smoke test on the dev subset).
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
- **`modal_data.py`:** CPU-only app `egohos-data` with its own lighter image (no torch) for data jobs. Entrypoints: `download`, `stats`, `preresize`, `dev_subset`, `sanity`.
- **`src/data.py`** is shared by both apps and must stay torch-free. Training reads `EgoHOSDataset("train", "/data/egohos_512", TrainTransform(), file_list=...)`; evaluation reads the original splits with `EgoHOSDataset(split, "/data/egohos", EvalTransform())` (variable sizes, so batch size 1 or grouping by size).
- **Dev subset:** `configs/dev_subset_500.txt` (file stems) lives in the repo; Phase 3 must add `configs/` to the GPU image (e.g. `.add_local_dir("configs", "/root/configs")`).
- **Local multiprocessing note:** on macOS, `ProcessPoolExecutor` code must run from a script file with an `if __name__ == "__main__":` guard (spawn start method); Modal's Linux containers use fork.
- **`modal_app.py`:** GPU app `egohos-segformer` with the torch image. Entrypoint `check`. The GPU type comes from the `EGOHOS_GPU` env var (default `L40S`).
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

**Build**
- `src/model.py`: SegFormer-B0 from `nvidia/mit-b0` with a new head (any number of classes) and a helper that freezes the encoder.
- `src/train.py`: AdamW (encoder lr 6e-5, head 10x), warmup + polynomial decay, bf16, checkpointing, metrics JSON, YAML configs.
- `src/evaluate.py`: confusion-matrix per-class IoU + mIoU at full resolution, and Boundary IoU.

**Verify** (short run on the 500-image dev subset, about 2–3 min on GPU)
- Loss goes down.
- The model can overfit a handful of images (near-perfect IoU on them).
- Evaluation runs on the full val set and writes a JSON file.
- Predicted masks visualised for a few val images.

---

## Phase 4: Single-stage experiments (full data, run in parallel)

**Runs** (3 L40S containers at once, about 20–40 min each)
- (a) Head-only (frozen encoder), 3 classes.
- (b) Full fine-tune, 3 classes: the main single-stage model.
- (c) Hands-only model (background vs hand): stage 1 of the cascade.

**Produces**
- Per-class IoU, mIoU and Boundary IoU on val / test_indomain / test_outdomain for every run.
- `results/phase4_table.md`.

**Verify**
- Full fine-tune beats head-only.
- The numbers are sensible next to the EgoHOS paper (not directly comparable, but the same ballpark).
- Prediction overlays look right.

---

## Phase 5: Two-stage cascade

**Build**
- 3 -> 4 channel patch-embedding surgery: copy the RGB weights and zero-init the new channel.
- Corrupted ground-truth hand masks for training: random dilation/erosion (3–15 px), drop a hand (p≈0.1), random blobs.
- Stage 2 trained to predict background vs interacting object, from RGB + corrupted hand mask.
- `src/cascade.py`: stage-1 predictions -> stage 2 -> final mask (stage-1 hands + stage-2 objects).
- To decide here: in the stage-2 target, hand pixels are background or ignore.

**Produces**
- Interacting-object IoU (and the full 3-class table) for:
  - single-stage
  - cascade with predicted hands
  - cascade with ground-truth hands (oracle)

**Verify**
- Unit check: at initialisation, the 4-channel model gives the same output as the 3-channel model for any hand mask.
- Overlays of the corrupted masks look realistic.
- Ordering: oracle ≥ cascade with predicted hands. Whether the cascade beats single-stage is the actual finding.

---

## Phase 6: Clutter analysis with SAM

**Build** (`src/clutter_sam.py`)
- SAM ViT-H automatic mask generator on test_indomain + test_outdomain (1,626 images), split across GPUs and cached on the Volume.
- Filtering:
  - remove masks that overlap ground-truth hands/objects (IoU > 0.3 or >50% inside)
  - remove huge masks (>30% of the image) and tiny ones (<0.1%)
- Per clutter object:
  - fraction of its pixels predicted as "interacting object" -> false positive if above 0.5
  - distance to the ground-truth hand (normalised by the image diagonal), binned

**Produces**
- `results/clutter_fp_vs_distance.png` (single-stage vs cascade) and the JSON behind it.

**Verify**
- **You look at about 10 images** with their clutter masks and agree they are untouched objects.
- The plot makes sense: more false positives near the hand.

---

## Phase 7: Domain gap and failure gallery

**Produces**
- In-domain vs out-of-domain per-class IoU, and the drop, for every model.
- Failure gallery: transparent objects, occluded objects, motion blur, small objects, two-hand objects, lighting.

**Verify**
- Each failure category has real examples picked from the predictions, with a one-line explanation.

---

## Phase 8: Write-up

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
