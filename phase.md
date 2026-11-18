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

---

## Phase 0: Environment and repo skeleton

**Build**
- [x] Python 3.12 venv, `modal` installed, Modal login verified (`modal volume list` works)
- [ ] `git init`, `.gitignore` (`.venv/`, `data/`, checkpoints, large outputs)
- [ ] `requirements.txt` (local) and folder skeleton: `src/`, `configs/`, `scripts/`, `results/`
- [ ] `modal_app.py` with the pinned GPU image (PyTorch, transformers, albumentations, opencv-headless, gdown) and the `egohos` Volume

**Verify**
- A trivial Modal GPU function runs on L40S and prints the torch version, CUDA availability and GPU name.

---

## Phase 1: Dataset onto the Modal Volume

**Build**
- Modal function that downloads EgoHOS with `gdown` straight into the `egohos` Volume and unzips it (fallback: browser download + `modal volume put`).
- Dataset inspection script.

**Produces**
- `/data/egohos/{train,val,test_indomain,test_outdomain}/{image,label,contact}` on the Volume.
- `results/dataset_stats.json`: file counts, image sizes, unique label values, per-class pixel share.

**Verify**
- Counts match the paper: train 8,993 / val 1,124 / test_indomain 1,126 / test_outdomain 500.
- Every image has a matching label file.
- Label PNGs contain only values 0–8.

---

## Phase 2: Data pipeline and sanity check

**Build** (`src/data.py`)
- Label remap 0–8 -> {0, 1, 2, 255}.
- Training augmentations: random resize 0.5–2.0, random crop 512, horizontal flip, colour jitter.
- Pre-resized training copy on the Volume (size chosen from the Phase 1 image-size stats) to keep data loading fast.
- Fixed 500-image dev subset (saved list of file names).
- Script to render overlays.

**Produces**
- `results/sanity/`: 20 random samples (image | original label | remapped label overlay), plus a few augmented samples.

**Verify**
- **You look at the overlays:** hands, 1st-order objects and ignored 2nd-order objects are coloured correctly.
- Remapped masks contain only {0, 1, 2, 255}.

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
