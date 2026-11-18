# Project: "What is the person actually handling?"
### Hand + interacting-object segmentation on EgoHOS with SegFormer-B0

This file is the full context for the computer vision project I'm building for the **Skild AI Machine Learning Engineer (fresher) role** that is hiring at IIT BHU (posting: Bengaluru). Paste this into a new chat to continue.

---

## 1. Context

- **Me:** Mohan Kumar, IIT (BHU) Varanasi, IDD Mining Engineering (2027), CGPA 8.17. Resume is currently LLM/agents/open-source heavy (LFX WasmEdge, Protocol Labs py-libp2p, SellerSetu, Excali QLoRA fine-tune of Qwen2.5-Coder-7B, ChatJarvis). It has no computer vision project yet, even though I did a computer vision course.
- **Goal of this project:** add a strong, defensible computer vision project to the resume that maps to the Skild AI JD.
- **Where it goes on the resume:** first item under Projects, **replacing ChatJarvis**. Also add "Relevant Coursework: Computer Vision, Machine Learning, Deep Learning" and move PyTorch/OpenCV/computer vision to the front of Skills.
- **Constraint:** no self-collected dataset. Use an existing public dataset only.
- **Plan:** build the core version in one continuous night, with the AI writing code and me running it on Modal GPUs. Then spend 1–2 days understanding every part before putting it on the resume.

### Why this project fits Skild AI
- Skild's core bet is that robots learn by watching human video (egocentric/head-camera video) and then need less than 1 hour of robot data. Their blog: https://www.skild.ai/blogs/learning-by-watching
- A founding researcher said they use perception models on egocentric video to find "where their hands are grasping objects." This project is that perception step.
- JD lines it covers:
  - "Train and fine-tune computer vision models for … segmentation … and scene understanding."
  - "Analyze model failures and guide improvements to datasets, model architectures, and training strategies."
  - "Optimize models for efficient inference on edge devices."
  - "Explore new research and open-source models, and adapt promising approaches."
  - Experimental evaluation and systematic debugging.

---

## 2. What the model does

**Input:** one RGB image, usually from a head-mounted (egocentric) camera.

**Output:** a same-size segmentation mask. Every pixel gets one of 3 labels:
- `0` **background**: everything else, *including objects nobody is touching (clutter)*
- `1` **hand**
- `2` **interacting object**: whatever a hand is directly touching/holding right now

It does **not** name objects. It says "these pixels are the thing being handled," not "knife."

### Example
Kitchen photo: right hand holds a knife, left hand holds an onion on a cutting board, a cup and a phone are on the counter.

| Thing in photo | Label | Why |
|---|---|---|
| Both hands | hand | they are hands |
| Knife | interacting object | right hand touching it |
| Onion | interacting object | left hand touching it |
| Cutting board | background / ignored | only touched indirectly, through the onion (2nd-order) |
| Cup, phone | background | clutter, not touched |

**The hard case, and the core research question:** a hand rests right next to the phone but isn't touching it. A weak model colours the phone as "interacting object"; a good model leaves it as background.

> **Research question:** Can the model distinguish an object being used from nearby clutter, and does giving it explicit hand information (a cascade) help?

### What it will NOT do (v1). Good "next steps" for the interview:
- No object names / categories.
- No left vs right hand (merged).
- No action understanding (e.g., "cutting").
- No 3D, no temporal reasoning (each frame independent).

---

## 3. Dataset: EgoHOS

- Paper: *Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications* (Zhang et al., ECCV 2022). https://arxiv.org/abs/2208.03826
- Repo: https://github.com/owenzlz/EgoHOS
- **11,243 egocentric images** from Ego4D (7,458), EPIC-KITCHENS (2,121), THU-READ (806), and the authors' own videos (350).
- **Splits:**
  - train: **8,993**
  - val: **1,124**
  - test_indomain: **1,126**
  - test_outdomain: **500** (sampled from 30 YouTube egocentric videos; this is the generalization test)
- **Download:** `download_datasets.sh` does:
  ```bash
  pip install gdown
  mkdir data && cd data
  gdown https://drive.google.com/file/d/1sk0TVEhZESNF67OW3fz9D5coqpIWkwuK/view?usp=sharing
  unzip egohos_dataset.zip && rm egohos_dataset.zip
  ```
  Exact size isn't stated (probably a few GB). **Risk:** Google Drive / gdown sometimes hits a quota error. Fallback: download in the browser, then `modal volume put <volume> egohos_dataset.zip`.
- **Folder structure:**
  ```
  [egohos root]
    train/          image/  label/  contact/
    val/            image/  label/  contact/
    test_indomain/  image/  label/  contact/
    test_outdomain/ image/  label/  contact/
  ```
- **Original label ids (in `label/` PNGs):**
  ```
  0 -> background
  1 -> left hand
  2 -> right hand
  3 -> 1st order interacting object by left hand
  4 -> 1st order interacting object by right hand
  5 -> 1st order interacting object by both hands
  6 -> 2nd order interacting object by left hand
  7 -> 2nd order interacting object by right hand
  8 -> 2nd order interacting object by both hands
  ```
  `contact/` masks: ones = dense hand-object contact boundary region. The paper generates these by dilating the hand and object masks and taking their overlap.
- 1st order = directly touched. 2nd order = indirectly interacted (e.g., bowl under the spoon). The paper itself only studies directly interacting objects.

### Label mapping (our 3 classes)
```
0        -> 0 background
1, 2     -> 1 hand
3, 4, 5  -> 2 interacting object
6, 7, 8  -> 255 (ignore in loss)   [default choice]
            alternative: -> 0 background (run as an ablation if time allows)
```
Be ready to justify this choice in interviews. "Ignore" avoids punishing the model for pixels whose label is ambiguous under our 3-class definition.

**Note:** because left/right are merged, **horizontal flip augmentation is safe**. With left/right labels, flipping would require swapping the classes.

---

## 4. Model: SegFormer-B0

- Paper: *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers* (Xie et al., NeurIPS 2021). https://arxiv.org/abs/2105.15203
- Hugging Face: `SegformerForSemanticSegmentation`. Initialise the encoder from `nvidia/mit-b0` (ImageNet-pretrained) with a new 3-class decoder head (`num_labels=3`, `id2label`, `ignore_mismatched_sizes=True` if loading a fine-tuned checkpoint).
- About **3.7M parameters**, small enough for edge deployment.
- **Architecture points to know cold:**
  - Hierarchical **Mix Transformer (MiT)** encoder with 4 stages, producing features at 1/4, 1/8, 1/16, 1/32 resolution.
  - **Overlapping patch embeddings** (strided convs with overlap).
  - **Efficient self-attention:** keys/values are spatially reduced by ratio R, so attention cost goes from O(N²) to O(N²/R).
  - **Mix-FFN:** a 3×3 depthwise conv inside the FFN provides positional information, so there's **no positional encoding**, which makes the model robust to different input resolutions.
  - **Lightweight all-MLP decoder** fuses the 4 scales. The output is at 1/4 resolution and gets upsampled to the input size.

---

## 5. Experiments (the version we're building)

### Must-have (one night)

1. **Data pipeline + sanity check**
   - Remap labels to 3 classes; visualise 20 random samples with overlays to confirm the mapping is right.
   - A **dev subset of ~500 training images** for fast debugging (2–3 min runs).
   - **Final runs use the full 8,993 training images.** Evaluation always uses the **full** val / test_indomain / test_outdomain sets.

2. **Single-stage SegFormer-B0 (3 classes)**
   - (a) **Head-only**: freeze the encoder, train only the decoder. This is a quick sanity check; full fine-tuning is expected to win.
   - (b) **Full fine-tuning**: the main single-stage model.

3. **Two-stage cascade**, testing the EgoHOS paper's finding with a tiny model
   - EgoHOS paper (Section 4, Table 5): predicting hands first and feeding the hand masks into the next stage improves interacting-object segmentation. They used large models (Swin-L + UperNet, or ResNet-18 + HRNet). **Our question: does this still hold for a tiny edge model (B0)?**
   - **Stage 1:** B0 hand model (background vs hand).
   - **Stage 2:** B0 with a **4-channel input** (RGB + hand mask) predicting the interacting object (and/or all 3 classes).
     - Modify the first overlapping patch-embedding conv from 3 to 4 input channels: copy the pretrained RGB weights and **initialise the new channel's weights to zero**, so training starts as if it were the pretrained RGB model.
   - **Train/test mismatch:** at test time stage 2 gets *predicted* hand masks, which are imperfect. So **train stage 2 on corrupted ground-truth hand masks**: random dilation/erosion (kernel ~3–15 px), occasionally drop a hand (p≈0.1), add small random blobs.
   - Report three numbers:
     - single-stage
     - cascade with predicted hands (the real-world number)
     - cascade with ground-truth hands (an **oracle upper bound**)

4. **Clutter analysis**: makes the research question measurable
   - Problem: EgoHOS does **not** label non-interacting objects (clutter is just background), so there's no ground truth for clutter.
   - Fix: run **SAM's automatic mask generator** on the test images (in-domain + out-of-domain, 1,626 images) to get masks for every object.
     - Remove SAM masks that overlap a ground-truth hand or interacting object (e.g., IoU > 0.3 or >50% of the mask inside a ground-truth region).
     - Remove huge masks (>~30% of the image: walls/table) and tiny masks (<~0.1%).
     - The remaining masks are **clutter objects**.
   - For each clutter object: the fraction of its pixels the model predicts as "interacting object." Count it as a **false positive** if above a threshold (e.g., 0.5; report the threshold).
   - **Distance to hand:** the minimum pixel distance from the clutter mask to the ground-truth hand mask, normalised by the image diagonal. Bin it (e.g., touching-adjacent / near / far).
   - **Output:** a plot of clutter false-positive rate vs distance to hand, **single-stage vs cascade**.
   - SAM is a *measurement tool* here, not part of the model.

5. **In-domain vs out-of-domain gap**
   - Per-class IoU on test_indomain vs test_outdomain for every model. Report the drop.

6. **Failure gallery**
   - Transparent/glass objects, objects hidden by fingers, motion blur, small objects, two-hand objects, strong lighting changes.

### Stretch (pick 1–2 if time remains)
- **Edge deployment:** export to ONNX, int8 quantization (dynamic and/or static), measure **FPS on Mac CPU** (warm-up runs, batch size 1, fixed thread count) and the accuracy drop.
- **B0 vs B2:** accuracy vs speed trade-off.
- **Out-of-domain augmentation:** strong colour/brightness/contrast jitter + blur, or a simplified version of the EgoHOS paper's context-aware compositional augmentation (paste hand-object pairs onto other backgrounds). Measure whether it closes the in-domain → out-of-domain gap.
- **Three-stage cascade:** hands → contact boundary (from `contact/` labels) → interacting object, each stage concatenating previous outputs as extra input channels (exactly the EgoHOS pipeline).
- **Data scaling curve:** train B0 on 10% / 25% / 50% / 100% of the training set and plot IoU vs data size ("would more labelled data still help?"). About 1 extra GPU-hour.
- **Zero-shot baseline:** Grounding DINO + SAM prompted with "hand" and "object held by hand" to show how far off-the-shelf foundation models get without training.
- **Label ablation:** 2nd-order objects as ignore vs as background.
- **Video demo:** run frame-by-frame on a YouTube egocentric cooking clip, produce an overlay video/GIF for the README.

### Reference
- Use the EgoHOS paper's numbers (Table 5 etc.) as a reference point, but note that their label granularity (left/right hands, left/right/both-hand objects) differs from our 3 classes, so the numbers are **not directly comparable**.

---

## 6. Training setup (starting point)

| Setting | Value |
|---|---|
| Model | SegFormer-B0, encoder from `nvidia/mit-b0`, 3-class head |
| Input | 512×512 crops |
| Augmentation | random resize (0.5–2.0) + random crop 512, horizontal flip, colour jitter |
| Loss | cross-entropy with `ignore_index=255`; optionally + Dice loss for class imbalance (hands/objects cover few pixels) |
| Optimizer | AdamW, lr ≈ 6e-5 for encoder, ~10× for the decoder head, weight decay 0.01 |
| Schedule | linear warmup (~1,000–1,500 iters) then polynomial decay (power 1.0) |
| Batch size | 16 |
| Epochs | ~20 (≈ 11k iterations at bs 16 on 8,993 images) |
| Precision | bf16 mixed precision |
| Evaluation | per-class IoU from a confusion matrix, on full val/test sets at full resolution (upsample logits to the original size) |
| Data loading | pre-resize images once; `num_workers` ≈ 8; request ≥ 8 CPU cores per Modal container |
| Logging | save metrics JSON + checkpoints to a Modal Volume (optionally wandb) |

### Metrics
- **Per-class IoU** (hand, interacting object) and mIoU.
- **Boundary IoU** (Cheng et al. 2021): IoU computed only on pixels within a small distance of the mask contour. It matters for thin objects like screwdrivers and knives.
- **Clutter false-positive rate** vs distance to hand.
- **In-domain → out-of-domain drop.**
- **FPS / latency** on CPU (stretch).
- Report all of these for: head-only, full fine-tune, cascade (predicted hands), cascade (oracle hands).

---

## 7. Compute: Modal

### GPU choice: **NVIDIA L40S** (`gpu="L40S"`)
SegFormer-B0 is tiny (3.7M params) and uses only a few GB of GPU memory at 512×512. The bottleneck is usually CPU image decoding/augmentation, not GPU power, so a bigger GPU mostly sits idle.

Modal prices (Nov 2026, https://modal.com/pricing):

| GPU | Memory | $/hour | Verdict |
|---|---|---|---|
| L4 | 24 GB | $0.80 | fine for B0, slower |
| A10 | 24 GB | $1.10 | fine for B0 |
| **L40S** | 48 GB | **$1.95** | **chosen: best value, fast for B0, B2 and SAM** |
| A100 40 GB | 40 GB | $2.10 | very good alternative |
| A100 80 GB | 80 GB | $2.50 | unnecessary |
| RTX PRO 6000 (Blackwell) | 96 GB | $3.03 | works but overkill |
| H100 | 80 GB | $3.95 | unnecessary |

- **Run experiments in parallel** on several L40S containers (e.g., head-only, full fine-tune and hands-only stage 1 at the same time). Several cheap GPUs in parallel beat one expensive GPU run sequentially.
- **Time per run:** about 20–40 min for a full B0 run on the full training set.
- **Total GPU usage:** about **4–7 GPU-hours** → roughly **$8–14 on L40S**. Most of it runs in parallel, so actual waiting is ~2–3 hours of the night.
- **If using RTX PRO 6000 (`gpu="RTX-PRO-6000"`) or any Blackwell GPU:** it needs a PyTorch build for **CUDA 12.8+ (PyTorch ≥ 2.7)**; older builds fail with "no kernel image is available". Pin a recent PyTorch in the Modal image regardless.
- **To save more GPU if needed:** train on a ~3K-image subset (~10 min runs), use 384 px instead of 512, or use a coarser SAM point grid (`points_per_side`).

### Modal pieces we'll use
- `modal.App`, `@app.function(gpu="L40S", cpu=8, timeout=..., volumes={...})`
- `modal.Volume.from_name("egohos", create_if_missing=True)` for the dataset, checkpoints, predictions and metrics.
- A custom `modal.Image` with pinned PyTorch, `transformers`, `albumentations`, `opencv-python-headless`, `gdown`, SAM.
- Launch parallel jobs with `.spawn()` / `.map()`; run with `modal run <file>.py`.

### Local environment (my Mac)
- Apple Silicon (arm64), system Python **3.14**, which is too new for some libraries.
- **Use a Python 3.12 environment**, e.g. via `uv`:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv && source .venv/bin/activate
  uv pip install modal
  ```
- `~/.modal.toml` already exists (probably already authenticated), but the `modal` Python package is not installed yet.
- The Mac is only used for: launching Modal jobs, CPU latency benchmarks (ONNX), plots and the README.

---

## 8. Suggested repo structure

```
egohos-segformer/
  README.md                 # problem, results table, plots, failure gallery, demo GIF
  modal_app.py              # Modal app: image, volume, download/train/eval/sam functions
  src/
    data.py                 # EgoHOS dataset, label remap, augmentations, corrupted-hand-mask generator
    model.py                # SegFormer-B0 builder, 3->4 channel patch-embed surgery, freezing helpers
    train.py                # training loop (bf16, AdamW, warmup + poly LR, checkpointing)
    evaluate.py             # confusion-matrix IoU, boundary IoU, in- vs out-of-domain eval
    cascade.py              # stage-1 inference -> stage-2 inference pipeline
    clutter_sam.py          # SAM masks -> clutter filtering -> false-positive vs distance analysis
    export_onnx.py          # (stretch) ONNX export, int8 quantization, CPU FPS benchmark
  configs/                  # one YAML per experiment (head_only, full_ft, hands_stage1, cascade_stage2, ...)
  scripts/                  # visualise samples, make plots, make demo video
  results/                  # metrics JSON, plots, gallery images (small files only)
```

---

## 9. One-night plan (~8–9 hours + 1–2 h buffer)

| Time | Work | GPU running in parallel |
|---|---|---|
| 0:00–0:45 | Python 3.12 env, install Modal, create Volume, download EgoHOS into it, label remap, visualise 20 samples | dataset download |
| 0:45–1:45 | Training script + evaluation script (per-class IoU, in/out-of-domain); smoke test on the 500-image dev subset | — |
| 1:45–2:45 | Launch 3 runs at once; meanwhile write cascade code (4-channel input) and SAM clutter script | full fine-tune, head-only, hands-only stage 1 (~20–40 min each) |
| 2:45–4:00 | Train stage 2 of the cascade (RGB + corrupted hand masks); run stage-1 inference on test sets | cascade stage 2 + SAM masks on 1,626 test images split across several GPUs |
| 4:00–5:30 | Evaluate everything: per-class IoU, boundary IoU, in- vs out-of-domain drop, clutter false-positive vs distance plot, oracle cascade | — |
| 5:30–7:00 | Stretch: pick 1–2 (ONNX int8 FPS on Mac, B0 vs B2, out-of-domain augmentation, three-stage contact cascade, data scaling curve) | stretch runs |
| 7:00–8:30 | Results table, failure gallery, README, push to GitHub | — |

**Risks:**
- gdown quota error → browser download + `modal volume put`.
- Data loading bottleneck → pre-resize images, more CPU cores, more workers.
- Small IoU differences (1–2 points) between experiments can be noise → use full training data for final runs and full test sets; if time allows, repeat the key comparison with a second seed.

---

## 10. What I need to learn

**Refresh (from my computer vision course):** convolutions, attention, cross-entropy, data augmentation, IoU.

**New:**
1. **Semantic segmentation in practice:** per-pixel cross-entropy, `ignore_index`, class imbalance (weighted cross-entropy / Dice), per-class IoU from a confusion matrix, Boundary IoU.
2. **SegFormer internals:** hierarchical MiT encoder, overlapping patch embeddings, efficient self-attention with reduction ratio R (O(N²) → O(N²/R)), Mix-FFN (no positional encoding → resolution-flexible), all-MLP decoder, why it suits edge devices.
3. **Transfer learning:** frozen encoder vs full fine-tune, lower lr for encoder, AdamW, warmup + polynomial LR schedule, bf16 mixed precision.
4. **Cascade design:** 3→4 channel patch-embedding surgery (copy RGB weights, zero-init new channel), train/test mismatch (ground-truth vs predicted masks) handled with corrupted ground-truth masks, oracle upper bound.
5. **SAM:** promptable segmentation; automatic mask generator = grid of point prompts + filtering by predicted IoU and stability score.
6. **Domain shift:** why out-of-domain accuracy drops (lighting, camera, object differences), how augmentation helps.
7. **Deployment:** ONNX export, dynamic vs static int8 quantization, measuring latency properly (warm-up, batch size 1, fixed threads).
8. **Tools:** Modal (App, GPU Functions, Volumes, parallel `.spawn()`/`.map()`), Hugging Face `SegformerForSemanticSegmentation`, `albumentations`.

**Reading after the build night:**
- SegFormer paper, Section 3 (method): https://arxiv.org/abs/2105.15203
- EgoHOS paper, Section 4 + Table 5 ablation: https://arxiv.org/abs/2208.03826

**Rule:** read every file as it's written, and don't put the project on the resume until I can explain every design choice.

---

## 11. Resume bullets (fill in real numbers after experiments)

**Project title idea:** *EgoHOS Interaction Segmentation: What Is the Person Actually Handling?*
Exposure: PyTorch, SegFormer, Hugging Face, SAM, Modal, ONNX

- Fine-tuned SegFormer-B0 (3.7M params) to segment hands and actively manipulated objects in egocentric video frames (EgoHOS, 11K images), reaching **X** hand IoU and **Y** interacting-object IoU, with a **Z**-point drop on out-of-domain YouTube footage.
- Built a hand-guided two-stage cascade (RGB + predicted hand mask, trained with corrupted masks to match test-time errors) that improved interacting-object IoU by **A** points and reduced clutter false positives by **B%** over single-stage.
- Designed a clutter-confusion benchmark using SAM-generated object masks to measure false "in-use" predictions vs distance from the hand, since the dataset has no labels for untouched objects.
- *(stretch)* Exported to int8 ONNX, running at **N FPS** on a laptop CPU with **<W** IoU loss.

---

## 12. Interview questions to prepare for this project

- Why SegFormer-B0 and not a bigger model / Mask R-CNN / SAM fine-tuning?
- Explain SegFormer's efficient self-attention and why it has no positional encoding.
- Why did you map 2nd-order objects to ignore instead of background? What changes if you don't?
- Why is horizontal flip safe here, and when would it not be?
- How did you handle class imbalance?
- How is IoU computed? Why per-class IoU instead of pixel accuracy? What does Boundary IoU add?
- Why did full fine-tuning beat head-only? What did the head-only result tell you?
- How did you modify the model to take 4 input channels? Why zero-init the new channel?
- What's the train/test mismatch in the cascade, and how did you handle it? What does the oracle number tell you?
- How did you measure clutter confusion without clutter labels? What are the weaknesses of using SAM for this?
- Why does performance drop out-of-domain? What would you try next to fix it?
- Is a 1–2 point IoU difference significant? How would you check?
- What failure modes did you find, and what data would you collect to fix them?
- How would you deploy this on a robot? Latency, quantization, accuracy trade-off.
- How does this help a robot learn from human video? (→ which object is used, where the hand contacts it, filtering clips, then tracking / grasp estimation as next steps.)
- What would v2 add? (temporal consistency across video frames, left/right hands, object categories, contact points, 3D.)
