"""Training loop: iteration-based, AdamW (encoder lr, decode head lr x head_lr_mult, no weight decay on norms and
biases), linear warmup + polynomial decay, bf16 autocast, cross-entropy on logits upsampled to the crop size
(255 ignored). Evaluation of the final model runs in the same process.

Run directory: config.yaml, train_names.txt, train_log.jsonl, checkpoint_last.pt (only while a run is unfinished),
model_final.pt, loss_curve.png, summary.json, eval/<split>.json, eval/<split>_vis/*.png; with eval.val_every > 0 also
val_log.jsonl + val_curve.png (a curve only: the reported model is always the final one).
"""

import json
import math
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler, get_worker_info

from src.data import (
    DATA_ROOT,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMREAD_LABEL,
    SHORT_SIDE,
    SOURCES,
    EgoHOSDataset,
    EvalTransform,
    TrainTransform,
    read_file_list,
    remap_label,
    source_of,
)
from src.evaluate import evaluate, evaluate_split
from src.labels import apply_label_map, label_lut, with_label_map
from src.model import add_hand_channel, build_model, freeze_encoder, param_groups, save_model, set_train_mode
from src.sanity import _to_png

VAL_CURVE_KEYS = ("iou", "miou", "miou_fg", "boundary_iou", "boundary_miou_fg", "eval_seconds")

# (height, width) after a 512 px shorter side: the widest stored aspect is EPIC 456x256 -> 912x512.
FIXED_PAD = (512, 912)


class FixedTransform:
    """No randomness (overfit test): shorter side to 512 px as in evaluation (mask nearest-neighbour), pad
    bottom/right to a common size (image 0, mask 255) so images batch together, ImageNet normalisation."""

    def __init__(self, pad_to: tuple[int, int] = FIXED_PAD):
        self.pad_to = pad_to
        self.aug = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=SHORT_SIDE,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST_EXACT,
                    area_for_downscale="image",
                ),
                A.PadIfNeeded(
                    min_height=pad_to[0],
                    min_width=pad_to[1],
                    position="top_left",
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=IGNORE_INDEX,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.aug(image=image, mask=mask)
        if out["mask"].shape != self.pad_to:
            raise ValueError(f"FixedTransform output {out['mask'].shape} does not match pad size {self.pad_to}")
        return out["image"], out["mask"]


class InfiniteSampler(Sampler):
    """Endless stream of shuffled indices (a fresh permutation per pass), so training counts iterations, not epochs."""

    def __init__(self, n: int, seed: int):
        self.n = n
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.n, generator=g).tolist()


def seed_worker(worker_id: int) -> None:
    """albumentations creates its generator once, in the main process; forked (or pickled) workers would all
    inherit the same state and draw identical augmentations. Reseed it from torch's per-worker seed."""
    info = get_worker_info()
    aug = getattr(getattr(info.dataset, "transform", None), "aug", None)
    if aug is not None:
        aug.set_random_seed(info.seed % 2**32)


def lr_factor(step: int, warmup_iters: int, max_iters: int, power: float) -> float:
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    return (1.0 - (step - warmup_iters) / max(1, max_iters - warmup_iters)) ** power


def select_subset(
    names: list[str], label_paths: dict, per_source: int, min_class_share: float, num_classes: int, seed: int,
    lut: np.ndarray | None = None,
) -> list[str]:
    """Up to `per_source` names per source, each with at least `min_class_share` of its pixels in every
    non-background class (after the run's label map). Deterministic for a given seed."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    picked: dict[str, list[str]] = {s: [] for s in SOURCES}
    for name in order:
        source = source_of(name)
        if len(picked[source]) >= per_source:
            continue
        mask = apply_label_map(remap_label(cv2.imread(str(label_paths[name]), IMREAD_LABEL)), lut)
        if min((mask == c).mean() for c in range(1, num_classes)) >= min_class_share:
            picked[source].append(name)
        if all(len(v) >= per_source for v in picked.values()):
            break
    return sorted(n for v in picked.values() for n in v)


def resolve_train_names(cfg: dict, repo_dir: str | Path) -> list[str]:
    d = cfg["data"]
    names = None
    if d["train_file_list"] is not None:
        path = Path(d["train_file_list"])
        names = read_file_list(path if path.is_absolute() else Path(repo_dir) / path)
    dataset = EgoHOSDataset("train", d["train_root"], file_list=names)
    names = dataset.names
    if d["subset"]:
        num_classes = len(cfg["model"]["class_names"])
        names = select_subset(
            names, dataset.labels, d["subset"]["per_source"], d["subset"]["min_class_share"],
            num_classes, cfg["seed"], label_lut(d["label_map"], num_classes),
        )
    return names


def prepare_run_dir(run_dir: Path, overwrite: bool, resume: bool) -> None:
    if resume:
        if not (run_dir / "checkpoint_last.pt").exists():
            raise FileNotFoundError(f"No {run_dir}/checkpoint_last.pt to resume from (finished runs delete it).")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists. Pass --overwrite to replace it or --resume to continue it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _save_checkpoint(path: Path, model, optimizer, scheduler, step: int, cfg: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "config": cfg},
        tmp,
    )
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def plot_loss_curve(log: list[dict], title: str) -> bytes:
    steps = [r["step"] for r in log]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [r["loss"] for r in log], color="tab:blue", label="train loss (mean per log window)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    for key, colour in (("lr_head", "tab:orange"), ("lr_encoder", "tab:green")):
        if log and log[0].get(key) is not None:
            ax2.plot(steps, [r[key] for r in log], color=colour, alpha=0.6, linestyle="--", label=key)
    ax2.set_ylabel("learning rate")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=10)
    return _to_png(fig)


def plot_val_curve(val_log: list[dict], title: str) -> bytes:
    steps = [r["step"] for r in val_log]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "predicted" in val_log[0]:
        for mode, colour in (("predicted", "tab:blue"), ("oracle", "tab:orange")):
            ax.plot(steps, [r[mode]["iou"]["interacting_object"] for r in val_log], "o-", color=colour, label=f"object IoU ({mode})")
            ax.plot(steps, [r[mode]["boundary_iou"]["interacting_object"] for r in val_log], "s--", color=colour,
                    alpha=0.6, label=f"object Boundary IoU ({mode})")
        ax.set(xlabel="iteration", ylabel="full val, composed masks", ylim=(0, 1), title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return _to_png(fig)
    classes = [c for c in val_log[0]["iou"] if c != "background"]
    for c, colour in zip(classes, ("tab:blue", "tab:orange")):
        ax.plot(steps, [r["iou"][c] for r in val_log], "o-", color=colour, label=f"IoU {c}")
        ax.plot(steps, [r["boundary_iou"][c] for r in val_log], "s--", color=colour, alpha=0.6, label=f"Boundary IoU {c}")
    if len(classes) > 1:
        ax.plot(steps, [r["miou_fg"] for r in val_log], "^-", color="black", label="mIoU_fg")
    ax.set_xlabel("iteration")
    ax.set_ylabel("val (full, original resolution)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)
    return _to_png(fig)


def val_point(model, dataset: EgoHOSDataset, cfg: dict, step: int, device, stage1=None) -> dict:
    if cfg.get("cascade", {}).get("enabled"):
        from src.cascade import evaluate_cascade

        metrics, _ = evaluate_cascade(stage1, model, dataset, cfg, device)
        return {"step": step, **{mode: {k: m[k] for k in VAL_CURVE_KEYS} for mode, m in metrics.items()}}
    metrics, _ = evaluate(
        model, dataset, cfg["model"]["class_names"], cfg["eval"]["boundary_dilation_ratio"],
        cfg["data"]["num_workers"], bf16=cfg["train"]["bf16"], device=device,
    )
    return {"step": step, **{k: metrics[k] for k in VAL_CURVE_KEYS}}


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def train(
    cfg: dict, run_dir: Path, train_names: list[str], device="cuda", commit=lambda: None, resume: bool = False,
    data_root: str = DATA_ROOT,
):
    """Returns (model, log records, summary). Writes train_log.jsonl, val_log.jsonl (if eval.val_every),
    checkpoint_last.pt and model_final.pt."""
    device = torch.device(device)
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    max_iters, log_every, ckpt_every = t["max_iters"], t["log_every"], t["ckpt_every"]
    val_every = cfg["eval"]["val_every"]
    tag = f"[{cfg['run_name']}]"
    lut = label_lut(d["label_map"], len(m["class_names"]))

    torch.manual_seed(cfg["seed"])  # the new decode head is initialised from the global RNG
    model = build_model(m["class_names"], m["pretrained"]).to(device)
    is_cascade = cfg.get("cascade", {}).get("enabled", False)
    stage1 = None
    if is_cascade:
        add_hand_channel(model)
        if val_every:
            from src.cascade import load_stage1

            stage1 = load_stage1(cfg["cascade"]["stage1_checkpoint"], device)
    if m["freeze_encoder"]:
        freeze_encoder(model)
    optimizer = torch.optim.AdamW(param_groups(model, t["lr"], t["head_lr_mult"], t["weight_decay"]), lr=t["lr"])
    scheduler = LambdaLR(optimizer, lambda s: lr_factor(s, t["warmup_iters"], max_iters, t["poly_power"]))

    start = 0
    log_path, val_path = run_dir / "train_log.jsonl", run_dir / "val_log.jsonl"
    log: list[dict] = []
    val_log: list[dict] = []
    if resume:
        ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["step"]
        log = [r for r in _read_jsonl(log_path) if r["step"] <= start]
        val_log = [r for r in _read_jsonl(val_path) if r["step"] <= start]
    log_path.write_text("".join(json.dumps(r) + "\n" for r in log))
    if val_every:
        val_path.write_text("".join(json.dumps(r) + "\n" for r in val_log))
        val_data = EgoHOSDataset("val", data_root, EvalTransform() if is_cascade else with_label_map(EvalTransform(), lut))

    seed = cfg["seed"] + start
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rgb_transform = TrainTransform() if d["augment"] else FixedTransform()
    if is_cascade:
        from src.cascade_data import CascadeTrainTransform

        transform = CascadeTrainTransform(rgb_transform, cfg["cascade"]["corruption"], seed)
    else:
        transform = with_label_map(rgb_transform, lut)
    dataset = EgoHOSDataset("train", d["train_root"], transform, file_list=train_names)
    loader = DataLoader(
        dataset,
        batch_size=t["batch_size"],
        sampler=InfiniteSampler(len(dataset), seed),
        num_workers=d["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        persistent_workers=d["num_workers"] > 0,
    )

    set_train_mode(model)
    batches = iter(loader)
    loss_sum, n_window, data_s, val_s = torch.zeros((), device=device), 0, 0.0, 0.0
    t_run = t_window = time.perf_counter()
    for step in range(start, max_iters):
        t_data = time.perf_counter()
        images, masks = next(batches)
        data_s += time.perf_counter() - t_data
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=t["bf16"]):
            logits = model(pixel_values=images).logits
        logits = F.interpolate(logits.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.detach()
        n_window += 1

        done = step + 1
        if done % log_every == 0 or done == max_iters:
            loss_mean = (loss_sum / n_window).item()
            if not math.isfinite(loss_mean):
                raise FloatingPointError(f"Loss is {loss_mean} at iteration {done}")
            elapsed = time.perf_counter() - t_window
            lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            record = {
                "step": done,
                "loss": round(loss_mean, 5),
                "lr_encoder": lrs.get("encoder_decay"),
                "lr_head": lrs.get("head_decay"),
                "imgs_per_s": round(n_window * t["batch_size"] / elapsed, 1),
                "data_wait_frac": round(data_s / elapsed, 3),
                "elapsed_s": round(time.perf_counter() - t_run, 1),
            }
            if device.type == "cuda":
                record["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            log.append(record)
            _append_jsonl(log_path, record)
            print(tag, json.dumps(record), flush=True)
            loss_sum.zero_()
            n_window, data_s, t_window = 0, 0.0, time.perf_counter()

        if val_every and done % val_every == 0 and done < max_iters:
            t_val = time.perf_counter()
            point = val_point(model, val_data, cfg, done, device, stage1)
            set_train_mode(model)
            val_log.append(point)
            _append_jsonl(val_path, point)
            brief = {mode: p["iou"] for mode, p in point.items() if mode != "step"} if is_cascade else {"iou": point["iou"], "miou_fg": point["miou_fg"]}
            print(tag, "val", json.dumps({"step": done, **brief}), flush=True)
            dt = time.perf_counter() - t_val
            val_s += dt
            t_window += dt  # keep the val pass out of the next window's img/s

        if ckpt_every and done % ckpt_every == 0 and done < max_iters:
            _save_checkpoint(run_dir / "checkpoint_last.pt", model, optimizer, scheduler, done, cfg)
            commit()

    train_seconds = time.perf_counter() - t_run
    save_model(run_dir / "model_final.pt", model, config=cfg, step=max_iters)
    (run_dir / "checkpoint_last.pt").unlink(missing_ok=True)
    commit()

    steady = log[1:] or log
    summary = {
        "n_train_images": len(train_names),
        "train_sources": dict(Counter(map(source_of, train_names))),
        "train_names": train_names if len(train_names) <= 20 else None,
        "steps": max_iters,
        "resumed_from": start or None,
        "train_seconds": round(train_seconds, 1),
        "val_curve_seconds": round(val_s, 1),
        "first_loss": log[0]["loss"],
        "last_loss": log[-1]["loss"],
        "imgs_per_s": round(float(np.mean([r["imgs_per_s"] for r in steady])), 1),
        "data_wait_frac": round(float(np.mean([r["data_wait_frac"] for r in steady])), 3),
        "peak_gpu_mem_gb": log[-1].get("gpu_mem_gb"),
        "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "n_trainable_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2),
    }
    return model, log, summary


def train_checks(log: list[dict]) -> dict[str, bool]:
    losses = [r["loss"] for r in log]
    return {
        "loss_finite": all(math.isfinite(v) for v in losses),
        "loss_decreased_by_half": len(losses) > 1 and losses[-1] <= 0.5 * losses[0],
    }


def run(
    cfg: dict, runs_dir: str | Path, repo_dir: str | Path, overwrite: bool = False, resume: bool = False,
    device="cuda", commit=lambda: None, data_root: str = DATA_ROOT,
) -> dict:
    """Train one config, then evaluate the final model on cfg.eval.splits. Returns summary, checks and the small
    output files (relative path -> bytes) for the local results/ copy."""
    run_dir = Path(runs_dir) / cfg["run_name"]
    prepare_run_dir(run_dir, overwrite, resume)
    if resume:
        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        from src.config import DEFAULTS, _merge

        if _merge(DEFAULTS, saved) != cfg:
            raise ValueError(f"Config differs from {run_dir}/config.yaml; --resume needs the same config.")
        train_names = read_file_list(run_dir / "train_names.txt")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        train_names = resolve_train_names(cfg, repo_dir)
        (run_dir / "train_names.txt").write_text("\n".join(train_names) + "\n")
        commit()

    model, log, train_summary = train(cfg, run_dir, train_names, device, commit, resume, data_root)
    (run_dir / "loss_curve.png").write_bytes(plot_loss_curve(log, f"{cfg['run_name']}: {len(train_names)} images"))

    checks = train_checks(log)
    evals, files = {}, {}
    is_cascade = cfg.get("cascade", {}).get("enabled", False)
    stage1 = None
    if is_cascade:
        from src.cascade import evaluate_cascade_split, load_stage1

        stage1 = load_stage1(cfg["cascade"]["stage1_checkpoint"], device)
    for split in cfg["eval"]["splits"]:
        if is_cascade:
            result = evaluate_cascade_split(stage1, model, cfg, split, run_dir, train_names, device, data_root)
            evals.update({f"{split}_{mode}": metrics for mode, metrics in result["metrics"].items()})
        else:
            result = evaluate_split(model, cfg, split, run_dir, train_names, device, data_root)
            evals[split] = result["metrics"]
        checks.update(result["checks"])
        files.update(result["files"])
        commit()

    names = ["config.yaml", "train_names.txt", "train_log.jsonl", "loss_curve.png", "summary.json"]
    if cfg["eval"]["val_every"]:
        val_path = run_dir / "val_log.jsonl"
        if "val" in evals:
            _append_jsonl(val_path, {"step": cfg["train"]["max_iters"], **{k: evals["val"][k] for k in VAL_CURVE_KEYS}})
        elif is_cascade and "val_predicted" in evals:
            _append_jsonl(val_path, {"step": cfg["train"]["max_iters"], **{
                mode: {k: evals[f"val_{mode}"][k] for k in VAL_CURVE_KEYS} for mode in ("predicted", "oracle")
            }})
        val_log = _read_jsonl(val_path)
        if val_log:
            (run_dir / "val_curve.png").write_bytes(plot_val_curve(val_log, f"{cfg['run_name']}: full val during training"))
            names += ["val_log.jsonl", "val_curve.png"]

    summary = {"run_name": cfg["run_name"], "run_dir": str(run_dir), "train": train_summary, "eval": evals, "checks": checks}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    commit()
    for name in names:
        files[name] = (run_dir / name).read_bytes()
    return {"summary": summary, "checks": checks, "files": files}
