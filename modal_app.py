"""Modal app: container image, shared Volume and remote functions for the project."""

import os

import modal

APP_NAME = "egohos-segformer"
VOLUME_NAME = "egohos"
VOL_DIR = "/data"
RUNS_DIR = f"{VOL_DIR}/runs"
REPO_DIR = "/root"  # configs/ is added here; relative paths in configs resolve against it
LOCAL_RUNS_DIR = "results/runs"
GPU = os.environ.get("EGOHOS_GPU", "L40S")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "unzip")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.2",
        "transformers==5.22.2",
        "albumentations==2.0.8",
        "opencv-python-headless==4.14.0.94",
        "numpy==2.3.5",
        "scipy==1.18.1",
        "pyyaml==6.0.3",
        "matplotlib==3.11.2",
        "tqdm==4.70.1",
        "gdown==6.5.1",
    )
    .add_local_python_source("src")
    .add_local_dir("configs", remote_path=f"{REPO_DIR}/configs")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(gpu=GPU, cpu=2, timeout=10 * 60, volumes={VOL_DIR: volume})
def check_env() -> dict:
    import albumentations
    import cv2
    import numpy
    import torch
    import transformers
    from transformers import SegformerForSemanticSegmentation

    info = {
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "transformers": transformers.__version__,
        "albumentations": albumentations.__version__,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(),
        "volume_contents": sorted(os.listdir(VOL_DIR)),
    }

    id2label = {0: "background", 1: "hand", 2: "interacting_object"}
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", id2label=id2label, label2id={v: k for k, v in id2label.items()}
    ).cuda().eval()
    x = torch.randn(2, 3, 512, 512, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(pixel_values=x).logits
    info["segformer_b0_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    info["segformer_logits_shape"] = list(logits.shape)
    return info


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=2 * 60 * 60, volumes={VOL_DIR: volume})
def train_run(cfg: dict, overwrite: bool = False, resume: bool = False) -> dict:
    from src.train import run

    return run(cfg, RUNS_DIR, REPO_DIR, overwrite=overwrite, resume=resume, commit=volume.commit)


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=60 * 60, volumes={VOL_DIR: volume})
def eval_run(run_name: str, split: str) -> dict:
    from src.evaluate import evaluate_run

    result = evaluate_run(f"{RUNS_DIR}/{run_name}", split)
    volume.commit()
    return result


@app.function(cpu=1, memory=2048, timeout=10 * 60, volumes={VOL_DIR: volume})
def fetch_run(run_name: str) -> dict:
    """Small files of a run (everything except checkpoints), e.g. after the local client disconnected."""
    from pathlib import Path

    run_dir = Path(RUNS_DIR) / run_name
    if not run_dir.is_dir():
        return {"exists": False, "finished": False, "has_checkpoint": False, "files": {}}
    files = {
        str(p.relative_to(run_dir)): p.read_bytes()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.suffix not in (".pt", ".tmp")
    }
    return {
        "exists": True,
        "finished": (run_dir / "summary.json").exists(),
        "has_checkpoint": (run_dir / "checkpoint_last.pt").exists(),
        "files": files,
    }


@app.function(cpu=2, memory=4096, timeout=10 * 60, volumes={VOL_DIR: volume})
def cascade_preview_run(cfg: dict) -> dict:
    from pathlib import Path
    from src.cascade_preview import build_preview

    result = build_preview(cfg)
    out_dir = Path(VOL_DIR) / "cascade_previews" / cfg["run_name"]
    for rel, data in result["files"].items():
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_bytes(data)
    volume.commit()
    return result


@app.local_entrypoint()
def cascade_preview(config: str = "configs/b0_cascade_stage2.yaml"):
    from pathlib import Path
    from src.config import load_config

    cfg = load_config(config)
    if not cfg["cascade"]["enabled"]:
        raise ValueError("cascade_preview needs a cascade config")
    result = cascade_preview_run.remote(cfg)
    out_dir = Path("results/cascade_previews") / cfg["run_name"]
    _write_files(out_dir, result["files"])
    print(f"{len(result['files']) - 1} corruption panels -> {out_dir}/")
    print("Inspect random panels plus labelled stress panels; added pixels are green, removed pixels red.")
    print("The binary loss target must remain clean. Stress panels deliberately force each operation.")


@app.local_entrypoint()
def check():
    for key, value in check_env.remote().items():
        print(f"{key}: {value}")


def _write_files(out_dir, files: dict) -> None:
    for rel, data in files.items():
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _fmt(value) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def _print_eval(m: dict) -> None:
    fg = m["class_names"][1:]
    mode = f" ({m['hand_mode']} hands)" if "hand_mode" in m else ""
    print(f"{m['split']}{mode}: {m['n_images']} images, scored at original resolution in {m['eval_seconds']} s")
    print("  IoU          " + "   ".join(f"{c} {_fmt(v)}" for c, v in m["iou"].items()))
    print(f"  mIoU {_fmt(m['miou'])}   mIoU_fg {_fmt(m['miou_fg'])}   pixel acc {_fmt(m['pixel_accuracy'])}")
    print("  Boundary IoU " + "   ".join(f"{c} {_fmt(m['boundary_iou'][c])}" for c in fg)
          + f"   mean {_fmt(m['boundary_miou_fg'])}   (d = {m['boundary_dilation_ratio']:.0%} of image diagonal)")


def _print_checks(checks: dict) -> None:
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")


def _replace_local_run(run_name: str, files: dict):
    import shutil
    from pathlib import Path

    out_dir = Path(LOCAL_RUNS_DIR) / run_name
    shutil.rmtree(out_dir, ignore_errors=True)
    _write_files(out_dir, files)
    return out_dir


def _print_val_curve(raw: bytes) -> None:
    import json

    print("  val curve (full val, no checkpoint selection):")
    for line in raw.decode().splitlines():
        r = json.loads(line)
        if "predicted" in r:
            scores = "   ".join(f"object ({mode}) {_fmt(r[mode]['iou']['interacting_object'])}" for mode in ("predicted", "oracle"))
            print(f"    step {r['step']:>6}   IoU {scores}")
            continue
        ious = "   ".join(f"{c} {_fmt(v)}" for c, v in r["iou"].items() if c != "background")
        print(f"    step {r['step']:>6}   IoU {ious}   mIoU_fg {_fmt(r['miou_fg'])}")


def _report(run_name: str, summary: dict, checks: dict, files: dict, out_dir) -> None:
    t = summary["train"]
    print(f"\ntrain: {t['steps']} iterations in {t['train_seconds']} s on {t['n_train_images']} images {t['train_sources']}")
    if t.get("val_curve_seconds"):
        print(f"  (of which {t['val_curve_seconds']} s for the val curve)")
    print(f"  loss {t['first_loss']:.4f} (first log window) -> {t['last_loss']:.4f} (last)")
    print(f"  {t['imgs_per_s']} img/s, data wait {t['data_wait_frac']:.0%} of step time, peak GPU memory {t['peak_gpu_mem_gb']} GB")
    print(f"  trainable parameters {t['n_trainable_M']}M of {t['n_params_M']}M")
    if t["train_names"]:
        print(f"  images: {t['train_names']}")
    if "val_log.jsonl" in files:
        _print_val_curve(files["val_log.jsonl"])
    for metrics in summary["eval"].values():
        print()
        _print_eval(metrics)
    print()
    _print_checks(checks)
    print(f"\nOutputs: {out_dir}/ (loss_curve.png, train_log.jsonl, eval/) and {RUNS_DIR}/{run_name} on the Volume")


@app.local_entrypoint()
def train(config: str, overwrite: bool = False, resume: bool = False):
    from src.config import load_config

    cfg = load_config(config)
    print(f"run {cfg['run_name']} on {GPU} -> {RUNS_DIR}/{cfg['run_name']}")
    result = train_run.remote(cfg, overwrite, resume)
    out_dir = _replace_local_run(cfg["run_name"], result["files"])
    _report(cfg["run_name"], result["summary"], result["checks"], result["files"], out_dir)


@app.local_entrypoint()
def train_many(configs: str, overwrite: bool = False, resume: bool = False):
    """Several configs in parallel (comma-separated), one GPU container each. Launch with `modal run --detach` so the
    runs keep going if this client disconnects; then `fetch` copies their outputs."""
    from src.config import load_config

    cfgs = [load_config(c.strip()) for c in configs.split(",") if c.strip()]
    names = [c["run_name"] for c in cfgs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate run names: {names}")
    calls = []
    for cfg in cfgs:
        calls.append(train_run.spawn(cfg, overwrite, resume))
        print(f"spawned {cfg['run_name']} on {GPU} -> {RUNS_DIR}/{cfg['run_name']}")

    status = {}
    for name, call in zip(names, calls):
        try:
            result = call.get()
        except Exception as e:  # one failed run must not hide the others
            status[name] = f"FAILED ({type(e).__name__}: {e})"
            print(f"\n===== {name}: FAILED: {type(e).__name__}: {e}")
            continue
        print(f"\n===== {name}")
        out_dir = _replace_local_run(name, result["files"])
        _report(name, result["summary"], result["checks"], result["files"], out_dir)
        status[name] = "ok" if all(result["checks"].values()) else "finished, some checks FAILED"

    print("\n===== all runs")
    for name, s in status.items():
        print(f"  {name}: {s}")
    table_script = "phase5_table.py" if all(c.get("cascade", {}).get("enabled") for c in cfgs) else "phase4_table.py"
    print(f"\nTable: .venv/bin/python scripts/{table_script}")


@app.local_entrypoint()
def fetch(run_names: str):
    """Copy the small files of finished (or still running) runs from the Volume into results/runs/<run_name>/."""
    names = [n.strip() for n in run_names.split(",") if n.strip()]
    for name, res in zip(names, fetch_run.map(names)):
        if not res["exists"]:
            print(f"{name}: not on the Volume")
            continue
        out_dir = _replace_local_run(name, res["files"])
        if res["finished"]:
            state = "finished"
        elif res["has_checkpoint"]:
            state = "unfinished (checkpoint_last.pt present: still running, or rerun `train --config ... --resume`)"
        else:
            state = "unfinished (no checkpoint yet)"
        print(f"{name}: {state}; {len(res['files'])} files -> {out_dir}/")


@app.local_entrypoint()
def evaluate(run_name: str, split: str = "val"):
    from pathlib import Path

    result = eval_run.remote(run_name, split)
    out_dir = Path(LOCAL_RUNS_DIR) / run_name
    _write_files(out_dir, result["files"])
    if "predicted" in result["metrics"]:
        for metrics in result["metrics"].values():
            _print_eval(metrics)
    else:
        _print_eval(result["metrics"])
    print()
    _print_checks(result["checks"])
    print(f"\nOutputs: {out_dir}/eval/")
