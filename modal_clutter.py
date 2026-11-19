"""Isolated Phase-6 SAM analysis app. The user launches all remote work.

Long commands use `modal run --detach`; fetch is deliberately allowlisted.
A named Dict's atomic writer lease excludes concurrent launches, including
different ephemeral apps. Interrupted full launches retain the lease until recovery.
"""

import json
import os
from pathlib import Path

import modal

from src.clutter_config import SAM_REVISION, load_config
from src.clutter_lock import recover_writer, require_writer, writer_lease

GPU = os.environ.get("EGOHOS_GPU", "L40S")
volume = modal.Volume.from_name("egohos", create_if_missing=False)
leases = modal.Dict.from_name("egohos-clutter-writers", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0")
    .pip_install(
        "torch==2.14.1", "torchvision==0.29.2", "transformers==5.22.2",
        "albumentations==2.0.8", "opencv-python-headless==4.14.0.94", "numpy==2.3.5",
        "scipy==1.18.1", "pyyaml==6.0.3", "matplotlib==3.11.2", "tqdm==4.70.1", "gdown==6.5.1",
    )
    .pip_install(f"segment-anything @ https://github.com/facebookresearch/segment-anything/archive/{SAM_REVISION}.tar.gz",
                 extra_options="--no-deps")
    .add_local_python_source("src")
)
app = modal.App("egohos-clutter", image=image)


@app.function(cpu=8, memory=16384, timeout=3600, max_containers=1, volumes={"/data": volume})
def prepare_remote(cfg):
    from src.clutter_workflow import prepare_analysis
    with writer_lease(leases, cfg["analysis_name"], "prepare", modal.current_function_call_id()):
        volume.reload()
        manifest = prepare_analysis(cfg, volume.commit)
    return {"analysis_fingerprint": manifest["analysis_fingerprint"], "n_images": len(manifest["inventory"]),
            "sam_checkpoint": manifest["sam_checkpoint"], "pilot_images": manifest["pilot_images"]}


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=1800, max_containers=1, volumes={"/data": volume})
def pilot_remote(cfg):
    from src.clutter_workflow import pilot_analysis
    with writer_lease(leases, cfg["analysis_name"], "pilot", modal.current_function_call_id()):
        volume.reload()
        return pilot_analysis(cfg, volume.commit)


@app.function(cpu=1, memory=2048, timeout=600, volumes={"/data": volume})
def approve_remote(cfg):
    from src.clutter_workflow import approve_pilot
    with writer_lease(leases, cfg["analysis_name"], "approve", modal.current_function_call_id()):
        volume.reload()
        return approve_pilot(cfg, volume.commit)


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=7200, max_containers=4, volumes={"/data": volume})
def worker_remote(cfg, shard, shards, writer_token):
    from src.clutter_workflow import run_shard
    require_writer(leases, writer_token)
    volume.reload()
    return run_shard(cfg, shard, shards, volume.commit)


@app.function(cpu=2, memory=8192, timeout=86400, max_containers=1, volumes={"/data": volume})
def coordinate_remote(cfg, shards):
    with writer_lease(leases, cfg["analysis_name"], "full", modal.current_function_call_id()) as token:
        volume.reload()
        return _coordinate(cfg, shards, token)


def _coordinate(cfg, shards, writer_token):
    from src.clutter_sam import compatible_record, file_sha256, read_record, write_record
    from src.clutter_workflow import analysis_dir, load_manifest, merge_analysis, require_approval

    manifest = load_manifest(cfg)
    require_approval(cfg, manifest)
    root = analysis_dir(cfg)
    path = root / "execution.json"
    if path.exists():
        plan = read_record(path)
        if plan["shards"] != shards or plan["analysis_fingerprint"] != manifest["analysis_fingerprint"]:
            raise ValueError("Resume must retain the original shard count and analysis identity")
    else:
        plan = write_record(path, {"shards": shards, "analysis_fingerprint": manifest["analysis_fingerprint"],
                            "gpu_request": GPU, "worker_timeout_seconds": cfg["execution"]["worker_timeout_seconds"],
                            "execution_settings": cfg["execution"]})
        volume.commit()
    summary, _ = compatible_record(root / "summary.json", {"analysis_fingerprint": manifest["analysis_fingerprint"]})
    if summary is not None:
        measurements = root / "measurements.jsonl.gz"
        if not measurements.exists() or file_sha256(measurements) != summary["measurements_sha256"]:
            summary = merge_analysis(cfg, volume.commit)
            return {"state": "recovered merged measurements from worker records", "n_images": summary["n_images"]}
        return {"state": "already complete", "n_images": summary["n_images"]}
    calls = []
    for shard in range(shards):
        calls.append(worker_remote.with_options(timeout=cfg["execution"]["worker_timeout_seconds"]).spawn(cfg, shard, shards, writer_token))
    failures, completed = {}, []
    for shard, call in enumerate(calls):
        try:
            completed.append(call.get())
        except Exception as exc:
            failures[shard] = f"{type(exc).__name__}: {exc}"
    volume.reload()  # All worker writes are closed/committed; this coordinator owns no open files.
    if failures:
        raise RuntimeError(f"Some shards failed; completed work remains committed. Inspect status, stop the old app/workers, run recover --confirm-stopped, then rerun to resume. {failures}")
    summary = merge_analysis(cfg, volume.commit)
    return {"state": "complete", "n_images": summary["n_images"], "workers": completed}


@app.function(cpu=1, memory=4096, timeout=600, volumes={"/data": volume})
def inspect_remote(cfg, fetch=False):
    from src.clutter_workflow import small_output_files, status_analysis
    volume.reload()
    result = {"status": {**status_analysis(cfg), "writer_lease": leases.get("writer")}}
    if fetch:
        result["files"] = small_output_files(cfg)
    return result


@app.function(cpu=1, memory=1024, timeout=120)
def recover_remote(cfg, confirm_stopped):
    return recover_writer(leases, cfg["analysis_name"], confirm_stopped)


def _print(value):
    print(json.dumps(value, indent=2))


def _fetch(cfg):
    result = inspect_remote.remote(cfg, fetch=True)
    root = Path("results/phase6") / cfg["analysis_name"]
    for rel, content in result["files"].items():
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    print(f"Fetched {len(result['files'])} allowlisted files -> {root}/")
    _print(result["status"])


@app.local_entrypoint()
def prepare(config: str = "configs/phase6/clutter_sam_v1.yaml"):
    _print(prepare_remote.remote(load_config(config)))


@app.local_entrypoint()
def pilot(config: str = "configs/phase6/clutter_sam_v1.yaml"):
    cfg = load_config(config)
    _print(pilot_remote.remote(cfg))
    _fetch(cfg)


@app.local_entrypoint()
def approve_pilot(config: str = "configs/phase6/clutter_sam_v1.yaml"):
    """Run ONLY after personally accepting the candidate-only pilot previews."""
    _print(approve_remote.remote(load_config(config)))


@app.local_entrypoint()
def run(config: str = "configs/phase6/clutter_sam_v1.yaml", shards: int = 0):
    """Full analysis or safe resume. Requires the separately recorded pilot approval."""
    cfg = load_config(config)
    shards = shards or cfg["execution"]["shards"]
    if not 1 <= shards <= 4:
        raise ValueError("Use 1 to 4 shards")
    _print(coordinate_remote.remote(cfg, shards))
    _fetch(cfg)


@app.local_entrypoint()
def status(config: str = "configs/phase6/clutter_sam_v1.yaml"):
    _print(inspect_remote.remote(load_config(config))["status"])


@app.local_entrypoint()
def fetch(config: str = "configs/phase6/clutter_sam_v1.yaml"):
    _fetch(load_config(config))


@app.local_entrypoint()
def recover(config: str = "configs/phase6/clutter_sam_v1.yaml", confirm_stopped: bool = False):
    """Release a stale lease ONLY after stopping the old app and all its workers."""
    _print(recover_remote.remote(load_config(config), confirm_stopped))
