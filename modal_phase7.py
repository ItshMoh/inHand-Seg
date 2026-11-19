"""User-launched Phase-7 CPU survey and selected L40S rendering; no training/SAM run."""

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import modal

from src.clutter_sam import atomic_bytes, read_record
from src.phase7_analysis import sealed
from src.phase7_config import load_config

GPU = os.environ.get("EGOHOS_GPU", "L40S")
volume = modal.Volume.from_name("egohos", create_if_missing=False)
leases = modal.Dict.from_name("egohos-phase7-writers", create_if_missing=True)
# Reuse the working runtime, including SAM's installed metadata for strict Phase-6
# compatibility checks. No proposer, weight download or SAM inference is invoked.
if modal.is_local():
    # Only the launcher constructs the image. Containers hydrate the deployed
    # image and do not need to import/register the separate Phase-6 app.
    from modal_clutter import image as clutter_image

    image = (clutter_image
             .add_local_file("modal_phase7.py", "/root/modal_phase7.py")
             .add_local_file("scripts/phase7_report.py", "/root/scripts/phase7_report.py"))
else:
    image = None
app = modal.App("egohos-phase7", image=image)


@contextmanager
def writer(operation, analysis_name):
    token = uuid.uuid4().hex
    value = {"token": token, "operation": operation, "analysis_name": analysis_name,
             "function_call_id": modal.current_function_call_id()}
    if not leases.put("writer", value, skip_if_exists=True):
        raise RuntimeError(f"Another Phase-7 writer is active: {leases.get('writer')}. Check status; stop it before recovery.")
    try:
        yield
    finally:
        # There are no spawned children. A hard kill retains the lease for explicit recovery.
        if leases.get("writer", {}).get("token") == token:
            leases.pop("writer")


@app.function(cpu=8, memory=16384, timeout=1800, max_containers=1, volumes={"/data": volume})
def survey_remote(request):
    from src.phase7_workflow import survey
    with writer("survey", request["manifest"]["config"]["analysis_name"]):
        volume.reload()
        return survey(request, volume.commit)


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=1800, max_containers=1, volumes={"/data": volume})
def render_remote(request, selection, preview, approval):
    from src.phase7_workflow import render_selected
    with writer("preview" if preview else "gallery", request["manifest"]["config"]["analysis_name"]):
        volume.reload()
        return render_selected(request, selection, preview, approval, volume.commit)


@app.function(cpu=1, memory=2048, timeout=600, volumes={"/data": volume})
def inspect_remote(cfg, fetch=False):
    from src.phase7_workflow import fetch_files, status
    volume.reload()
    result = {"status": {**status(cfg), "writer_lease": leases.get("writer")}}
    if fetch:
        result["files"] = fetch_files(cfg)
        result["files_sha256"] = {p: hashlib.sha256(v).hexdigest() for p, v in result["files"].items()}
    return result


@app.function(cpu=1, memory=1024, timeout=120)
def recover_remote(cfg, confirm_stopped):
    if not confirm_stopped:
        raise ValueError("Stop the old Phase-7 app and its active function, then pass --confirm-stopped")
    value = leases.get("writer")
    if value is not None:
        if value["analysis_name"] != cfg["analysis_name"]:
            raise ValueError("Lease belongs to another analysis; use that config")
        leases.pop("writer")
    return {"state": "no outstanding lease" if value is None else "released after operator confirmed stopped"}


def _local_root(cfg):
    return Path("results/phase7") / cfg["analysis_name"]


def _request(cfg):
    request = read_record(_local_root(cfg) / "request.json")
    if request["manifest"]["config"] != cfg:
        raise ValueError("Config changed; rebuild local inputs first")
    return request


def _print(value):
    print(json.dumps(value, indent=2))


@app.local_entrypoint()
def survey(config: str = "configs/phase7/phase7_v1.yaml"):
    cfg = load_config(config)
    _print(survey_remote.with_options(timeout=cfg["execution"]["cpu_timeout_seconds"]).remote(_request(cfg)))


@app.local_entrypoint()
def preview(config: str = "configs/phase7/phase7_v1.yaml"):
    cfg = load_config(config)
    _print(render_remote.with_options(timeout=cfg["execution"]["gpu_timeout_seconds"]).remote(
        _request(cfg), read_record(_local_root(cfg) / "case_selection.json"), True, None))


@app.local_entrypoint()
def render(config: str = "configs/phase7/phase7_v1.yaml"):
    cfg = load_config(config)
    _print(render_remote.with_options(timeout=cfg["execution"]["gpu_timeout_seconds"]).remote(
        _request(cfg), read_record(_local_root(cfg) / "case_selection.json"), False,
        read_record(_local_root(cfg) / "preview_approval.json")))


@app.local_entrypoint()
def status(config: str = "configs/phase7/phase7_v1.yaml"):
    _print(inspect_remote.remote(load_config(config))["status"])


@app.local_entrypoint()
def fetch(config: str = "configs/phase7/phase7_v1.yaml"):
    cfg = load_config(config)
    result = inspect_remote.remote(cfg, fetch=True)
    root = _local_root(cfg)
    if "manifest.json" in result["files"] and (root / "manifest.json").exists():
        incoming = sealed(json.loads(result["files"]["manifest.json"]))
        if incoming != read_record(root / "manifest.json"):
            raise ValueError("Local directory belongs to another identity; preserve it before fetching")
    for rel, content in result["files"].items():
        p = Path(rel)
        if (p.is_absolute() or ".." in p.parts or not (root / p).resolve().is_relative_to(root.resolve())
                or hashlib.sha256(content).hexdigest() != result["files_sha256"][rel]):
            raise ValueError("Unsafe path or corrupt fetched output")
        atomic_bytes(root / p, content)
    print(f"Fetched {len(result['files'])} allowlisted files -> {root}")
    _print(result["status"])


@app.local_entrypoint()
def recover(config: str = "configs/phase7/phase7_v1.yaml", confirm_stopped: bool = False):
    _print(recover_remote.remote(load_config(config), confirm_stopped))
