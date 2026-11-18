"""CPU-only Modal app for dataset jobs (download, inspection, preprocessing).

Kept separate from modal_app.py so data jobs never register GPU functions.
"""

import modal

APP_NAME = "egohos-data"
VOLUME_NAME = "egohos"
VOL_DIR = "/data"
DATASET_DIR = f"{VOL_DIR}/egohos"
PRERESIZED_DIR = f"{VOL_DIR}/egohos_512"
SANITY_DIR = f"{VOL_DIR}/sanity"
DEV_SUBSET_FILE = "configs/dev_subset_500.txt"
RAW_ZIP = f"{VOL_DIR}/raw/egohos_dataset.zip"
DATASET_URL = "https://drive.google.com/file/d/1sk0TVEhZESNF67OW3fz9D5coqpIWkwuK/view?usp=sharing"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libglib2.0-0", "unzip")
    .pip_install(
        "numpy==2.3.5",
        "pillow",
        "opencv-python-headless==4.14.0.94",
        "albumentations==2.0.8",
        "matplotlib==3.11.2",
        "pyyaml==6.0.3",
        "tqdm==4.70.1",
        "gdown==6.5.1",
    )
    .add_local_python_source("src")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(cpu=4, memory=8192, timeout=3 * 60 * 60, volumes={VOL_DIR: volume})
def download_dataset() -> str:
    import shutil
    import subprocess
    from pathlib import Path

    dataset_dir = Path(DATASET_DIR)
    if (dataset_dir / "train").is_dir():
        return f"Dataset already present at {dataset_dir}, nothing to do."

    zip_path = Path(RAW_ZIP)
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        part = zip_path.with_name(zip_path.name + ".part")
        result = subprocess.run(["gdown", "--fuzzy", DATASET_URL, "-O", str(part)])
        if result.returncode != 0 or not part.exists():
            raise RuntimeError(
                "gdown failed (often a Google Drive quota error). Fallback: download the zip in a browser, then "
                "run `modal volume put egohos <local zip path> raw/egohos_dataset.zip` and rerun this job."
            )
        part.rename(zip_path)
        volume.commit()
    zip_gb = zip_path.stat().st_size / 1e9

    extract_dir = Path(VOL_DIR) / "_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(extract_dir)], check=True)

    roots = sorted(
        p.parent
        for p in extract_dir.rglob("train")
        if p.is_dir() and (p.parent / "val").is_dir() and "__MACOSX" not in p.parts
    )
    if not roots:
        top = sorted(str(p.relative_to(extract_dir)) for p in extract_dir.iterdir())
        raise RuntimeError(f"No folder with train/ and val/ found in the zip. Top level: {top}")
    shutil.move(str(roots[0]), str(dataset_dir))
    shutil.rmtree(extract_dir, ignore_errors=True)
    volume.commit()

    splits = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    return f"Zip {zip_gb:.2f} GB extracted to {dataset_dir}. Splits found: {splits}"


@app.function(cpu=8, memory=16384, timeout=60 * 60, volumes={VOL_DIR: volume})
def dataset_stats() -> dict:
    from pathlib import Path

    from src.dataset_stats import inspect_dataset

    if not Path(DATASET_DIR, "train").is_dir():
        raise RuntimeError(f"{DATASET_DIR} not found. Run `modal run modal_data.py::download` first.")
    return inspect_dataset(DATASET_DIR, workers=8)


@app.function(cpu=8, memory=16384, timeout=60 * 60, volumes={VOL_DIR: volume})
def preresize_train() -> dict:
    from src.preresize import preresize_split

    result = preresize_split(DATASET_DIR, PRERESIZED_DIR, "train", workers=8)
    volume.commit()
    return result


@app.function(cpu=4, memory=16384, timeout=30 * 60, volumes={VOL_DIR: volume})
def render_sanity() -> dict:
    import shutil
    from pathlib import Path

    from src.sanity import build_sanity

    if not Path(PRERESIZED_DIR, "train").is_dir():
        raise RuntimeError(f"{PRERESIZED_DIR} not found. Run `modal run modal_data.py::preresize` first.")
    result = build_sanity(DATASET_DIR, PRERESIZED_DIR)

    out_dir = Path(SANITY_DIR)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for name, data in result["files"].items():
        (out_dir / name).write_bytes(data)
    volume.commit()
    return result


@app.function(cpu=1, timeout=10 * 60, volumes={VOL_DIR: volume})
def train_names() -> list[str]:
    from pathlib import Path

    return sorted(p.stem for p in Path(DATASET_DIR, "train", "image").iterdir() if p.suffix.lower() == ".jpg")


@app.local_entrypoint()
def download():
    print(download_dataset.remote())


@app.local_entrypoint()
def stats():
    import json
    from pathlib import Path

    result = dataset_stats.remote()
    out = Path("results/dataset_stats.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"{'split':<16}{'paired':>8}{'expected':>10}{'contact':>9}  label values")
    for split, s in result["splits"].items():
        print(f"{split:<16}{s['n_paired']:>8}{s['expected']:>10}{s['n_contact']:>9}  {s['label_values']}")
    print()
    for name, per_split in result["checks"].items():
        status = "PASS" if all(per_split.values()) else "FAIL"
        print(f"{status}  {name}: {per_split}")
    print(f"\nFull stats written to {out}")


def _print_checks(checks: dict) -> None:
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")


@app.local_entrypoint()
def preresize():
    import json
    from pathlib import Path

    result = preresize_train.remote()
    out = Path("results/preresize_train.json")
    out.write_text(json.dumps(result, indent=2))

    print(f"source pairs: {result['n_source_pairs']}   actions: {result['actions']}")
    print(f"output: {result['n_out_images']} images, {result['n_out_labels']} labels, {result['n_out_pairs']} pairs in {result['dst']}")
    print(f"max shorter side: {result['max_shorter_side']}   label values: {result['label_values']}   remapped: {result['remapped_values']}")
    print(f"output sizes: {result['out_sizes_top']}\n")
    _print_checks(result["checks"])
    print(f"\nWritten to {out}")


@app.local_entrypoint()
def sanity():
    import json
    from pathlib import Path

    result = render_sanity.remote()
    out_dir = Path("results/sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    for name, data in result["files"].items():
        (out_dir / name).write_bytes(data)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2))

    summary = result["summary"]
    print(f"{len(result['files'])} panels written to {out_dir}/ (also on the Volume at {SANITY_DIR})")
    print(f"  label panels: {len(summary['label_panels'])}, aug panels: {len(summary['aug_panels'])}, eval panels: {len(summary['eval_panels'])}")
    for p in summary["eval_panels"]:
        print(f"  eval {p['name']}: original {p['original_size']} -> input {p['eval_input_size']}")
    print(f"  remapped values seen: {summary['remapped_values_seen']}\n")
    _print_checks(summary["checks"])


@app.local_entrypoint()
def dev_subset():
    from collections import Counter
    from pathlib import Path

    from src.data import SOURCES, make_dev_subset, source_of

    names = train_names.remote()
    subset = make_dev_subset(names, n=500, seed=0)
    out = Path(DEV_SUBSET_FILE)
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(subset) + "\n")

    full, dev = Counter(map(source_of, names)), Counter(map(source_of, subset))
    print(f"{'source':<8}{'train':>8}{'share':>8}{'dev':>6}{'share':>8}")
    for s in SOURCES:
        print(f"{s:<8}{full[s]:>8}{full[s] / len(names):>8.1%}{dev[s]:>6}{dev[s] / len(subset):>8.1%}")
    print(f"{'total':<8}{len(names):>8}{'':>8}{len(subset):>6}\n")
    _print_checks({
        "500_names": len(subset) == 500,
        "all_unique": len(set(subset)) == len(subset),
        "all_in_train": set(subset) <= set(names),
    })
    print(f"\nWritten to {out}")
