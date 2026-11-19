"""Real-training corruption previews, generated on CPU before the user launches stage 2."""

import copy
import json
import random

import numpy as np

from src.cascade_data import CascadeTrainTransform
from src.data import EgoHOSDataset, TrainTransform, denormalize, remap_label, source_of
from src.sanity import _figure, _to_png, overlay_classes


def render_preview(name, inputs, target, clean, meta):
    rgb = denormalize(inputs[..., :3].transpose(2, 0, 1))
    hand = inputs[..., 3].astype(np.uint8)
    clean_hand = (clean == 1).astype(np.uint8)
    changes = rgb.copy()
    changes[(hand == 1) & (clean_hand == 0)] = (40, 220, 80)
    changes[(hand == 0) & (clean_hand == 1)] = (230, 25, 25)
    object_target = np.where(target == 1, 2, target).astype(np.uint8)
    fig, axes = _figure(5, *target.shape, col_width=3.5)
    fig.suptitle(f"{name}  {source_of(name)}  {json.dumps(meta)}", fontsize=8)
    panels = [rgb, overlay_classes(rgb, clean_hand), overlay_classes(rgb, hand), changes,
              overlay_classes(rgb, object_target)]
    titles = ["augmented RGB", "clean hand", "actual corrupted input", "added green / removed red", "clean binary loss target"]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=8)
    return _to_png(fig)


def build_preview(cfg, n_per_source=1, draws=4):
    dataset = EgoHOSDataset("train", cfg["data"]["train_root"])
    rng = random.Random(0)
    picked = []
    for source in ("epic", "thu", "own", "ego4d"):
        pool = [n for n in dataset.names if source_of(n) == source]
        picked.extend(rng.sample(pool, min(n_per_source, len(pool))))
    transform = CascadeTrainTransform(TrainTransform(), cfg["cascade"]["corruption"], seed=0)
    files, records = {}, []
    index = {n: i for i, n in enumerate(dataset.names)}
    for i, name in enumerate(picked):
        image, label = dataset.load(index[name])
        for draw in range(draws):
            inputs, target, clean, meta = transform.draw(image, remap_label(label))
            stem = f"random_{i}_{draw}_{name}"
            files[f"{stem}.png"] = render_preview(name, inputs, target, clean, meta)
            records.append({"name": name, "source": source_of(name), "draw": draw, **meta,
                            "input_values": np.unique(inputs[..., 3]).tolist(),
                            "target_values": np.unique(target).tolist()})

    # Clearly labelled stress examples guarantee the user sees every operation even
    # if the random draws happen not to contain a drop or a large kernel.
    stress_sample = None
    for name in picked:
        image, label = dataset.load(index[name])
        for stress_seed in range(7, 23):
            _, crop = TrainTransform(seed=stress_seed)(image, remap_label(label))
            if np.count_nonzero(crop == 1) >= 64:
                stress_sample = (name, image, label, stress_seed)
                break
        if stress_sample is not None:
            break
    if stress_sample is not None:
        name, image, label, stress_seed = stress_sample
        for operation in ("clean", "dilation15", "erosion15", "drop", "blobs"):
            recipe = copy.deepcopy(cfg["cascade"]["corruption"])
            recipe.update(morph_probs=[1, 0, 0], drop_prob=0, blob_prob=0)
            if operation in ("dilation15", "erosion15"):
                recipe["morph_probs"] = [0, 1, 0] if operation == "dilation15" else [0, 0, 1]
                recipe["kernel_sizes"] = [15]
            elif operation == "drop":
                recipe["drop_prob"] = 1
            elif operation == "blobs":
                recipe["blob_prob"] = 1
            forced = CascadeTrainTransform(TrainTransform(), recipe, seed=stress_seed)
            inputs, target, clean, meta = forced.draw(image, remap_label(label))
            files[f"stress_{operation}.png"] = render_preview(f"{name} (FORCED {operation})", inputs, target, clean, meta)
    summary = {"run_name": cfg["run_name"], "seed": 0, "resolution": "after geometry, 512 crop",
               "corruption": cfg["cascade"]["corruption"], "random_draws": records,
               "stress_panels": "Forced examples on one identical nonempty-hand crop, not training frequencies",
               "stress_sample": {"name": stress_sample[0], "seed": stress_sample[3]} if stress_sample else None}
    files["summary.json"] = json.dumps(summary, indent=2).encode()
    return {"files": files, "summary": summary}
