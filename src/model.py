"""SegFormer builder: pretrained MiT encoder + new decode head, encoder freezing, optimizer groups, checkpoints."""

from pathlib import Path

import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def build_model(class_names: list[str], pretrained: str = "nvidia/mit-b0") -> SegformerForSemanticSegmentation:
    """ImageNet-pretrained MiT encoder from `pretrained`; the decode head is new (randomly initialised),
    with one output per class name. Logits come out at 1/4 of the input resolution."""
    id2label = dict(enumerate(class_names))
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained, id2label=id2label, label2id={v: k for k, v in id2label.items()}
    )
    if model.decode_head.classifier.out_channels != len(class_names):
        raise ValueError(f"Head has {model.decode_head.classifier.out_channels} outputs, expected {len(class_names)}")
    return model


def first_patch_projection(model):
    """Return (embedding, projection), supporting the installed and older HF layouts."""
    backbone = model.segformer
    if hasattr(backbone, "stages"):
        embedding = backbone.stages[0].patch_embeddings
    else:
        embedding = backbone.encoder.patch_embeddings[0]
    if not isinstance(embedding.proj, torch.nn.Conv2d):
        raise TypeError("First patch projection must be Conv2d")
    return embedding, embedding.proj


def add_hand_channel(model) -> None:
    """Expand only the loaded RGB projection. Preserve RGB weights/bias; zero the new input."""
    embedding, old = first_patch_projection(model)
    if old.in_channels != 3 or old.groups != 1:
        raise ValueError(f"Expected a dense RGB projection, got {old}")
    new = torch.nn.Conv2d(
        4, old.out_channels, old.kernel_size, old.stride, old.padding, old.dilation,
        old.groups, old.bias is not None, old.padding_mode,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.train(old.training)
    embedding.proj = new
    model.config.num_channels = 4


def freeze_encoder(model: SegformerForSemanticSegmentation) -> None:
    model.segformer.requires_grad_(False)
    model.segformer.eval()


def encoder_frozen(model: SegformerForSemanticSegmentation) -> bool:
    return not any(p.requires_grad for p in model.segformer.parameters())


def set_train_mode(model: SegformerForSemanticSegmentation) -> None:
    """model.train(), except a frozen encoder stays in eval mode (no drop-path/dropout on fixed features)."""
    model.train()
    if encoder_frozen(model):
        model.segformer.eval()


def param_groups(model: SegformerForSemanticSegmentation, lr: float, head_lr_mult: float, weight_decay: float) -> list[dict]:
    """Encoder at `lr`, decode head at `lr * head_lr_mult`; no weight decay on 1-D params (norms, biases)."""
    groups: dict[str, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if name.startswith("decode_head.") else "encoder"
        decay = p.ndim > 1
        key = f"{part}_{'decay' if decay else 'no_decay'}"
        if key not in groups:
            groups[key] = {
                "name": key,
                "params": [],
                "lr": lr * (head_lr_mult if part == "head" else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def save_model(path: str | Path, model: SegformerForSemanticSegmentation, **extra) -> None:
    """Weights + the architecture config, so the model can be rebuilt without the hub."""
    torch.save({"model": model.state_dict(), "model_config": model.config.to_dict(), **extra}, path)


def load_model(path: str | Path, map_location="cpu") -> tuple[SegformerForSemanticSegmentation, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model, ckpt
