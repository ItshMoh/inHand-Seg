"""Modal app: container image, shared Volume and remote functions for the project."""

import modal

APP_NAME = "egohos-segformer"
VOLUME_NAME = "egohos"
VOL_DIR = "/data"

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
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(gpu="L40S", cpu=2, timeout=10 * 60, volumes={VOL_DIR: volume})
def check_env() -> dict:
    import os

    import albumentations
    import cv2
    import numpy
    import torch
    import transformers
    from transformers import SegformerForSemanticSegmentation

    info = {
        "torch": torch.__version__,
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

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=3).cuda().eval()
    x = torch.randn(2, 3, 512, 512, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(pixel_values=x).logits
    info["segformer_b0_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    info["segformer_logits_shape"] = list(logits.shape)
    return info


@app.local_entrypoint()
def check():
    for key, value in check_env.remote().items():
        print(f"{key}: {value}")
