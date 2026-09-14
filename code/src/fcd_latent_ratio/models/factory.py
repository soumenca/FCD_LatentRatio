from __future__ import annotations

from .unet3d import AttentionUNet3D, CRILUNet3D, UNet3D


MODEL_REGISTRY = {
    "unet_e5": UNet3D,
    "attention_unet": AttentionUNet3D,
    "cril_unet": CRILUNet3D,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
