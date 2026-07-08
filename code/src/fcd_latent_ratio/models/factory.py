from __future__ import annotations

from .unet3d import CRILAttnUNet3D, CRILUNet3D, UNet3D


MODEL_REGISTRY = {
    "unet_e5": UNet3D,
    "cril_unet": CRILUNet3D,
    "cril_attn_unet": CRILAttnUNet3D,
    "cril_attention_unet": CRILAttnUNet3D,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
