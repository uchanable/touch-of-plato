"""Encoder loader registry for Table 1 of the paper.

Public API:
    from src.encoders import get_encoder, list_encoders

    enc = get_encoder("dinov2_base")
    tensor_out = enc.model(enc.preprocess(pil_image))  # (1, 768)

See paper §3.3 (Table 1) for the paper-side binding of each loader.
"""
from __future__ import annotations
from typing import Callable, Dict
from .base import LoadedEncoder
from .vision import (
    load_dinov2_small,
    load_dinov2_base,
    load_dinov2_large,
    load_clip_l_vision,
    load_siglip_base_vision,
)
from .language import load_clip_l_text, load_siglip_base_text, load_mpnet
from .tactile import load_sparsh_dino_base, load_sparsh_ijepa_base
from .anytouch import load_anytouch
from .tvl_vitb import load_tvl_vitb

_REGISTRY: Dict[str, Callable[[], LoadedEncoder]] = {
    "dinov2_small": load_dinov2_small,
    "dinov2_base": load_dinov2_base,
    "dinov2_large": load_dinov2_large,
    "clip_l_vision": load_clip_l_vision,
    "clip_l_text": load_clip_l_text,
    "siglip_base_vision": load_siglip_base_vision,
    "siglip_base_text": load_siglip_base_text,
    "mpnet": load_mpnet,
    "sparsh_dino_base": load_sparsh_dino_base,
    "sparsh_ijepa_base": load_sparsh_ijepa_base,
    "anytouch": load_anytouch,
    "tvl_vitb": load_tvl_vitb,
}


def get_encoder(name: str) -> LoadedEncoder:
    """String-dispatched encoder loader keyed by paper Table 1 rows."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_encoders() -> list[str]:
    """List all available encoder names."""
    return sorted(_REGISTRY.keys())


__all__ = ["get_encoder", "list_encoders", "LoadedEncoder"]
