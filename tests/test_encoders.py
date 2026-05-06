"""Smoke tests for encoder loaders (Paper Table 1).

Verifies each loader returns Tensor(1, feature_dim) on a dummy input.
All 8 loaders (Paper Table 1 rows 1-6, with rows 2 and 3 providing both
vision and text sides) are expected to work as of initial setup.
"""
import gc
import pytest
import torch
import numpy as np
from PIL import Image
from src.encoders import get_encoder, list_encoders


def _dummy_image():
    return Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))


def _dummy_text():
    return "A soft rubber object with a smooth surface."


ALL_ENCODERS = [
    "dinov2_small",
    "dinov2_base",
    "dinov2_large",
    "clip_l_vision",
    "clip_l_text",
    "siglip_base_vision",
    "siglip_base_text",
    "mpnet",
    "sparsh_dino_base",
    "sparsh_ijepa_base",
]


@pytest.mark.parametrize("name", ALL_ENCODERS)
def test_encoder_forward_shape(name):
    """Each encoder returns Tensor(1, feature_dim) from a dummy input."""
    enc = get_encoder(name)
    with torch.no_grad():
        if enc.modality in ("vision", "tactile"):
            inp = enc.preprocess(_dummy_image())
            out = enc.model(inp)
        else:
            batch = enc.preprocess([_dummy_text()])
            out = enc.model(batch)
    assert tuple(out.shape) == (1, enc.feature_dim), (
        f"{name} returned shape {tuple(out.shape)}, expected (1, {enc.feature_dim})"
    )
    gc.collect()


def test_list_encoders_count():
    """Paper Table 1 exposes 8 loader entries (rows 2-3 have v/t, rows 5-6 have 2 Sparsh)."""
    assert len(list_encoders()) == 10


def test_sparsh_tactile_input_is_6channel():
    """Sparsh uses 6-channel input (same-frame duplication of 3-channel RGB).

    See the paper's Limitations section.
    """
    enc = get_encoder("sparsh_dino_base")
    inp = enc.preprocess(_dummy_image())
    assert tuple(inp.shape) == (1, 6, 224, 224), f"Expected (1, 6, 224, 224), got {tuple(inp.shape)}"
