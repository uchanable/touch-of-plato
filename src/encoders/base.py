"""Encoder abstractions for Tab.~``tab:encoders``.

Each `load_*` function in this package returns a `LoadedEncoder`
tuple, giving the experiment runners a uniform interface:

    (model, preprocess, feature_dim, modality, name)

where `model(preprocess(raw_input)) -> Tensor[B, feature_dim]`.

See §``sec:encoders`` for the loader-row binding (Tab.~``tab:encoders``).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, NamedTuple, Literal
import torch.nn as nn

Modality = Literal["vision", "language", "tactile"]

CKPT_ROOT = os.environ.get(
    "PLATONIC_TOUCH_CKPT_ROOT",
    str(Path(__file__).resolve().parents[2] / "checkpoints"),
)


class LoadedEncoder(NamedTuple):
    """Uniform return type for all loader functions."""
    model: nn.Module
    preprocess: Callable
    feature_dim: int
    modality: Modality
    name: str
