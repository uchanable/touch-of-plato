"""Type aliases for alignment metric inputs.

Corresponds to Section 3.1 "Preliminaries" of the paper.
"""
from __future__ import annotations
from typing import Union
import numpy as np
import torch

# $\mathbf{Z}_X \in \mathbb{R}^{N \times d_X}$ in the paper.
# Accepts torch.Tensor or numpy.ndarray; functions internally cast to torch.
FeatureMatrix = Union[torch.Tensor, np.ndarray]


def _as_tensor(z: FeatureMatrix) -> torch.Tensor:
    """Cast to float32 CPU torch.Tensor with shape (N, d)."""
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    z = z.float().cpu()
    assert z.ndim == 2, f"Expected 2D feature matrix, got shape {tuple(z.shape)}"
    return z
