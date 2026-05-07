"""Null-calibrated alignment (Eq. 3 of the paper).

Implements §``sec:metrics`` (M3):

    A_null(Z_X, Z_Y) = (A_raw(Z_X, Z_Y) - mu_null) / sigma_null

where mu_null, sigma_null are estimated from B row-permutations of Z_Y.
Following Gröger et al. (2026, "Aristotelian"), this quantifies
alignment beyond random row-pairings.
"""
from __future__ import annotations
from typing import Callable
import torch
import numpy as np
from .types import FeatureMatrix, _as_tensor


def null_calibrated_alignment(
    feat_x: FeatureMatrix,
    feat_y: FeatureMatrix,
    base_metric: Callable[[torch.Tensor, torch.Tensor], float],
    n_perms: int = 100,
    seed: int = 0,
    return_raw: bool = False,
) -> float | tuple[float, float, float, float]:
    """Implements Eq. (3) of the paper: null-calibrated z-score of base_metric.

    Args:
        feat_x: (N, d_x).
        feat_y: (N, d_y), paired to feat_x.
        base_metric: callable (Z_X, Z_Y) -> float, e.g. mutual_knn_alignment.
        n_perms: number of row-permutations of Z_Y for the null distribution.
        seed: RNG seed for reproducibility.
        return_raw: if True, also return (raw, mu_null, sigma_null).

    Returns:
        Scalar z-score if return_raw=False (Eq. 3).
        (z_score, raw, mu_null, sigma_null) if return_raw=True.
    """
    zx = _as_tensor(feat_x)
    zy = _as_tensor(feat_y)
    assert zx.shape[0] == zy.shape[0]
    N = zx.shape[0]

    raw = base_metric(zx, zy)

    rng = np.random.default_rng(seed)
    null_scores = []
    for _ in range(n_perms):
        perm = rng.permutation(N)
        zy_perm = zy[perm]
        null_scores.append(base_metric(zx, zy_perm))
    null_arr = np.array(null_scores)
    mu = float(null_arr.mean())
    sigma = float(null_arr.std(ddof=1)) if n_perms > 1 else 1.0
    if sigma <= 1e-12:
        sigma = 1e-12
    z = (raw - mu) / sigma
    if return_raw:
        return z, raw, mu, sigma
    return z
