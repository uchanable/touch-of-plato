"""Unbiased CKA cross-check via platonic-rep (M4 of the paper).

Thin wrapper around `metrics.AlignmentMetrics.unbiased_cka` from
minyoungg/platonic-rep. Used to cross-validate our own implementation
in `debiased_cka.py` (Eq. 2).
"""
from __future__ import annotations
import torch
from .types import FeatureMatrix, _as_tensor

try:
    from metrics import AlignmentMetrics  # from platonic-rep clone via .pth
    _HAS_PLATONIC_REP = True
except ImportError:
    _HAS_PLATONIC_REP = False


def unbiased_cka_alignment(feat_x: FeatureMatrix, feat_y: FeatureMatrix) -> float:
    """Wraps platonic-rep's unbiased_cka (M4 cross-check of Eq. 2).

    platonic-rep's AlignmentMetrics.cka expects 2D tensors (N, d).
    """
    if not _HAS_PLATONIC_REP:
        raise ImportError("platonic-rep not available. Run scripts/setup_third_party.sh.")
    zx = _as_tensor(feat_x)  # (N, d)
    zy = _as_tensor(feat_y)
    score = AlignmentMetrics.unbiased_cka(zx, zy)
    if hasattr(score, "item"):
        score = score.item()
    return float(score)
