"""Alignment metrics for the Platonic Touch paper.

Public API (corresponds to §``sec:metrics``):
    - mutual_knn_alignment (Eq. 1)
    - debiased_cka_alignment (Eq. 2)
    - null_calibrated_alignment (Eq. 3)
    - unbiased_cka_alignment (M4 cross-check)
    - compute_alignment (string-dispatched convenience wrapper)

See paper §``sec:metrics`` (Notation) for the binding between code names and paper notation.
"""
from .mutual_knn import mutual_knn_alignment
from .debiased_cka import debiased_cka_alignment
from .null_calibrated import null_calibrated_alignment
from .unbiased_cka import unbiased_cka_alignment
from .types import FeatureMatrix

__all__ = [
    "mutual_knn_alignment",
    "debiased_cka_alignment",
    "null_calibrated_alignment",
    "unbiased_cka_alignment",
    "compute_alignment",
    "FeatureMatrix",
]


def compute_alignment(
    feat_x: FeatureMatrix,
    feat_y: FeatureMatrix,
    method: str = "mutual_knn",
    **kwargs,
) -> float:
    """String-dispatched entry point for all alignment metrics.

    Args:
        feat_x, feat_y: (N, d) paired feature matrices.
        method: one of 'mutual_knn', 'debiased_cka', 'unbiased_cka',
                'null_kNN' (null-calibrated mutual-kNN),
                'null_dCKA' (null-calibrated debiased CKA).
        **kwargs: passed to the underlying metric.

    Returns:
        Scalar alignment score.
    """
    if method == "mutual_knn":
        return mutual_knn_alignment(feat_x, feat_y, **kwargs)
    if method == "debiased_cka":
        return debiased_cka_alignment(feat_x, feat_y)
    if method == "unbiased_cka":
        return unbiased_cka_alignment(feat_x, feat_y)
    if method == "null_kNN":
        return null_calibrated_alignment(feat_x, feat_y, base_metric=mutual_knn_alignment, **kwargs)
    if method == "null_dCKA":
        return null_calibrated_alignment(feat_x, feat_y, base_metric=debiased_cka_alignment, **kwargs)
    raise ValueError(f"Unknown method: {method}")
