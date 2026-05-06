"""Random Gaussian null: A(Z_X, Z_Y) with independent Gaussian features
should be close to null-distribution mean.

Expected null behavior:
- mutual_knn with k=10, N=500: ~ 10/500 = 0.02 (random chance intersection)
- debiased_cka: ~ 0 (debiased estimator is unbiased at null)
- null_calibrated: ~ 0 by construction (z-score of null is 0)
"""
import torch
import numpy as np
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
)


def test_mutual_knn_null_small():
    """Random Gaussian features should give mutual_knn ≈ k/N."""
    torch.manual_seed(0)
    N, d, k = 500, 64, 10
    zx = torch.randn(N, d)
    zy = torch.randn(N, d)
    score = mutual_knn_alignment(zx, zy, k=k)
    expected_null = k / N  # 0.02
    assert score < 0.05, f"Expected ~{expected_null}, got {score}"


def test_debiased_cka_null_small():
    """Random Gaussian features should give debiased CKA ≈ 0."""
    torch.manual_seed(0)
    N, d = 500, 64
    zx = torch.randn(N, d)
    zy = torch.randn(N, d)
    score = debiased_cka_alignment(zx, zy)
    assert abs(score) < 0.05, f"Expected ≈0, got {score}"


def test_null_calibrated_is_zero_mean():
    """null_calibrated(Z_X, Z_Y) with random features should give z-score near 0."""
    torch.manual_seed(0)
    N, d = 500, 32
    zx = torch.randn(N, d)
    zy = torch.randn(N, d)
    z = null_calibrated_alignment(zx, zy, base_metric=mutual_knn_alignment, n_perms=50)
    # z should be near 0 because both raw and null come from the same null distribution
    assert abs(z) < 3.0, f"z-score of random-vs-random should be within 3 sigma, got {z}"
