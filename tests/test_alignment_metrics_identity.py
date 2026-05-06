"""Identity case: A(Z, Z) should be 1 for all alignment metrics."""
import torch
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    unbiased_cka_alignment,
)


def _random_features(N=200, d=64, seed=0):
    torch.manual_seed(seed)
    return torch.randn(N, d)


def test_mutual_knn_identity():
    z = _random_features()
    assert mutual_knn_alignment(z, z, k=10) == 1.0, "A(Z, Z) must be 1.0 for mutual-kNN"


def test_debiased_cka_identity():
    z = _random_features()
    score = debiased_cka_alignment(z, z)
    assert abs(score - 1.0) < 1e-5, f"A(Z, Z) should be ~1.0, got {score}"


def test_unbiased_cka_identity():
    z = _random_features()
    score = unbiased_cka_alignment(z, z)
    assert abs(score - 1.0) < 1e-4, f"A(Z, Z) should be ~1.0, got {score}"
