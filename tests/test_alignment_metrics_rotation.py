"""Rotation invariance: A(Z, RZ) = A(Z, Z) for orthogonal R."""
import torch
from src.alignment_metrics import mutual_knn_alignment, debiased_cka_alignment


def _random_features(N=200, d=64, seed=0):
    torch.manual_seed(seed)
    return torch.randn(N, d)


def _random_orthogonal(d, seed=1):
    torch.manual_seed(seed)
    A = torch.randn(d, d)
    Q, _ = torch.linalg.qr(A)
    return Q


def test_mutual_knn_rotation_invariance():
    z = _random_features()
    R = _random_orthogonal(z.shape[1])
    zr = z @ R
    a1 = mutual_knn_alignment(z, z, k=10)
    a2 = mutual_knn_alignment(z, zr, k=10)
    assert abs(a1 - a2) < 1e-5, f"mutual-kNN should be rotation-invariant: {a1} vs {a2}"


def test_debiased_cka_rotation_invariance():
    z = _random_features()
    R = _random_orthogonal(z.shape[1])
    zr = z @ R
    a1 = debiased_cka_alignment(z, z)
    a2 = debiased_cka_alignment(z, zr)
    assert abs(a1 - a2) < 1e-4, f"debiased CKA should be rotation-invariant: {a1} vs {a2}"
