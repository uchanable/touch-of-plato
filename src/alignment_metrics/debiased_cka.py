"""Debiased Centered Kernel Alignment (Eq. 2 of the paper).

Implements Section 3.2 (M2) of the paper:

    A_dCKA(Z_X, Z_Y) = HSIC_d(K_X, K_Y) / sqrt(HSIC_d(K_X, K_X) * HSIC_d(K_Y, K_Y))

where K_X = Z_X Z_X^T is the linear kernel and HSIC_d is the unbiased
estimator of Song et al. (2012), which corrects finite-sample bias on
high-dimensional features (Murphy et al., ICLR 2024 Re-Align).
"""
from __future__ import annotations
import torch
from .types import FeatureMatrix, _as_tensor


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    """Unbiased HSIC estimator (Song 2012, Eq. 4 in that paper).

    For (N,N) kernel matrices K, L with N >= 4, set diagonals to 0 and
    compute:

        HSIC_u = [tr(K~ L~) + (1' K~ 1)(1' L~ 1) / ((N-1)(N-2)) - 2/(N-2) * 1' K~ L~ 1] / (N * (N-3))
    """
    N = K.shape[0]
    assert N >= 4, f"unbiased HSIC needs N >= 4, got {N}"

    Kt = K.clone()
    Lt = L.clone()
    Kt.fill_diagonal_(0)
    Lt.fill_diagonal_(0)

    tr_KL = torch.trace(Kt @ Lt)
    sum_K = Kt.sum()
    sum_L = Lt.sum()
    sum_KL = Kt.sum(dim=0) @ Lt.sum(dim=0)  # 1^T K~ L~ 1

    hsic = (tr_KL + sum_K * sum_L / ((N - 1) * (N - 2)) - 2.0 * sum_KL / (N - 2)) / (N * (N - 3))
    return float(hsic.item())


def debiased_cka_alignment(feat_x: FeatureMatrix, feat_y: FeatureMatrix) -> float:
    """Implements Eq. (2) of the paper: debiased CKA via unbiased HSIC.

    Args:
        feat_x: (N, d_x) feature matrix.
        feat_y: (N, d_y) feature matrix, paired to feat_x.

    Returns:
        Scalar. Approximately in [0, 1] but can be slightly negative for
        truly unrelated features due to finite-sample variance — this is
        expected and is precisely why we use the debiased estimator.
    """
    zx = _as_tensor(feat_x)
    zy = _as_tensor(feat_y)
    assert zx.shape[0] == zy.shape[0], f"N mismatch: {zx.shape[0]} vs {zy.shape[0]}"

    # Linear kernels (no centering: HSIC handles that internally).
    Kx = zx @ zx.T
    Ky = zy @ zy.T

    num = _unbiased_hsic(Kx, Ky)
    prod = _unbiased_hsic(Kx, Kx) * _unbiased_hsic(Ky, Ky)
    if prod <= 0:
        return 0.0
    den = prod ** 0.5
    if den <= 0:
        return 0.0
    return num / den
