"""Mutual k-Nearest Neighbor alignment (Eq. 1 of the paper).

Implements Section 3.2 (M1) of the paper:

    A_kNN(Z_X, Z_Y; k) = (1/N) * sum_i |N_k^X(i) ∩ N_k^Y(i)| / k

where N_k^X(i) is the set of top-k nearest neighbors of row i in Z_X
under cosine distance. Invariant to feature rotation and scaling.
"""
from __future__ import annotations
import torch
from .types import FeatureMatrix, _as_tensor


def _cosine_topk_neighbors(z: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k neighbor indices for each row under cosine distance.

    Args:
        z: (N, d) feature matrix.
        k: neighborhood size (excluding self).

    Returns:
        (N, k) LongTensor of neighbor indices.
    """
    N = z.shape[0]
    z_norm = torch.nn.functional.normalize(z, dim=1)
    sim = z_norm @ z_norm.T  # (N, N) cosine similarity
    sim.fill_diagonal_(-float("inf"))  # exclude self
    topk = torch.topk(sim, k=k, dim=1).indices  # (N, k)
    return topk


def mutual_knn_alignment(
    feat_x: FeatureMatrix,
    feat_y: FeatureMatrix,
    k: int = 10,
) -> float:
    """Implements Eq. (1) of the paper: mutual k-NN alignment.

    Args:
        feat_x: (N, d_x) paired feature matrix for modality X.
        feat_y: (N, d_y) paired feature matrix for modality Y (rows aligned to feat_x).
        k: neighborhood size.

    Returns:
        Scalar in [0, 1]. 1.0 = identical neighbor structures, 0.0 = fully disjoint.
    """
    zx = _as_tensor(feat_x)
    zy = _as_tensor(feat_y)
    assert zx.shape[0] == zy.shape[0], f"N mismatch: {zx.shape[0]} vs {zy.shape[0]}"
    N = zx.shape[0]
    assert k < N, f"k={k} must be < N={N}"

    nbrs_x = _cosine_topk_neighbors(zx, k)  # (N, k)
    nbrs_y = _cosine_topk_neighbors(zy, k)  # (N, k)

    # Intersection size per row, averaged.
    # Use boolean mask: for row i, count how many of nbrs_x[i] are in nbrs_y[i].
    inter = torch.zeros(N)
    for i in range(N):
        set_x = set(nbrs_x[i].tolist())
        set_y = set(nbrs_y[i].tolist())
        inter[i] = len(set_x & set_y) / k

    return float(inter.mean().item())
