#!/usr/bin/env python3
"""
M5 — Orthogonal Procrustes alignment metric.

Computes Procrustes-based alignment for all 66 unordered encoder pairs (12 encoders, C(12,2)) over the
full TVL feature set (N=43,502).

Definition:
  M5(X, Y) = 1 - ||X R* - Y||_F^2 / max(||X||_F^2, ||Y||_F^2)
  where R* = U V^T from SVD of X^T Y, after row-centering and Frobenius-norming
  X and Y, and after zero-padding the smaller dim to the larger.

This sits in the orthogonal-alignment family used by Kapoor et al. (2025
NeurIPS, ZlWxWlevZ0, "Bridging Critical Gaps") as one of three alignment
families. M5 = 1 corresponds to perfect Procrustes-recoverable alignment;
M5 = 0 corresponds to no orthogonal-rotation match.

Run after feature extraction (the script reads .npy features
    written by the per-experiment runners under experiments/{alignment_matrix_full,
    anytouch_full, tvl_vitb_full}/features/):
        python scripts/procrustes_m5.py
"""
from __future__ import annotations

import csv
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("PLATONIC_TOUCH_ROOT", Path(__file__).resolve().parents[1]))
FEAT_FIG1 = ROOT / "experiments" / "alignment_matrix_full" / "features"
FEAT_ANYTOUCH = ROOT / "experiments" / "anytouch_full" / "features"
OUT_CSV = ROOT / "experiments" / "alignment_matrix_full" / "procrustes_m5.csv"

ENCODERS = [
    "dinov2_small",
    "dinov2_base",
    "dinov2_large",
    "clip_l_vision",
    "siglip_base_vision",
    "clip_l_text",
    "siglip_base_text",
    "mpnet",
    "sparsh_dino_base",
    "sparsh_ijepa_base",
    "anytouch",
    "tvl_vitb",
]


FEAT_TVL_VITB = ROOT / "experiments" / "tvl_vitb_full" / "features"


def load_feature(name: str) -> np.ndarray:
    if name == "anytouch":
        path = FEAT_ANYTOUCH / "anytouch.npy"
    elif name == "tvl_vitb":
        path = FEAT_TVL_VITB / "tvl_vitb.npy"
    else:
        path = FEAT_FIG1 / f"{name}.npy"
    return np.load(path)


def procrustes_m5(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes alignment, normalised to [0, 1]."""
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Pad to common dim (the smaller one is zero-padded so M5 lives in
    # the larger space; orthogonal Procrustes is well-defined either way).
    d_x = X.shape[1]
    d_y = Y.shape[1]
    d = max(d_x, d_y)
    if d_x < d:
        X = np.pad(X, ((0, 0), (0, d - d_x)))
    if d_y < d:
        Y = np.pad(Y, ((0, 0), (0, d - d_y)))

    # Frobenius normalise (so M5 is a fractional residual)
    nx = np.linalg.norm(X)
    ny = np.linalg.norm(Y)
    if nx == 0 or ny == 0:
        return 0.0
    X = X / nx
    Y = Y / ny

    # Procrustes: R = argmin_{R: R^T R=I} ||X R - Y||_F.
    # Closed form: R = U V^T where USV^T = X^T Y.
    M = X.T @ Y  # (d, d)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    residual = np.linalg.norm(X @ R - Y) ** 2  # in [0, 2] after Frobenius-norming
    # Convert to a [0, 1] similarity. Both are unit-Frobenius, so
    # ||X||^2 = ||Y||^2 = 1 and ||X R - Y||^2 in [0, 4].
    # 0 → identical (M5=1); 2 → orthogonal (M5=0).
    m5 = max(0.0, 1.0 - residual / 2.0)
    return float(m5)


def main():
    # Load all features once
    feats = {}
    print(f"Loading {len(ENCODERS)} feature matrices ...")
    for name in ENCODERS:
        feats[name] = load_feature(name).astype(np.float32)
        print(f"  {name}: {feats[name].shape} {feats[name].dtype}")

    # Compute Procrustes for all 66 unordered pairs
    rows = []
    pairs = list(combinations(ENCODERS, 2))
    print(f"\nComputing M5 for {len(pairs)} pairs ...")
    t0 = time.time()
    for i, (a, b) in enumerate(pairs, 1):
        m5 = procrustes_m5(feats[a], feats[b])
        rows.append({"encoder_a": a, "encoder_b": b, "metric": "procrustes_m5", "value": m5})
        elapsed = time.time() - t0
        eta = elapsed / i * (len(pairs) - i)
        print(f"  [{i}/{len(pairs)}] {a} x {b}: M5={m5:.4f}  (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["encoder_a", "encoder_b", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    # Console summary by block
    from collections import defaultdict
    MOD = {
        **{e: "V" for e in ["dinov2_small", "dinov2_base", "dinov2_large", "clip_l_vision", "siglip_base_vision"]},
        **{e: "L" for e in ["clip_l_text", "siglip_base_text", "mpnet"]},
        **{e: "T" for e in ["sparsh_dino_base", "sparsh_ijepa_base", "anytouch", "tvl_vitb"]},
    }
    block_vals = defaultdict(list)
    for r in rows:
        blk = "".join(sorted([MOD[r["encoder_a"]], MOD[r["encoder_b"]]]))
        block_vals[blk].append(r["value"])
    print("\nBlock-level M5 means (66 pairs):")
    for blk in ["VV", "LL", "TT", "LV", "TV", "LT"]:
        v = block_vals.get(blk, [])
        print(f"  {blk[0]}-{blk[1]}: n={len(v)}  mean={np.mean(v):.4f}  min={min(v):.4f}  max={max(v):.4f}")


if __name__ == "__main__":
    main()
