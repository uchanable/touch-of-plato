#!/usr/bin/env python3
"""Quick sanity test (~5 min on a laptop CPU).

Loads two encoders (one vision, one tactile), runs them on the first
500 TVL samples, and computes their mutual-kNN alignment. The number
itself is not the paper headline (that requires N=43,502); the goal
is to detect catastrophic regressions in the loaders or dataset code
before committing to a multi-hour full run.

Run:
    python scripts/sanity_test.py [--n 500]

Exit status is non-zero on any failure, with a short diagnostic line
suitable for CI.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.encoders import get_encoder, list_encoders
from src.alignment_metrics.mutual_knn import mutual_knn_alignment
from src.datasets.tvl import TVLDataset, DEFAULT_TVL_ROOT


def extract_features(enc, samples, side: str) -> np.ndarray:
    """Extract (N, d) features for the requested side ("vision" or "tactile")."""
    out = []
    with torch.no_grad():
        for s in samples:
            img = s[side]   # TVLDataset.__getitem__ returns {"vision": PIL, "tactile": PIL, ...}
            x = enc.preprocess(img)
            f = enc.model(x)
            out.append(f.detach().cpu().numpy().reshape(-1))
    return np.stack(out, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500,
                        help="Number of TVL samples to use (default 500).")
    parser.add_argument("--vision", type=str, default="dinov2_small",
                        help="Vision encoder name (default dinov2_small for speed).")
    parser.add_argument("--tactile", type=str, default="sparsh_dino_base",
                        help="Tactile encoder name (default sparsh_dino_base).")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tvl-root", type=str,
                        default=os.environ.get("TVL_ROOT", str(DEFAULT_TVL_ROOT)))
    args = parser.parse_args()

    available = set(list_encoders())
    for enc_name in (args.vision, args.tactile):
        if enc_name not in available:
            print(f"[sanity] unknown encoder '{enc_name}'. Available: {sorted(available)}", file=sys.stderr)
            return 1

    print(f"[sanity] tvl-root = {args.tvl_root}")
    if not Path(args.tvl_root).exists():
        print(f"[sanity] TVL root does not exist. Download per scripts/download_tvl.sh "
              f"or set TVL_ROOT.", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"[sanity] loading TVL (subset=all, max_samples={args.n}) ...")
    ds = TVLDataset(root=Path(args.tvl_root), subset="all", max_samples=args.n)
    print(f"[sanity]   N = {len(ds)}  (loaded in {time.time()-t0:.1f}s)")

    if len(ds) < 50:
        print(f"[sanity] dataset too small ({len(ds)} < 50). Aborting.", file=sys.stderr)
        return 1

    samples = [ds[i] for i in range(len(ds))]

    print(f"[sanity] loading encoder '{args.vision}' (vision) ...")
    enc_v = get_encoder(args.vision)
    print(f"[sanity] loading encoder '{args.tactile}' (tactile) ...")
    enc_t = get_encoder(args.tactile)

    t1 = time.time()
    feat_v = extract_features(enc_v, samples, side="vision")
    feat_t = extract_features(enc_t, samples, side="tactile")
    print(f"[sanity] features: V={feat_v.shape}, T={feat_t.shape}  "
          f"(extracted in {time.time()-t1:.1f}s)")

    score = mutual_knn_alignment(feat_v, feat_t, k=args.k)
    print(f"[sanity] mutual-kNN({args.vision} <-> {args.tactile}, k={args.k}) "
          f"on N={len(ds)} = {score:.4f}")

    # The paper-side headline T-V mutual-kNN on full TVL is ~0.37 across
    # the 20 T-V pairs. On 500 samples with a single encoder pair, the
    # raw value can drift by 0.05-0.1 in either direction. We just
    # check it is "tactile-vision plausible" rather than a regression.
    if not (0.05 < score < 0.95):
        print(f"[sanity] FAIL: score {score:.4f} outside the plausible range (0.05, 0.95).",
              file=sys.stderr)
        return 1

    print(f"[sanity] PASS  (total {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
