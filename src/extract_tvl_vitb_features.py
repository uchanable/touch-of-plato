"""Extract TVL standalone tactile encoder features over the TVL dataset.

Mirrors `src/extract_anytouch_features.py` but for the 12th encoder
(TVL ViT-Base trained on TVL with cross-modal contrastive objective —
the "enforce" baseline against the 11 frozen encoders).

Usage:
    python -m src.extract_tvl_vitb_features \
        --tvl-root data/tvl \
        --output-dir experiments/tvl_vitb_full/features
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.encoders import get_encoder
from src.datasets.tvl import TVLDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvl-root", type=str, required=True)
    parser.add_argument("--subset", type=str, choices=["ssvtp", "hct", "all"], default="all")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="ViT-Base is small; 64 fits comfortably on a single mid-range GPU.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "tvl_vitb.npy"
    if cache_path.exists():
        print(f"[extract] {cache_path} already exists; skipping extraction.")
        return

    dataset = TVLDataset(root=Path(args.tvl_root), subset=args.subset)
    N = len(dataset)
    print(f"[extract] TVL subset={args.subset} N={N}")

    enc = get_encoder("tvl_vitb")
    print(f"[extract] encoder loaded; feature_dim={enc.feature_dim} modality={enc.modality}")

    feats = np.zeros((N, enc.feature_dim), dtype=np.float32)
    enc.model.eval()

    t0 = time.time()
    bs = args.batch_size
    with torch.no_grad():
        for start in tqdm(range(0, N, bs), desc="tvl_vitb"):
            stop = min(start + bs, N)
            batch_in = []
            for i in range(start, stop):
                sample = dataset[i]
                batch_in.append(enc.preprocess(sample["tactile"]))
            batch = torch.cat(batch_in, dim=0)
            out = enc.model(batch)
            feats[start:stop] = out.detach().cpu().numpy().astype(np.float32)

    dt = time.time() - t0
    np.save(cache_path, feats)
    print(f"[extract] wrote {cache_path} shape={feats.shape} in {dt:.1f}s "
          f"({N / max(dt, 1e-6):.1f} samples/s)")


if __name__ == "__main__":
    main()
