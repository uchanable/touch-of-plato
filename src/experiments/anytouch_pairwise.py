"""Pairwise alignment: AnyTouch x {9 other Table 1 encoders}.

Headline robustness check for §5.7 limitation (ii) of the paper: does the
existing tactile-encoder result reproduce when we swap in AnyTouch (a
sensor-conditioned tactile foundation model with a different backbone)?

Inputs:
    --features-dir : directory with all encoder .npy files (each (N, d_i)),
                     including anytouch.npy AND the 9 baseline encoders
                     (DINOv2 small/base/large, CLIP-L vision/text,
                      SigLIP-Base vision/text, mpnet, Sparsh-DINO,
                      Sparsh-IJEPA).
    --output-dir   : where results.csv / summary.txt / meta.json land.

Computes 4 metrics for each AnyTouch x other pair (9 pairs):
    mutual_knn, debiased_cka, null_knn_z (B=100 perms), unbiased_cka.

This mirrors `src/experiments/fig1_alignment_matrix.py::compute_pairwise_metrics`
but restricted to pairs that involve AnyTouch.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
    unbiased_cka_alignment,
)


_DEFAULT_PARTNERS = [
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
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--anchor", type=str, default="anytouch")
    parser.add_argument("--partners", type=str, nargs="+", default=_DEFAULT_PARTNERS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--null-seed", type=int, default=0)
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = feat_dir / f"{args.anchor}.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor features missing: {anchor_path}")
    anchor = np.load(anchor_path)
    N = anchor.shape[0]
    print(f"[pairwise] anchor={args.anchor} shape={anchor.shape}")

    # Load partners; drop any that are missing on disk (warn loudly).
    partner_feats: dict[str, np.ndarray] = {}
    for name in args.partners:
        p = feat_dir / f"{name}.npy"
        if not p.exists():
            print(f"[pairwise] WARN: missing {p}; skipping.")
            continue
        z = np.load(p)
        if z.shape[0] != N:
            print(f"[pairwise] WARN: {name} N={z.shape[0]} != anchor N={N}; skipping.")
            continue
        partner_feats[name] = z
        print(f"[pairwise]   loaded {name} shape={z.shape}")

    Za = torch.from_numpy(anchor)

    records: list[dict] = []
    t0 = time.time()
    for name, z in tqdm(partner_feats.items(), desc="pairs"):
        Zb = torch.from_numpy(z)

        a_knn = mutual_knn_alignment(Za, Zb, k=args.k)
        a_dcka = debiased_cka_alignment(Za, Zb)
        a_null, a_raw, mu, sigma = null_calibrated_alignment(
            Za, Zb,
            base_metric=lambda x, y: mutual_knn_alignment(x, y, k=args.k),
            n_perms=args.n_perms,
            seed=args.null_seed,
            return_raw=True,
        )
        a_ucka = unbiased_cka_alignment(Za, Zb)

        for metric, val in [
            ("mutual_knn", a_knn),
            ("debiased_cka", a_dcka),
            ("null_knn_z", float(a_null)),
            ("unbiased_cka", a_ucka),
            ("null_knn_raw_mu", mu),
            ("null_knn_raw_sigma", sigma),
        ]:
            records.append({
                "encoder_a": args.anchor,
                "encoder_b": name,
                "metric": metric,
                "value": float(val),
            })

    dt = time.time() - t0
    print(f"[pairwise] computed {len(records)} records in {dt:.1f}s")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["encoder_a", "encoder_b", "metric", "value"])
        w.writeheader()
        w.writerows(records)
    print(f"[pairwise] wrote {csv_path}")

    # Build a compact summary keyed by partner (one row per partner).
    summary_lines = ["partner\tmutual_knn\tdebiased_cka\tnull_knn_z\tunbiased_cka"]
    by_partner: dict[str, dict[str, float]] = {}
    for r in records:
        by_partner.setdefault(r["encoder_b"], {})[r["metric"]] = r["value"]
    for name in args.partners:
        if name not in by_partner:
            continue
        m = by_partner[name]
        summary_lines.append(
            f"{name}\t{m.get('mutual_knn', float('nan')):.4f}\t"
            f"{m.get('debiased_cka', float('nan')):.4f}\t"
            f"{m.get('null_knn_z', float('nan')):.2f}\t"
            f"{m.get('unbiased_cka', float('nan')):.4f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    meta = {
        "anchor": args.anchor,
        "partners": list(partner_feats.keys()),
        "n_samples": int(N),
        "k": args.k,
        "n_perms": args.n_perms,
        "null_seed": args.null_seed,
        "elapsed_seconds": dt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[pairwise] done.")


if __name__ == "__main__":
    main()
