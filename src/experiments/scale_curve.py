"""Scale curve of touch-vision alignment on TVL.

Paper binding (LaTeX labels):
    Fig.~``fig:fig2`` (scale curve);
    body §``sec:exp-fig2`` (Scale curve, RO2);
    numerical detail in appendix §``sec:fig2-numerical``
    (Tab.~``tab:fig2-mknn``, Tab.~``tab:fig2-dcka``).

Protocol:
    - Vision-size axis: DINOv2-{Small, Base, Large}  (reference vision encoder;
      Sparsh is released only at Base, so the touch leg is fixed to one point
      and we sweep the vision-side size as a proxy for "scale")
    - Data-fraction axis: TVL subset sizes {10%, 30%, 50%, 100%} of the full
      43,502 paired samples
    - Touch encoders: Sparsh-DINO-Base, Sparsh-IJEPA-Base
    - Metrics: mutual-kNN (Eq. 1), debiased CKA (Eq. 2), null-calibrated z (Eq. 3)

    3 vision sizes x 4 data fractions x 2 touch encoders x 3 metrics
        = 72 measurement rows total.

PRH prediction: for each (vision_size, touch_encoder) curve,
    alignment should rise monotonically with the data fraction.
    Across vision sizes, alignment should rise monotonically with the
    vision encoder's capacity (S < B < L).

Usage:
    python -m src.experiments.scale_curve \
        --output-dir experiments/scale_curve_full \
        --figure-path paper/figures/scale_curve.pdf

Output:
    - experiments/scale_curve_full/results.csv
    - experiments/scale_curve_full/features/{encoder_name}_frac{f}.npy
    - paper/figures/scale_curve.pdf  (2 subplots: Sparsh-DINO, Sparsh-IJEPA)
"""
from __future__ import annotations
import argparse
import csv
import gc
import os
from pathlib import Path

import numpy as np
import torch

from src.encoders import get_encoder
from src.datasets.tvl import TVLDataset
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
)
from src.experiments.alignment_matrix import extract_features


VISION_SIZES = ["dinov2_small", "dinov2_base", "dinov2_large"]
TOUCH_ENCODERS = ["sparsh_dino_base", "sparsh_ijepa_base"]
DATA_FRACTIONS = [0.10, 0.30, 0.50, 1.00]


def extract_at_fraction(
    encoder_name: str,
    dataset_full: TVLDataset,
    fraction: float,
    cache_path: Path,
) -> np.ndarray:
    """Extract features over the first `fraction * len(dataset_full)` samples."""
    if cache_path.exists():
        return np.load(cache_path)
    n = int(round(len(dataset_full) * fraction))
    ds_sub = TVLDataset(
        root=dataset_full.root,
        subset=dataset_full.subset,
        max_samples=n,
        split=dataset_full.split,
    )
    enc = get_encoder(encoder_name)
    feats = extract_features(enc, ds_sub)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feats)
    del enc
    gc.collect()
    return feats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvl-root", type=str,
                        default=os.environ.get("TVL_ROOT", "data/tvl"))
    parser.add_argument("--subset", type=str, choices=["ssvtp", "hct", "all"], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/scale_curve_full")
    parser.add_argument("--figure-path", type=str,
                        default="paper/figures/scale_curve.pdf")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(exist_ok=True)

    dataset_full = TVLDataset(
        root=Path(args.tvl_root),
        subset=args.subset,
        max_samples=None,
    )
    N_full = len(dataset_full)
    print(f"[fig2] dataset={args.subset}  N_full={N_full}")
    print(f"[fig2] vision sizes: {VISION_SIZES}")
    print(f"[fig2] touch encoders: {TOUCH_ENCODERS}")
    print(f"[fig2] data fractions: {DATA_FRACTIONS}")

    records: list[dict] = []

    # Iterate over (fraction, vision_encoder, touch_encoder) triples
    for frac in DATA_FRACTIONS:
        n_sub = int(round(N_full * frac))
        print(f"\n[fig2] === fraction={frac}  N={n_sub} ===")

        # Extract features for each vision encoder at this fraction
        vision_feats: dict[str, np.ndarray] = {}
        for v_name in VISION_SIZES:
            cache = feat_dir / f"{v_name}_frac{frac:.2f}.npy"
            print(f"[fig2] extract {v_name} @ frac={frac}")
            vision_feats[v_name] = extract_at_fraction(v_name, dataset_full, frac, cache)

        # Extract features for each touch encoder at this fraction
        touch_feats: dict[str, np.ndarray] = {}
        for t_name in TOUCH_ENCODERS:
            cache = feat_dir / f"{t_name}_frac{frac:.2f}.npy"
            print(f"[fig2] extract {t_name} @ frac={frac}")
            touch_feats[t_name] = extract_at_fraction(t_name, dataset_full, frac, cache)

        # Compute pairwise alignment for each (vision, touch) pair
        for v_name, z_v_np in vision_feats.items():
            for t_name, z_t_np in touch_feats.items():
                z_v = torch.from_numpy(z_v_np)
                z_t = torch.from_numpy(z_t_np)
                a_knn = mutual_knn_alignment(z_v, z_t, k=args.k)
                a_dcka = debiased_cka_alignment(z_v, z_t)
                a_null = null_calibrated_alignment(
                    z_v, z_t,
                    base_metric=mutual_knn_alignment,
                    n_perms=args.n_perms,
                )
                for metric_name, val in [
                    ("mutual_knn", a_knn),
                    ("debiased_cka", a_dcka),
                    ("null_knn_z", float(a_null)),
                ]:
                    rec = dict(
                        fraction=frac,
                        n_samples=n_sub,
                        vision_encoder=v_name,
                        touch_encoder=t_name,
                        metric=metric_name,
                        value=float(val),
                    )
                    records.append(rec)
                    print(f"  {v_name:15s} <-> {t_name:17s} {metric_name:14s} {val:+.4f}")

    # Write CSV
    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fraction", "n_samples", "vision_encoder", "touch_encoder",
                        "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[fig2] wrote {len(records)} records to {csv_path}")

    # Plot 2-subplot scale curve: one per touch encoder
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    metric_to_plot = "mutual_knn"
    for ax_i, t_name in enumerate(TOUCH_ENCODERS):
        ax = axes[ax_i]
        for v_name in VISION_SIZES:
            xs, ys = [], []
            for frac in DATA_FRACTIONS:
                for r in records:
                    if (r["fraction"] == frac and r["vision_encoder"] == v_name
                            and r["touch_encoder"] == t_name and r["metric"] == metric_to_plot):
                        xs.append(r["n_samples"])
                        ys.append(r["value"])
                        break
            ax.plot(xs, ys, marker="o", label=v_name.replace("dinov2_", "DINOv2-"))
        ax.set_xscale("log")
        ax.set_xlabel("TVL data fraction (N samples)")
        ax.set_ylabel(f"mutual-kNN alignment (k={args.k})")
        ax.set_title(f"{t_name.replace('_', '-')} ↔ DINOv2")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 2 — Scale curve: touch–vision alignment vs. vision-encoder size and data fraction")
    fig.tight_layout()
    Path(args.figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] wrote figure to {args.figure_path}")


if __name__ == "__main__":
    main()
