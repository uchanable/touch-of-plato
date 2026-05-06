"""Figure 1: Pairwise alignment matrix on TVL.

Paper binding: Section 3.3 / Fig. 1 of the paper.

Protocol:
    1. Load selected encoders from Table 1.
    2. Build a TVLDataset with the requested subset and max_samples.
    3. For each encoder, extract features over the whole dataset using
       the view that matches its modality:
           vision encoder  -> sample['vision']
           tactile encoder -> sample['tactile']
           language encoder -> sample['text']
    4. Cache per-encoder feature matrix Z[name] as (N, d) .npy.
    5. For each unordered pair (a, b), compute each metric and append a
       row to results.csv (encoder_a, encoder_b, metric, value).
    6. Plot a 2x2 heatmap grid (one subplot per metric) into
       paper/figures/fig1_alignment_matrix.pdf.

Usage (small sanity run, HCT-only, 100 samples):
    python -m src.experiments.fig1_alignment_matrix \
        --subset hct --max-samples 100 --output-dir experiments/fig1_smoke

Full run (all encoders, all TVL):
    python -m src.experiments.fig1_alignment_matrix \
        --subset all --output-dir experiments/fig1
"""
from __future__ import annotations
import argparse
import csv
import gc
import itertools
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from src.encoders import get_encoder, list_encoders, LoadedEncoder
from src.datasets.tvl import TVLDataset
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
    unbiased_cka_alignment,
)

# Matching between encoder modality and TVL dict key.
_VIEW_FOR_MODALITY = {
    "vision": "vision",
    "tactile": "tactile",
    "language": "text",
}


def extract_features(
    enc: LoadedEncoder,
    dataset: TVLDataset,
) -> np.ndarray:
    """Run encoder over all dataset items, returning (N, feature_dim)."""
    view_key = _VIEW_FOR_MODALITY[enc.modality]
    N = len(dataset)
    feats = np.zeros((N, enc.feature_dim), dtype=np.float32)

    enc.model.eval()
    with torch.no_grad():
        for i in tqdm(range(N), desc=f"feat[{enc.name}]", leave=False):
            sample = dataset[i]
            item = sample[view_key]
            if enc.modality in ("vision", "tactile"):
                inp = enc.preprocess(item)
                out = enc.model(inp)
            else:
                batch = enc.preprocess([item])
                out = enc.model(batch)
            feats[i] = out.squeeze(0).cpu().numpy().astype(np.float32)
    return feats


def compute_pairwise_metrics(
    features: dict[str, np.ndarray],
    k: int,
    n_perms: int,
) -> list[dict]:
    """Compute all 3 alignment metrics for every unordered encoder pair.

    Returns a flat list of records:
        [{encoder_a, encoder_b, metric, value}, ...]
    """
    records: list[dict] = []
    names = sorted(features.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    for a, b in tqdm(pairs, desc="pairs"):
        Za = torch.from_numpy(features[a])
        Zb = torch.from_numpy(features[b])

        a_knn = mutual_knn_alignment(Za, Zb, k=k)
        a_dcka = debiased_cka_alignment(Za, Zb)
        a_null = null_calibrated_alignment(
            Za, Zb,
            base_metric=mutual_knn_alignment,
            n_perms=n_perms,
        )
        a_ucka = unbiased_cka_alignment(Za, Zb)

        for metric, val in [
            ("mutual_knn", a_knn),
            ("debiased_cka", a_dcka),
            ("null_knn_z", float(a_null)),
            ("unbiased_cka", a_ucka),
        ]:
            records.append({
                "encoder_a": a,
                "encoder_b": b,
                "metric": metric,
                "value": float(val),
            })
    return records


def write_results_csv(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder_a", "encoder_b", "metric", "value"])
        writer.writeheader()
        writer.writerows(records)


def plot_heatmap_grid(
    records: list[dict],
    encoder_names: list[str],
    out_path: Path,
    title_prefix: str = "",
) -> None:
    """2x2 heatmap grid: one subplot per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = sorted({r["metric"] for r in records})
    # Build a dict[(metric, a, b)] = value
    val = {(r["metric"], r["encoder_a"], r["encoder_b"]): r["value"] for r in records}

    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    name_to_idx = {n: i for i, n in enumerate(encoder_names)}
    N = len(encoder_names)

    for ax_i, metric in enumerate(metrics):
        mat = np.full((N, N), np.nan)
        for i, a in enumerate(encoder_names):
            mat[i, i] = 1.0  # diagonal
            for j, b in enumerate(encoder_names):
                if i == j:
                    continue
                key1 = (metric, a, b)
                key2 = (metric, b, a)
                if key1 in val:
                    mat[i, j] = val[key1]
                elif key2 in val:
                    mat[i, j] = val[key2]
        im = axes[ax_i].imshow(mat, cmap="viridis", aspect="auto")
        axes[ax_i].set_xticks(range(N))
        axes[ax_i].set_yticks(range(N))
        axes[ax_i].set_xticklabels(encoder_names, rotation=45, ha="right", fontsize=7)
        axes[ax_i].set_yticklabels(encoder_names, fontsize=7)
        axes[ax_i].set_title(f"{title_prefix}{metric}", fontsize=10)
        fig.colorbar(im, ax=axes[ax_i], fraction=0.046)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvl-root", type=str,
                        default=os.environ.get("TVL_ROOT", "data/tvl"))
    parser.add_argument("--subset", type=str, choices=["ssvtp", "hct", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="experiments/fig1")
    parser.add_argument("--encoders", type=str, nargs="+", default=None,
                        help="Encoder names to run (default: all).")
    parser.add_argument("--figure-path", type=str,
                        default="paper/figures/fig1_alignment_matrix.pdf")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    chosen = args.encoders or list_encoders()
    print(f"[fig1] encoders: {chosen}")
    print(f"[fig1] TVL subset={args.subset} max_samples={args.max_samples}")

    # Load dataset once (image loading is lazy per __getitem__)
    dataset = TVLDataset(
        root=Path(args.tvl_root),
        subset=args.subset,
        max_samples=args.max_samples,
    )
    print(f"[fig1] dataset length: {len(dataset)}")

    # Extract features encoder-by-encoder (load -> extract -> free)
    features: dict[str, np.ndarray] = {}
    for name in chosen:
        feat_cache = feat_dir / f"{name}.npy"
        if feat_cache.exists():
            print(f"[fig1] load cached {name}")
            features[name] = np.load(feat_cache)
            continue
        print(f"[fig1] extract {name}")
        enc = get_encoder(name)
        feats = extract_features(enc, dataset)
        np.save(feat_cache, feats)
        features[name] = feats
        del enc
        gc.collect()

    # Compute pairwise metrics
    records = compute_pairwise_metrics(features, k=args.k, n_perms=args.n_perms)
    write_results_csv(records, out_dir / "results.csv")
    print(f"[fig1] wrote {len(records)} records to {out_dir / 'results.csv'}")

    # Plot heatmap grid
    plot_heatmap_grid(
        records,
        encoder_names=chosen,
        out_path=Path(args.figure_path),
        title_prefix=f"TVL {args.subset} (N={len(dataset)}) - ",
    )
    print(f"[fig1] wrote figure to {args.figure_path}")

    # Print summary
    print("\n=== Top 5 pairs by mutual_knn ===")
    knn_records = sorted(
        [r for r in records if r["metric"] == "mutual_knn"],
        key=lambda r: -r["value"],
    )
    for r in knn_records[:5]:
        print(f"  {r['encoder_a']:24s} <-> {r['encoder_b']:24s}  {r['value']:.4f}")


if __name__ == "__main__":
    main()
