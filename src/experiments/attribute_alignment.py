"""Figure 4: Attribute-level alignment analysis on TVL.

Paper binding: Section 3.6 of the paper (RO4).

Protocol:
    1. Parse TVL captions into attribute tags using a small hand-curated
       lexicon (hardness, roughness, material, shape).
    2. For each attribute category c, subset the dataset to samples
       whose caption contains at least one positive tag from c.
    3. Compute pairwise alignment (mutual-kNN, debiased CKA) on each
       subset for a fixed encoder pair pool (Sparsh-DINO vs DINOv2,
       CLIP-L, SigLIP, mpnet).
    4. Bar-plot per-attribute alignment to reveal which physical property
       axes carry the tactile-visual-linguistic correspondence.

This is the paper's "attribute-level" answer to the observation that
touch cross-modal alignment is weaker than vision-language
cross-modal alignment: we ask *which* attributes survive.

Usage:
    python -m src.experiments.attribute_alignment \
        --output-dir experiments/attribute_alignment_full \
        --figure-path paper/figures/attribute_alignment.pdf
"""
from __future__ import annotations
import argparse
import csv
import gc
import os
import re
from pathlib import Path

import numpy as np
import torch

from src.encoders import get_encoder
from src.datasets.tvl import TVLDataset, TVLItem
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
)
from src.experiments.alignment_matrix import extract_features


# Hand-curated attribute lexicon. Informed by TVL captions we inspected
# (e.g., "smooth, reflective, hard, cool, sleek"; "textured, rough,
# woven, fibrous, rigid"; "flat, lined, hard").
ATTRIBUTE_LEXICON: dict[str, list[str]] = {
    "hardness": [
        "hard", "firm", "rigid", "stiff", "solid",
        "soft", "squishy", "flexible", "pliable", "cushy", "spongy",
    ],
    "roughness": [
        "rough", "textured", "coarse", "grainy", "bumpy",
        "woven", "fibrous", "ridged", "patterned", "uneven",
        "smooth", "flat", "slick", "glossy", "sleek", "lined",
    ],
    "material": [
        "fabric", "cloth", "metal", "wood", "plastic", "rubber",
        "paper", "glass", "leather", "stone", "ceramic",
    ],
    "thermal_appearance": [
        "cool", "cold", "warm", "hot",
        "reflective", "shiny", "dull", "matte",
    ],
}

# Encoder pairs to evaluate (vision/language on the left, touch on the right).
# Keep it small to bound compute: 4 cross-modal pairs per attribute subset.
PAIRS: list[tuple[str, str]] = [
    ("dinov2_base", "sparsh_dino_base"),
    ("clip_l_vision", "sparsh_dino_base"),
    ("siglip_base_vision", "sparsh_dino_base"),
    ("mpnet", "sparsh_dino_base"),
]

# Also add a baseline: vision-vision + language-language for comparison
BASELINE_PAIRS: list[tuple[str, str]] = [
    ("dinov2_base", "clip_l_vision"),
    ("clip_l_text", "mpnet"),
]


def _tokenize_caption(text: str) -> set[str]:
    """Lowercase, split on non-letters."""
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def _tag_item(item: TVLItem) -> dict[str, bool]:
    """Return a dict {attribute: has_any_positive_token}."""
    tokens = _tokenize_caption(item.text)
    tags = {}
    for attr, lex in ATTRIBUTE_LEXICON.items():
        tags[attr] = any(w in tokens for w in lex)
    return tags


def _subset_indices(dataset: TVLDataset, attribute: str) -> list[int]:
    """Indices of dataset samples whose caption has a positive tag for attribute."""
    idxs = []
    for i, item in enumerate(dataset._index):
        if _tag_item(item)[attribute]:
            idxs.append(i)
    return idxs


def _subset_features(features: dict[str, np.ndarray], idxs: list[int]) -> dict[str, np.ndarray]:
    return {name: feats[idxs] for name, feats in features.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvl-root", type=str,
                        default=os.environ.get("TVL_ROOT", "data/tvl"))
    parser.add_argument("--subset", type=str, choices=["ssvtp", "hct", "all"], default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-subset-size", type=int, default=50,
                        help="Skip attributes with fewer than this many positive samples.")
    parser.add_argument("--output-dir", type=str, default="experiments/attribute_alignment_full")
    parser.add_argument("--figure-path", type=str,
                        default="paper/figures/attribute_alignment.pdf")
    parser.add_argument("--features-from", type=str, default=None,
                        help="Optional: reuse features/*.npy from a previous fig1 run.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(exist_ok=True)

    dataset = TVLDataset(
        root=Path(args.tvl_root),
        subset=args.subset,
        max_samples=args.max_samples,
    )
    print(f"[fig4] dataset={args.subset}  N={len(dataset)}")

    # Collect all encoder names we need
    all_names = sorted({n for p in PAIRS + BASELINE_PAIRS for n in p})
    print(f"[fig4] encoders: {all_names}")

    # Extract (or load) features
    features: dict[str, np.ndarray] = {}
    for name in all_names:
        cache_candidates = [feat_dir / f"{name}.npy"]
        if args.features_from:
            cache_candidates.insert(0, Path(args.features_from) / f"{name}.npy")
        feats = None
        for c in cache_candidates:
            if c.exists():
                feats = np.load(c)
                if feats.shape[0] == len(dataset):
                    print(f"[fig4] load cached {name} from {c}  shape={feats.shape}")
                    break
                else:
                    feats = None
        if feats is None:
            print(f"[fig4] extract {name}")
            enc = get_encoder(name)
            feats = extract_features(enc, dataset)
            np.save(feat_dir / f"{name}.npy", feats)
            del enc
            gc.collect()
        features[name] = feats

    # Tag attribute subsets
    attribute_subsets: dict[str, list[int]] = {}
    for attr in ATTRIBUTE_LEXICON:
        idxs = _subset_indices(dataset, attr)
        attribute_subsets[attr] = idxs
        print(f"[fig4] attribute '{attr}' subset size = {len(idxs)}")

    # Also a "full" (no subsetting) baseline
    attribute_subsets["ALL"] = list(range(len(dataset)))

    # Compute pairwise alignment per (pair, attribute)
    records: list[dict] = []
    for attr, idxs in attribute_subsets.items():
        if len(idxs) < args.min_subset_size:
            print(f"[fig4] skip attribute '{attr}' (only {len(idxs)} samples)")
            continue
        sub = _subset_features(features, idxs)
        for pair_kind, pair_list in [("cross", PAIRS), ("baseline", BASELINE_PAIRS)]:
            for a, b in pair_list:
                zx = torch.from_numpy(sub[a])
                zy = torch.from_numpy(sub[b])
                a_knn = mutual_knn_alignment(zx, zy, k=args.k)
                a_dcka = debiased_cka_alignment(zx, zy)
                for metric, val in [("mutual_knn", a_knn), ("debiased_cka", a_dcka)]:
                    records.append(dict(
                        attribute=attr,
                        subset_size=len(idxs),
                        pair_kind=pair_kind,
                        encoder_a=a,
                        encoder_b=b,
                        metric=metric,
                        value=float(val),
                    ))
                print(f"  [{attr:18s} {pair_kind:8s}] {a:18s} <-> {b:17s} knn={a_knn:+.4f} dcka={a_dcka:+.4f}")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["attribute", "subset_size", "pair_kind",
                        "encoder_a", "encoder_b", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"[fig4] wrote {len(records)} records to {csv_path}")

    # Plot: bar chart of mutual_knn per attribute, grouped by encoder pair
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attributes_plotted = [a for a in list(ATTRIBUTE_LEXICON.keys()) + ["ALL"]
                          if attribute_subsets.get(a, []) and len(attribute_subsets[a]) >= args.min_subset_size]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(attributes_plotted))
    width = 0.13

    pair_records: dict[str, list[float]] = {}
    for a, b in PAIRS + BASELINE_PAIRS:
        key = f"{a[:10]}↔{b[:10]}"
        vals = []
        for attr in attributes_plotted:
            val = next(
                (r["value"] for r in records
                 if r["attribute"] == attr and r["encoder_a"] == a and r["encoder_b"] == b
                 and r["metric"] == "mutual_knn"),
                np.nan,
            )
            vals.append(val)
        pair_records[key] = vals

    for i, (key, vals) in enumerate(pair_records.items()):
        ax.bar(x + (i - len(pair_records) / 2 + 0.5) * width, vals, width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(attributes_plotted, rotation=20, ha="right")
    ax.set_ylabel("mutual-kNN alignment")
    ax.set_title("Fig. 4 — Per-attribute alignment on TVL")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(args.figure_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] wrote figure to {args.figure_path}")


if __name__ == "__main__":
    main()
