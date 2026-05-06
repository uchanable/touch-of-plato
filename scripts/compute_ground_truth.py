#!/usr/bin/env python3
"""Aggregate the per-pair CSVs into the paper's headline numbers.

Reads four per-pair Fig. 1 CSVs that together cover the 12-encoder grid
(C(12,2) = 66 pairs), reads the Procrustes M5 CSV for the same grid, and
emits ``data/results/ground_truth.json`` with:

  - 6 modality blocks (V-V, L-L, T-T, V-L, T-V, L-T) x 5 metrics
    (mutual_knn, debiased_cka, null_knn_z, unbiased_cka, procrustes_m5),
    each with mean / sd / min / max / n.
  - Headline cross-vs-within ratios (V-L / V-V, T-V / V-V, L-T / V-V,
    T-V / T-T, L-T / L-L, V-L / L-L, mean cross / mean within).
  - The full 12-encoder modality table.

Inputs (under ``data/results/`` relative to repo root):

  alignment_matrix_base.csv          # 45 pairs (10-encoder baseline) x 4 metrics
  alignment_matrix_anytouch.csv      # 10 pairs (AnyTouch x existing) x 4 metrics
                                 # CSV holds 6 columns; we keep the 4 standard
  alignment_matrix_tvl_vitb.csv      # 11 pairs (TVL-ViT-B x existing) x 4 metrics
                                 # CSV holds 6 columns; we keep the 4 standard
  alignment_matrix_procrustes_m5.csv         # 66 pairs x 1 metric (M5)

After merging, total = 66 pairs x 5 metrics = 330 rows. Each block
aggregation is exposed as the paper-quoted statistic.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "data" / "results"
OUT_JSON = RESULTS / "ground_truth.json"

MODALITY = {
    # Vision
    "dinov2_small": "V", "dinov2_base": "V", "dinov2_large": "V",
    "clip_l_vision": "V", "siglip_base_vision": "V",
    # Language
    "clip_l_text": "L", "siglip_base_text": "L", "mpnet": "L",
    # Tactile
    "sparsh_dino_base": "T", "sparsh_ijepa_base": "T",
    "anytouch": "T", "tvl_vitb": "T",
}

KEEP_METRICS = ("mutual_knn", "debiased_cka", "null_knn_z", "unbiased_cka", "procrustes_m5")
BLOCKS = ("V-V", "L-L", "T-T", "V-L", "T-V", "L-T")


def block_label(a: str, b: str) -> str:
    """Modality-block label for an (encoder_a, encoder_b) pair.

    Within-modality blocks are V-V / L-L / T-T regardless of order.
    Cross blocks are normalised to V-L / T-V / L-T (matching the paper
    table layout).
    """
    ma, mb = MODALITY[a], MODALITY[b]
    if ma == mb:
        return f"{ma}-{mb}"
    pair = "".join(sorted((ma, mb)))   # 'LV', 'LT', 'TV'
    return {"LV": "V-L", "TV": "T-V", "LT": "L-T"}[pair]


def read_pairs_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            metric = r["metric"]
            if metric not in KEEP_METRICS:
                continue   # drop null_knn_raw_mu / sigma intermediates
            rows.append({
                "encoder_a": r["encoder_a"],
                "encoder_b": r["encoder_b"],
                "metric": metric,
                "value": float(r["value"]),
            })
    return rows


def main() -> int:
    inputs = [
        RESULTS / "alignment_matrix_base.csv",
        RESULTS / "alignment_matrix_anytouch.csv",
        RESULTS / "alignment_matrix_tvl_vitb.csv",
        RESULTS / "alignment_matrix_procrustes_m5.csv",
    ]
    for p in inputs:
        if not p.exists():
            print(f"[ground_truth] missing input: {p}", file=sys.stderr)
            return 1

    rows: list[dict] = []
    for p in inputs:
        rows.extend(read_pairs_csv(p))

    # Sanity: should be exactly 66 pairs x 5 metrics = 330 rows.
    # Normalise pair order alphabetically so different CSVs that store
    # (a, b) vs (b, a) for the same pair collapse into one key.
    metrics_per_pair = defaultdict(set)
    for r in rows:
        key = tuple(sorted((r["encoder_a"], r["encoder_b"])))
        metrics_per_pair[key].add(r["metric"])

    encoders = sorted(MODALITY)
    expected_pairs = {tuple(sorted((a, b))) for a, b in combinations(encoders, 2)}
    actual_pairs = set(metrics_per_pair.keys())

    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs
    if missing or extra:
        print(f"[ground_truth] pair coverage mismatch: missing={len(missing)} extra={len(extra)}", file=sys.stderr)
        if missing:
            print("  first few missing:", sorted(missing)[:5], file=sys.stderr)
        return 1

    incomplete = [p for p, ms in metrics_per_pair.items() if set(KEEP_METRICS) - ms]
    if incomplete:
        print(f"[ground_truth] {len(incomplete)} pairs missing some metrics; first: {incomplete[0]} -> {metrics_per_pair[incomplete[0]]}", file=sys.stderr)
        return 1

    # Aggregate per-block, per-metric.
    by_bm: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        blk = block_label(r["encoder_a"], r["encoder_b"])
        by_bm[(blk, r["metric"])].append(r["value"])

    blocks_summary = {}
    for blk in BLOCKS:
        block_obj = {}
        for metric in KEEP_METRICS:
            vals = by_bm.get((blk, metric), [])
            if not vals:
                continue
            block_obj[metric] = {
                "mean": round(mean(vals), 4),
                "sd": round(stdev(vals), 4) if len(vals) > 1 else None,
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "n": len(vals),
            }
        blocks_summary[blk] = block_obj

    # Headline ratios (mean m-kNN).
    def m(blk: str) -> float:
        return blocks_summary[blk]["mutual_knn"]["mean"]

    headlines = {
        "block_pair_counts": {
            "V-V": 10, "L-L": 3, "T-T": 6, "V-L": 15, "T-V": 20, "L-T": 12,
        },
        "mutual_knn": {
            "V-V_mean": m("V-V"),
            "L-L_mean": m("L-L"),
            "T-T_mean": m("T-T"),
            "V-L_mean": m("V-L"),
            "T-V_mean": m("T-V"),
            "L-T_mean": m("L-T"),
        },
        "ratios_mutual_knn": {
            "VL_over_VV": round(m("V-L") / m("V-V"), 4),
            "TV_over_VV": round(m("T-V") / m("V-V"), 4),
            "LT_over_VV": round(m("L-T") / m("V-V"), 4),
            "TV_over_TT": round(m("T-V") / m("T-T"), 4),
            "LT_over_LL": round(m("L-T") / m("L-L"), 4),
            "VL_over_LL": round(m("V-L") / m("L-L"), 4),
        },
    }
    cross = (m("V-L") + m("T-V") + m("L-T")) / 3.0
    within = (m("V-V") + m("L-L") + m("T-T")) / 3.0
    headlines["ratios_mutual_knn"]["mean_cross_over_mean_within"] = round(cross / within, 4)
    headlines["ratios_mutual_knn"]["TV_over_VL"] = round(m("T-V") / m("V-L"), 4)

    out = {
        "schema_version": "1.0",
        "n_encoders": len(encoders),
        "n_pairs": len(actual_pairs),
        "n_metrics": len(KEEP_METRICS),
        "metrics": list(KEEP_METRICS),
        "encoders_by_modality": {
            "V": [e for e in encoders if MODALITY[e] == "V"],
            "L": [e for e in encoders if MODALITY[e] == "L"],
            "T": [e for e in encoders if MODALITY[e] == "T"],
        },
        "blocks": blocks_summary,
        "headlines": headlines,
        "tvl_n": 43502,
        "source_csvs": [p.name for p in inputs],
    }

    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[ground_truth] wrote {OUT_JSON.relative_to(HERE)}")
    print(f"[ground_truth] {out['n_pairs']} pairs x {out['n_metrics']} metrics aggregated into {len(BLOCKS)} blocks")
    print()
    print(f"  m-kNN means:  V-V={m('V-V'):.4f}  L-L={m('L-L'):.4f}  T-T={m('T-T'):.4f}")
    print(f"                V-L={m('V-L'):.4f}  T-V={m('T-V'):.4f}  L-T={m('L-T'):.4f}")
    print(f"  T-V / V-L  = {headlines['ratios_mutual_knn']['TV_over_VL']:.2f}x")
    print(f"  cross/within (mean m-kNN) = {headlines['ratios_mutual_knn']['mean_cross_over_mean_within']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
