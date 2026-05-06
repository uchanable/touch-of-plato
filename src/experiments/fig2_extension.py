"""Fig. 2 (Scale curve) extension — 14 new vision×touch encoder combos.

Existing fig2_full (Sparsh-DINO/IJEPA × DINOv2-S/B/L = 6 combos × 4 fractions
× 3 metrics = 72 rows) covers only the original encoder pool. This extension
adds 14 new combos to cover every vision×tactile pair the 12-encoder
PlatonicTouch pool can form:

    Sparsh-DINO   × {CLIP-L_v, SigLIP_v}
    Sparsh-IJEPA  × {CLIP-L_v, SigLIP_v}
    AnyTouch      × {DINOv2-S, DINOv2-B, DINOv2-L, CLIP-L_v, SigLIP_v}
    tvl_vitb      × {DINOv2-S, DINOv2-B, DINOv2-L, CLIP-L_v, SigLIP_v}

= 14 combos × 4 fractions × 2 metrics (m-kNN + dCKA) = 112 cells.

Output schema matches existing fig2_full exactly:
    fraction, n_samples, vision_encoder, touch_encoder, metric, value

Sampling: fig2 uses ``TVLDataset(max_samples=n)`` which is the deterministic
*first n entries* (not random with seed). So we just slice cached full
features ``feats[:n_eff]`` — gives bitwise-identical fraction subsets to
the original fig2_full.

Feature extraction is NOT needed: every encoder's full-N feature matrix is
already cached from fig1_full (10 encoders) + anytouch_full + tvl_vitb_full.

Resumable: every cell (combo × fraction × metric) is appended to the CSV
as soon as it's computed, with line-buffered flush. On restart, the script
loads the existing CSV, builds a done-set, and skips already-computed cells.

Spot-check: before running new combos, the script can re-compute one cell
that's already in fig2_full (Sparsh-DINO × DINOv2-Small × fraction=0.10,
expected m-kNN = 0.0410) to verify slicing matches the original sampling.
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
)


VISION = ["dinov2_small", "dinov2_base", "dinov2_large",
          "clip_l_vision", "siglip_base_vision"]
TOUCH = ["sparsh_dino_base", "sparsh_ijepa_base", "anytouch", "tvl_vitb"]
DATA_FRACTIONS = [0.10, 0.30, 0.50, 1.00]

# 14 NEW combos (everything in VISION × TOUCH except the 6 already-measured
# Sparsh × DINOv2 combos)
EXISTING = {(v, t) for v in ["dinov2_small", "dinov2_base", "dinov2_large"]
            for t in ["sparsh_dino_base", "sparsh_ijepa_base"]}
NEW_COMBOS = [(v, t) for t in TOUCH for v in VISION if (v, t) not in EXISTING]
assert len(NEW_COMBOS) == 14, f"expected 14 new combos, got {len(NEW_COMBOS)}"

METRICS = ["mutual_knn", "debiased_cka"]
FIELDNAMES = ["fraction", "n_samples", "vision_encoder", "touch_encoder",
              "metric", "value"]


def load_feature(name: str, fig1_dir: Path, anytouch_path: Path,
                 tvl_vitb_path: Path) -> np.ndarray:
    """Load full N=43,502 features for an encoder from the appropriate cache."""
    if name == "anytouch":
        return np.load(anytouch_path)
    if name == "tvl_vitb":
        return np.load(tvl_vitb_path)
    p = fig1_dir / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing cached feature: {p}")
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig1-features-dir", default=str(
        Path("experiments/fig1_full/features")))
    ap.add_argument("--anytouch-features", default=str(
        Path("experiments/anytouch_full/features/anytouch.npy")))
    ap.add_argument("--tvl-vitb-features", default=str(
        Path("experiments/tvl_vitb_full/features/tvl_vitb.npy")))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--spot-check", action="store_true",
                    help="Re-compute one cell from fig2_full and assert match (≈0.0410).")
    args = ap.parse_args()

    fig1_dir = Path(args.fig1_features_dir)
    anytouch_path = Path(args.anytouch_features)
    tvl_vitb_path = Path(args.tvl_vitb_features)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    # ---- Load all features (12 encoders, full N) ----
    print(f"[fig2-ext] loading features ...")
    feats: dict[str, np.ndarray] = {}
    for name in VISION + TOUCH:
        feats[name] = load_feature(name, fig1_dir, anytouch_path, tvl_vitb_path)
        print(f"  {name:20s} shape={feats[name].shape}")
    N_full = feats[VISION[0]].shape[0]
    for name, arr in feats.items():
        if arr.shape[0] != N_full:
            raise ValueError(f"{name} N={arr.shape[0]} != {N_full}")
    print(f"[fig2-ext] N_full={N_full}")

    # ---- Optional spot-check: reproduce one fig2_full cell ----
    if args.spot_check:
        n10 = int(round(N_full * 0.10))
        z_v = torch.from_numpy(feats["dinov2_small"][:n10])
        z_t = torch.from_numpy(feats["sparsh_dino_base"][:n10])
        val = mutual_knn_alignment(z_v, z_t, k=args.k)
        expected = 0.0410
        diff = abs(val - expected)
        print(f"[spot-check] sparsh_dino × dinov2_small @ frac=0.10: "
              f"got {val:.4f}, expected {expected}, |Δ|={diff:.4f}")
        if diff > 1e-3:
            raise SystemExit(f"spot-check FAILED — sampling mismatch (Δ={diff:.4e} > 1e-3)")
        print(f"[spot-check] PASS")

    # ---- Resume: read existing CSV, build done set ----
    done: set[tuple[float, str, str, str]] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                done.add((float(row["fraction"]), row["vision_encoder"],
                          row["touch_encoder"], row["metric"]))
        print(f"[resume] {len(done)} cells already done; will skip")

    # ---- Compute remaining cells ----
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    f_csv = csv_path.open("a", buffering=1)
    writer = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
        f_csv.flush()

    total_cells = len(NEW_COMBOS) * len(DATA_FRACTIONS) * len(METRICS)
    print(f"[fig2-ext] total cells = {len(NEW_COMBOS)} combos × "
          f"{len(DATA_FRACTIONS)} fractions × {len(METRICS)} metrics = {total_cells}; "
          f"{total_cells - len(done)} remaining")

    t_start = time.time()
    completed_this_run = 0

    try:
        for v_name, t_name in NEW_COMBOS:
            for frac in DATA_FRACTIONS:
                n_eff = int(round(N_full * frac))
                # Slice once per (combo, fraction) — both metrics share it.
                z_v_full = feats[v_name]
                z_t_full = feats[t_name]
                z_v_sub = z_v_full[:n_eff]
                z_t_sub = z_t_full[:n_eff]
                z_v_t = torch.from_numpy(z_v_sub)
                z_t_t = torch.from_numpy(z_t_sub)
                for metric in METRICS:
                    key = (frac, v_name, t_name, metric)
                    if key in done:
                        continue
                    cell_t0 = time.time()
                    if metric == "mutual_knn":
                        val = mutual_knn_alignment(z_v_t, z_t_t, k=args.k)
                    elif metric == "debiased_cka":
                        val = debiased_cka_alignment(z_v_t, z_t_t)
                    else:
                        raise KeyError(metric)
                    cell_dt = time.time() - cell_t0
                    writer.writerow({
                        "fraction": frac,
                        "n_samples": n_eff,
                        "vision_encoder": v_name,
                        "touch_encoder": t_name,
                        "metric": metric,
                        "value": float(val),
                    })
                    f_csv.flush()
                    done.add(key)
                    completed_this_run += 1
                    print(f"  [{completed_this_run:3d}/{total_cells - (len(done) - completed_this_run):3d}] "
                          f"frac={frac}  {v_name:20s} × {t_name:20s} {metric:14s} = "
                          f"{val:.4f}  ({cell_dt:.1f}s)")
    finally:
        f_csv.close()

    total_dt = time.time() - t_start
    print(f"\n[fig2-ext] done. {completed_this_run} new cells in {total_dt:.0f}s "
          f"= {total_dt / 60:.1f} min")

    # ---- Summary ----
    print(f"\n[fig2-ext] CSV at {csv_path}")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"[fig2-ext] total rows: {len(rows)}")


if __name__ == "__main__":
    main()
