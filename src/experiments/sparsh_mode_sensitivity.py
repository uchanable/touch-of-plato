"""Sparsh 6-channel sensitivity analysis (Mode A vs Mode B).

Paper binding: appendix §``sec:sparsh-sensitivity``,
Tab.~``tab:sparsh-mode-b`` (alignment delta under the temporal-stride
input). Addresses §``sec:limitations`` caveat (i) (shared optical
pathway / Sparsh-native input format).

Tests whether our same-frame duplication choice (Mode A: I_t || I_t) for
Sparsh's 6-channel input is empirically equivalent to the temporal stride
that Sparsh was natively trained with (Mode B: I_t || I_{t-k}, k=5
frames ~ 80 ms).

Outputs (under --output-dir, default experiments/sparsh_mode_sensitivity):
    pairs_index.csv      — sample_id, run_dir, t_frame, t5_frame, vision_frame, frame_idx
    features/
        sparsh_dino_base.mode_a.npy   (N, 768)
        sparsh_dino_base.mode_b.npy
        sparsh_ijepa_base.mode_a.npy
        sparsh_ijepa_base.mode_b.npy
        dinov2_base.npy               (N, 768)
    cosine_per_sample.csv             — per-sample cosine sim (Mode A vs Mode B) for each Sparsh variant
    results.csv                       — alignment metrics (resumable, append per cell)
    meta.json

Resumability:
    - pairs_index.csv: built once on first run, kept thereafter (deterministic, seed=0)
    - features/*.npy: skipped if already exists
    - results.csv: read on startup, build done set, skip already-computed cells
    - cosine_per_sample.csv: rewritten when both Mode A/B features exist for a Sparsh variant

Usage:
    python -m src.experiments.sparsh_mode_sensitivity \
        --tvl-root data/tvl \
        --output-dir experiments/sparsh_mode_sensitivity \
        --n-samples 500 --k 5
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.encoders import get_encoder
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
)


# ===== Pair index ==========================================================

def build_pair_index(tvl_root: Path, k: int, seed: int, n_samples: int) -> list[dict]:
    """Walk HCT runs, find every train-row with an (I_t, I_{t-k}) pair on disk,
    then deterministic-sample n_samples.

    Determinism: rows are produced in (data_dir, train.csv row order); the
    sampling uses ``np.random.default_rng(seed).permutation`` to pick n_samples
    indices. Same seed → same pair index across runs and hosts.
    """
    hct_root = tvl_root / "tvl_dataset/hct"
    if not hct_root.is_dir():
        raise FileNotFoundError(f"HCT root not found: {hct_root}")

    candidates: list[dict] = []
    for data_dir in sorted(hct_root.glob("data*")):
        if not data_dir.is_dir():
            continue
        csv_path = data_dir / "train.csv"
        if not csv_path.exists():
            continue
        # Build per-run frame-idx set so we can answer "does I_{t-k} exist?" in O(1).
        run_frame_indices: dict[str, set[int]] = defaultdict(set)
        for run_dir in data_dir.glob("*-*"):
            if not run_dir.is_dir():
                continue
            tac_dir = run_dir / "tactile"
            if not tac_dir.is_dir():
                continue
            for f in tac_dir.glob("*.jpg"):
                try:
                    idx = int(f.name.split("-", 1)[0])
                except ValueError:
                    continue
                run_frame_indices[str(run_dir)].add(idx)

        # And a parallel map idx -> filename for O(1) frame-name lookup.
        run_frame_files: dict[str, dict[int, str]] = defaultdict(dict)
        for run_dir_str, idx_set in run_frame_indices.items():
            tac_dir = Path(run_dir_str) / "tactile"
            for f in tac_dir.glob("*.jpg"):
                try:
                    idx = int(f.name.split("-", 1)[0])
                except ValueError:
                    continue
                # If multiple frames share an idx (shouldn't happen), prefer
                # lexicographically smallest for determinism.
                cur = run_frame_files[run_dir_str].get(idx)
                if cur is None or f.name < cur:
                    run_frame_files[run_dir_str][idx] = f.name

        for row in csv.DictReader(csv_path.open()):
            tac_rel = row["tactile"]
            run_part, _, frame_name = tac_rel.partition("/tactile/")
            run_dir_str = str(data_dir / run_part)
            try:
                idx = int(frame_name.split("-", 1)[0])
            except ValueError:
                continue
            if idx < k:
                continue
            target = idx - k
            if target not in run_frame_indices.get(run_dir_str, set()):
                continue
            t5_name = run_frame_files[run_dir_str][target]
            candidates.append({
                "sample_id": f"{data_dir.name}__{run_part}__{idx}",
                "run_dir": run_dir_str,
                "t_frame": str(data_dir / tac_rel),
                "t5_frame": str(Path(run_dir_str) / "tactile" / t5_name),
                "vision_frame": str(data_dir / row["url"]),
                "frame_idx": idx,
            })

    print(f"[pairs] candidates with (I_t, I_t-{k}): {len(candidates)}")
    if len(candidates) < n_samples:
        raise SystemExit(f"only {len(candidates)} pairs available, need {n_samples}")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.permutation(len(candidates))[:n_samples]
    chosen_idx.sort()  # keep ascending order for stable downstream code
    return [candidates[i] for i in chosen_idx]


# ===== Feature extraction ==================================================

def _build_3ch_preprocess() -> Callable:
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf  # PIL → (3, 224, 224)


def extract_sparsh(
    pairs: list[dict],
    encoder_name: str,
    mode: str,           # "a" or "b"
    out_path: Path,
) -> np.ndarray:
    """Extract Sparsh features under Mode A (same-frame) or Mode B (temporal stride)."""
    if out_path.exists():
        print(f"[extract] {out_path.name} cached, loading.")
        return np.load(out_path)

    enc = get_encoder(encoder_name)
    enc.model.eval()
    device = next(enc.model.parameters()).device

    tf3 = _build_3ch_preprocess()
    N = len(pairs)
    feats = np.zeros((N, enc.feature_dim), dtype=np.float32)

    with torch.no_grad():
        for i, p in enumerate(tqdm(pairs, desc=f"{encoder_name}.mode_{mode}", leave=False)):
            t_pil = Image.open(p["t_frame"]).convert("RGB")
            a3 = tf3(t_pil).unsqueeze(0)            # (1, 3, 224, 224)
            if mode == "a":
                six = torch.cat([a3, a3], dim=1)
            elif mode == "b":
                t5_pil = Image.open(p["t5_frame"]).convert("RGB")
                b3 = tf3(t5_pil).unsqueeze(0)
                six = torch.cat([a3, b3], dim=1)
            else:
                raise ValueError(mode)
            out = enc.model(six.to(device))         # (1, 768)
            feats[i] = out.squeeze(0).cpu().numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feats)
    print(f"[extract] wrote {out_path} shape={feats.shape}")
    del enc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return feats


def extract_dinov2_reference(pairs: list[dict], out_path: Path) -> np.ndarray:
    """DINOv2-Base on the I_t RGB vision frame."""
    if out_path.exists():
        print(f"[extract] {out_path.name} cached, loading.")
        return np.load(out_path)
    enc = get_encoder("dinov2_base")
    enc.model.eval()
    device = next(enc.model.parameters()).device
    N = len(pairs)
    feats = np.zeros((N, enc.feature_dim), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(tqdm(pairs, desc="dinov2_base", leave=False)):
            pil = Image.open(p["vision_frame"]).convert("RGB")
            inp = enc.preprocess(pil)
            feats[i] = enc.model(inp).squeeze(0).cpu().numpy().astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feats)
    print(f"[extract] wrote {out_path} shape={feats.shape}")
    del enc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return feats


# ===== Cosine per sample ===================================================

def cosine_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between two (N, d) feature matrices."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (an * bn).sum(axis=1)


# ===== Main ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tvl-root", type=str,
                    default=os.environ.get("TVL_ROOT", "data/tvl"))
    ap.add_argument("--output-dir", type=str,
                    default="experiments/sparsh_mode_sensitivity")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / "pairs_index.csv"

    # ---- 1. Pair index (resumable) ----
    if pairs_csv.exists():
        with pairs_csv.open() as f:
            pairs = list(csv.DictReader(f))
            for p in pairs:
                p["frame_idx"] = int(p["frame_idx"])
        print(f"[pairs] loaded {len(pairs)} cached pairs from {pairs_csv}")
    else:
        pairs = build_pair_index(
            tvl_root=Path(args.tvl_root),
            k=args.k,
            seed=args.seed,
            n_samples=args.n_samples,
        )
        with pairs_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
        print(f"[pairs] wrote {pairs_csv} ({len(pairs)} samples)")

    if len(pairs) != args.n_samples:
        print(f"WARNING: cached pair count {len(pairs)} != requested {args.n_samples}")

    # ---- 2. Feature extraction (resumable per file) ----
    print("\n[extract] features ...")
    sd_a = extract_sparsh(pairs, "sparsh_dino_base", "a", feat_dir / "sparsh_dino_base.mode_a.npy")
    sd_b = extract_sparsh(pairs, "sparsh_dino_base", "b", feat_dir / "sparsh_dino_base.mode_b.npy")
    si_a = extract_sparsh(pairs, "sparsh_ijepa_base", "a", feat_dir / "sparsh_ijepa_base.mode_a.npy")
    si_b = extract_sparsh(pairs, "sparsh_ijepa_base", "b", feat_dir / "sparsh_ijepa_base.mode_b.npy")
    dv2 = extract_dinov2_reference(pairs, feat_dir / "dinov2_base.npy")

    feats = {
        "sparsh_dino_base.mode_a": sd_a,
        "sparsh_dino_base.mode_b": sd_b,
        "sparsh_ijepa_base.mode_a": si_a,
        "sparsh_ijepa_base.mode_b": si_b,
        "dinov2_base": dv2,
    }
    N = len(pairs)
    for name, arr in feats.items():
        if arr.shape[0] != N:
            raise SystemExit(f"feature {name} N={arr.shape[0]} != {N}")

    # ---- 3. Cosine per sample (rewritten on every run; cheap) ----
    cos_csv = out_dir / "cosine_per_sample.csv"
    cos_dino = cosine_per_sample(sd_a, sd_b)
    cos_ijepa = cosine_per_sample(si_a, si_b)
    with cos_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "sparsh_dino_cos_AB", "sparsh_ijepa_cos_AB"])
        for i, p in enumerate(pairs):
            w.writerow([p["sample_id"], float(cos_dino[i]), float(cos_ijepa[i])])
    print(f"[cosine] wrote {cos_csv}")
    print(f"  sparsh_dino_base   mean={cos_dino.mean():.4f} std={cos_dino.std():.4f} "
          f"min={cos_dino.min():.4f} max={cos_dino.max():.4f}")
    print(f"  sparsh_ijepa_base  mean={cos_ijepa.mean():.4f} std={cos_ijepa.std():.4f} "
          f"min={cos_ijepa.min():.4f} max={cos_ijepa.max():.4f}")

    # ---- 4. Alignment metrics (per-cell incremental, resumable) ----
    results_csv = out_dir / "results.csv"
    fieldnames = ["pair_a", "pair_b", "metric", "value", "n_samples", "k"]
    done: set[tuple[str, str, str]] = set()
    if results_csv.exists() and results_csv.stat().st_size > 0:
        with results_csv.open() as f:
            for row in csv.DictReader(f):
                done.add((row["pair_a"], row["pair_b"], row["metric"]))
        print(f"[align] resume: {len(done)} cells already done")

    # Cells:
    #  - {sd, si} × {a, b} ↔ dinov2_base × {mutual_knn, debiased_cka, null_knn_z} = 4 × 3 = 12
    #  - {sd, si}: mode_a ↔ mode_b × {mutual_knn, debiased_cka, cos_mean, cos_std, cos_p10, cos_p90} = 2 × 6 = 12
    cells: list[tuple[str, str, str]] = []
    for enc_id in ("sparsh_dino_base", "sparsh_ijepa_base"):
        for mode in ("a", "b"):
            for metric in ("mutual_knn", "debiased_cka", "null_knn_z"):
                cells.append((f"{enc_id}.mode_{mode}", "dinov2_base", metric))
    for enc_id in ("sparsh_dino_base", "sparsh_ijepa_base"):
        cells.append((f"{enc_id}.mode_a", f"{enc_id}.mode_b", "mutual_knn"))
        cells.append((f"{enc_id}.mode_a", f"{enc_id}.mode_b", "debiased_cka"))
        cells.append((f"{enc_id}.mode_a", f"{enc_id}.mode_b", "cos_mean"))
        cells.append((f"{enc_id}.mode_a", f"{enc_id}.mode_b", "cos_std"))
        cells.append((f"{enc_id}.mode_a", f"{enc_id}.mode_b", "cos_p10"))
        cells.append((f"{enc_id}.mode_a", f"{enc_id}.mode_b", "cos_p90"))

    write_header = (not results_csv.exists()) or results_csv.stat().st_size == 0
    f_csv = results_csv.open("a", buffering=1)
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f_csv.flush()

    try:
        for pa, pb, metric in cells:
            if (pa, pb, metric) in done:
                continue
            za = torch.from_numpy(feats[pa])
            zb = torch.from_numpy(feats[pb])
            if metric == "mutual_knn":
                val = mutual_knn_alignment(za, zb, k=args.knn_k)
            elif metric == "debiased_cka":
                val = debiased_cka_alignment(za, zb)
            elif metric == "null_knn_z":
                val = null_calibrated_alignment(
                    za, zb,
                    base_metric=lambda x, y: mutual_knn_alignment(x, y, k=args.knn_k),
                    n_perms=args.n_perms,
                    seed=args.seed,
                )
            elif metric.startswith("cos_"):
                cos = cos_dino if "sparsh_dino_base" in pa else cos_ijepa
                if metric == "cos_mean":
                    val = float(cos.mean())
                elif metric == "cos_std":
                    val = float(cos.std())
                elif metric == "cos_p10":
                    val = float(np.percentile(cos, 10))
                elif metric == "cos_p90":
                    val = float(np.percentile(cos, 90))
                else:
                    raise KeyError(metric)
            else:
                raise KeyError(metric)
            writer.writerow({
                "pair_a": pa, "pair_b": pb, "metric": metric,
                "value": float(val), "n_samples": N, "k": args.knn_k,
            })
            f_csv.flush()
            done.add((pa, pb, metric))
            print(f"  {pa} × {pb} {metric:14s} = {float(val):+.4f}")
    finally:
        f_csv.close()

    # ---- 5. Meta ----
    meta = {
        "tvl_root": args.tvl_root,
        "n_samples": N,
        "k": args.k,
        "knn_k": args.knn_k,
        "n_perms": args.n_perms,
        "seed": args.seed,
        "encoders": ["sparsh_dino_base", "sparsh_ijepa_base", "dinov2_base"],
        "modes": ["a", "b"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
