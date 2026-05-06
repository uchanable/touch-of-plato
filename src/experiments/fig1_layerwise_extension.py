"""Layer-wise probe extension: add AnyTouch + tvl_vitb to the existing 8-encoder
Layer-wise A nullcal pool (Stage 1.5-b).

Existing run: 8 encoder × 22 pair × 4 quartile = 88 cell (Sparsh-DINO/IJEPA,
DINOv2-S/B/L, CLIP-L vision, SigLIP-Base vision, ResNet-50 control).

This extension adds 2 tactile encoders (cross-architecture robustness check):
    AnyTouch  (CLIP-ViT-L/14, 24 blocks)  Q1=L6, Q2=L12, Q3=L18, Q4=L24
    tvl_vitb  (ViT-Base p16,  12 blocks)  Q1=L3, Q2=L6,  Q3=L9,  Q4=L12

New pair set (15 pair × 4 quartile = 60 cell):
    AnyTouch × {Sparsh-DINO, Sparsh-IJEPA, DINOv2-S/B/L, CLIP-L_v, SigLIP_v}  = 7
    tvl_vitb × {Sparsh-DINO, Sparsh-IJEPA, AnyTouch, DINOv2-S/B/L, CLIP-L_v, SigLIP_v} = 8

Patch-mean convention (matches fig1_layerwise.py): drop CLS-like prefix tokens,
mean over patch tokens.
    AnyTouch: seq is [CLS(1), sensor(5), patch(256)] → drop first 6 tokens.
    tvl_vitb: seq is [CLS(1), patch(196)] → drop first 1 token.

Reuses cached layer-wise features for the 8 baseline encoders from
``experiments/fig1_layerwise_optionA/features/``.

Output:
    <output-dir>/features/{anytouch.L6/12/18/24.npy, tvl_vitb.L3/6/9/12.npy}
    <output-dir>/results.csv  (60 cell × {m-kNN, null_knn_z} = 120 records)
    <output-dir>/summary.txt
    <output-dir>/meta.json

Usage:
    python -m src.experiments.fig1_layerwise_extension \
        --baseline-features-dir experiments/fig1_layerwise_optionA/features \
        --output-dir experiments/fig1_layerwise_optionA_extended \
        --n-perms 100
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.encoders import get_encoder, LoadedEncoder
from src.datasets.tvl import TVLDataset
from src.alignment_metrics import mutual_knn_alignment, null_calibrated_alignment


# ===== Encoder layout (must match fig1_layerwise.py) ======================
ENCODER_BLOCKS = {
    "sparsh_dino_base":   12,
    "sparsh_ijepa_base":  12,
    "dinov2_small":       12,
    "dinov2_base":        12,
    "dinov2_large":       24,
    "clip_l_vision":      24,
    "siglip_base_vision": 12,
    "anytouch":           24,   # CLIP-ViT-L/14
    "tvl_vitb":           12,   # ViT-Base p16
}

NEW_T = ["anytouch", "tvl_vitb"]
EXISTING_T = ["sparsh_dino_base", "sparsh_ijepa_base"]
V = ["dinov2_small", "dinov2_base", "dinov2_large", "clip_l_vision", "siglip_base_vision"]


def quartile_to_abs(n_blocks: int, q: int) -> int:
    return int(round(n_blocks * q / 4))


# ===== Layer extractors for the 2 new encoders ============================

class AnyTouchLayerExtractor:
    """Forward-hook patch-mean extractor for AnyTouch.

    AnyTouch wrapper inserts 5 sensor tokens after CLS, so the encoder sequence
    is [CLS(1), sensor(5), patch(256)] = 262. Patch-mean drops the first 6
    positions.
    """

    def __init__(self, wrapper: nn.Module, layer_indices: list[int]):
        self.wrapper = wrapper
        self.vm = wrapper.vision_model
        self.layers = layer_indices
        self.cache: dict[int, torch.Tensor] = {}
        self.handles: list = []
        for L in layer_indices:
            blk = self.vm.encoder.layers[L - 1]
            self.handles.append(blk.register_forward_hook(self._make_hook(L)))

    def _make_hook(self, L: int) -> Callable:
        def hook(module, inp, out):
            self.cache[L] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return hook

    def __call__(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        self.cache.clear()
        _ = self.wrapper(pixel_values)
        feats: dict[int, torch.Tensor] = {}
        for L, t in self.cache.items():
            # Drop CLS(1) + sensor(5) = first 6 tokens, mean over patches.
            feats[L] = t[:, 6:, :].mean(dim=1)
        return feats

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


class TvlVitbLayerExtractor:
    """Forward-hook patch-mean extractor for tvl_vitb (timm ViT-Base p16)."""

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        self.model = model
        self.layers = layer_indices
        self.cache: dict[int, torch.Tensor] = {}
        self.handles: list = []
        for L in layer_indices:
            blk = model.blocks[L - 1]
            self.handles.append(blk.register_forward_hook(self._make_hook(L)))

    def _make_hook(self, L: int) -> Callable:
        def hook(module, inp, out):
            self.cache[L] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
        return hook

    def __call__(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        self.cache.clear()
        _ = self.model(pixel_values)
        feats: dict[int, torch.Tensor] = {}
        for L, t in self.cache.items():
            # Drop 1 CLS token, mean over patches.
            feats[L] = t[:, 1:, :].mean(dim=1)
        return feats

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def make_extractor(name: str, enc: LoadedEncoder, layer_indices: list[int]):
    if name == "anytouch":
        return AnyTouchLayerExtractor(enc.model, layer_indices)
    if name == "tvl_vitb":
        return TvlVitbLayerExtractor(enc.model, layer_indices)
    raise KeyError(name)


def extract_layerwise(name: str, dataset: TVLDataset,
                      layer_indices: list[int]) -> dict[int, np.ndarray]:
    enc = get_encoder(name)
    ext = make_extractor(name, enc, layer_indices)
    N = len(dataset)
    feats: dict[int, np.ndarray] | None = None
    enc.model.eval()
    try:
        with torch.no_grad():
            for n in tqdm(range(N), desc=f"feat[{name}]", leave=False):
                pil = dataset[n]["tactile"]
                inp = enc.preprocess(pil)
                out_dict = ext(inp)
                if feats is None:
                    feats = {
                        L: np.zeros((N, t.shape[-1]), dtype=np.float32)
                        for L, t in out_dict.items()
                    }
                for L, v in out_dict.items():
                    feats[L][n] = v.squeeze(0).cpu().numpy().astype(np.float32)
    finally:
        ext.close()
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return feats or {}


# ===== Main ===============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/tvl")
    ap.add_argument("--subset", default="all", choices=["all", "ssvtp", "hct"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--baseline-features-dir", required=True,
                    help="Existing layerwise features dir for 8 baseline encoders.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--quartiles", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--null-seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    feats_dir = out_dir / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)
    base_feat_dir = Path(args.baseline_features_dir)

    # --- 1. Dataset ---
    print(f"[1/4] Loading TVL ({args.subset}) ...")
    dataset = TVLDataset(
        root=args.data_dir,
        subset=args.subset,
        max_samples=args.max_samples,
    )
    N = len(dataset)
    print(f"      N={N}")

    # --- 2. Extract NEW encoder features (8 quartile files) ---
    print(f"[2/4] Extracting layer-wise features for new encoders ...")
    for name in NEW_T:
        nb = ENCODER_BLOCKS[name]
        layer_abs = [quartile_to_abs(nb, q) for q in args.quartiles]
        todo = [L for L in layer_abs if not (feats_dir / f"{name}.L{L}.npy").exists()]
        if todo:
            print(f"      [{name}] extracting layers {todo}")
            extracted = extract_layerwise(name, dataset, todo)
            for L, arr in extracted.items():
                np.save(feats_dir / f"{name}.L{L}.npy", arr)
                print(f"        saved L{L} shape={arr.shape}")
        else:
            print(f"      [{name}] all layers cached")

    # --- 3. Build feature lookup table for ALL encoders involved in NEW pairs ---
    # NEW T encoders: features in feats_dir
    # Existing T encoders + V encoders: features in base_feat_dir
    features: dict[tuple[str, int], np.ndarray] = {}
    for name in NEW_T:
        nb = ENCODER_BLOCKS[name]
        for q in args.quartiles:
            L = quartile_to_abs(nb, q)
            features[(name, L)] = np.load(feats_dir / f"{name}.L{L}.npy")
    for name in EXISTING_T + V:
        nb = ENCODER_BLOCKS[name]
        for q in args.quartiles:
            L = quartile_to_abs(nb, q)
            p = base_feat_dir / f"{name}.L{L}.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"Baseline cache missing: {p}. Confirm existing layer-wise "
                    f"run features were synced into {base_feat_dir}."
                )
            features[(name, L)] = np.load(p)

    # --- 4. Pair list (NEW only: 15 pairs) ---
    pairs: list[tuple[str, str, str]] = []
    # AnyTouch × 7 partners (existing T + V)
    for partner in EXISTING_T + V:
        ptype = "T-T" if partner in EXISTING_T else "T-V"
        pairs.append(("anytouch", partner, ptype))
    # tvl_vitb × 8 partners (existing T + AnyTouch + V)
    for partner in EXISTING_T + ["anytouch"] + V:
        ptype = "T-T" if partner in EXISTING_T + ["anytouch"] else "T-V"
        pairs.append(("tvl_vitb", partner, ptype))
    assert len(pairs) == 15, f"expected 15 NEW pairs, got {len(pairs)}"

    # ---- Resumable CSV: append per cell, skip already-computed (a, b, q) ----
    csv_path = out_dir / "results.csv"
    fieldnames = [
        "encoder_a", "encoder_b", "pair_type",
        "quartile", "layer_a", "layer_b",
        "metric", "value", "n_samples", "k",
    ]
    done: set[tuple[str, str, int]] = set()
    if csv_path.exists():
        with csv_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                # A (a, b, q) cell is "done" iff its mutual_knn record (and
                # null_knn_z if n_perms>0) is already present.
                done.add((row["encoder_a"], row["encoder_b"], int(row["quartile"])))
        # Remove cells that are missing the null_knn_z record so the next run
        # re-computes them cleanly (avoid half-written cells).
        rows_keep: list[dict] = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                rows_keep.append(row)
        if args.n_perms > 0:
            mknn_keys = {(r["encoder_a"], r["encoder_b"], int(r["quartile"]))
                         for r in rows_keep if r["metric"] == "mutual_knn"}
            zknn_keys = {(r["encoder_a"], r["encoder_b"], int(r["quartile"]))
                         for r in rows_keep if r["metric"] == "null_knn_z"}
            half = mknn_keys - zknn_keys
            if half:
                print(f"      [resume] dropping {len(half)} half-written cells "
                      f"(have m-kNN but not null-z): {list(half)[:3]}...")
                rows_keep = [r for r in rows_keep
                             if (r["encoder_a"], r["encoder_b"], int(r["quartile"])) not in half]
                done = done - half
                # Rewrite CSV with full cells only.
                with csv_path.open("w") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(rows_keep)
        print(f"      [resume] {len(done)} (a, b, q) cells already done; will skip")

    total_cells = len(pairs) * len(args.quartiles)
    print(f"[3/4] Computing m-kNN + null-z for {len(pairs)} new pairs × "
          f"{len(args.quartiles)} quartiles = {total_cells} cells "
          f"({total_cells - len(done)} remaining)")

    # Open CSV in append mode; write header only if file is empty or missing.
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    f_csv = csv_path.open("a", buffering=1)  # line-buffered
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f_csv.flush()

    try:
        for a, b, ptype in tqdm(pairs, desc="pairs"):
            nb_a = ENCODER_BLOCKS[a]
            nb_b = ENCODER_BLOCKS[b]
            for q in args.quartiles:
                if (a, b, q) in done:
                    continue
                la = quartile_to_abs(nb_a, q)
                lb = quartile_to_abs(nb_b, q)
                Za = torch.from_numpy(features[(a, la)])
                Zb = torch.from_numpy(features[(b, lb)])
                val = mutual_knn_alignment(Za, Zb, k=args.k)
                writer.writerow({
                    "encoder_a": a, "encoder_b": b, "pair_type": ptype,
                    "quartile": q, "layer_a": la, "layer_b": lb,
                    "metric": "mutual_knn", "value": float(val),
                    "n_samples": N, "k": args.k,
                })
                if args.n_perms > 0:
                    z = null_calibrated_alignment(
                        Za, Zb,
                        base_metric=lambda x, y: mutual_knn_alignment(x, y, k=args.k),
                        n_perms=args.n_perms,
                        seed=args.null_seed,
                        return_raw=False,
                    )
                    writer.writerow({
                        "encoder_a": a, "encoder_b": b, "pair_type": ptype,
                        "quartile": q, "layer_a": la, "layer_b": lb,
                        "metric": "null_knn_z", "value": float(z),
                        "n_samples": N, "k": args.k,
                    })
                f_csv.flush()  # extra durability against silent kills
                done.add((a, b, q))
    finally:
        f_csv.close()

    # Re-load all records (existing + new) for summary/meta.
    with csv_path.open() as f:
        records = list(csv.DictReader(f))
        # cast value back to float for stats
        for r in records:
            r["value"] = float(r["value"])
            r["quartile"] = int(r["quartile"])
    print(f"[4/4] CSV at {csv_path} ({len(records)} records total)")

    # Summary by pair_type (NEW pairs only)
    summary_lines: list[str] = []
    for ptype in ["T-V", "T-T"]:
        sub_mknn = [r for r in records if r["pair_type"] == ptype and r["metric"] == "mutual_knn"]
        sub_z = [r for r in records if r["pair_type"] == ptype and r["metric"] == "null_knn_z"]
        if not sub_mknn:
            continue
        n_pairs = len(sub_mknn) // len(args.quartiles)
        summary_lines.append(f"\n=== NEW {ptype} (n_pairs={n_pairs}) ===")
        for q in args.quartiles:
            qs = [r["value"] for r in sub_mknn if r["quartile"] == q]
            qz = [r["value"] for r in sub_z if r["quartile"] == q]
            line = (f"  Q{q}: m-kNN mean={np.mean(qs):.4f}  "
                    f"min={min(qs):.4f}  max={max(qs):.4f}")
            if qz:
                line += f"  ||  null-z mean={np.mean(qz):.0f}"
            summary_lines.append(line)
    summary = "\n".join(summary_lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")

    meta = {
        "N": N, "k": args.k, "n_perms": args.n_perms,
        "quartiles": args.quartiles,
        "new_encoders": NEW_T,
        "baseline_features_dir": str(base_feat_dir),
        "new_pairs": [(a, b, p) for a, b, p in pairs],
        "n_pairs": len(pairs),
        "n_records": len(records),
        "subset": args.subset,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
