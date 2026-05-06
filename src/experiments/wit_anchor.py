"""WIT-1024 cross-dataset anchor replication (paper Appendix).

Reproduce Huh et al. (ICML 2024) V-L mutual-kNN on the same 1024-pair WIT subset
that Huh used (HuggingFace `minhuh/prh @ wit_1024` branch).

If this pipeline reproduces Huh's V-L ~0.16-0.22 range, then our TVL V-L = 0.027
cannot be a code bug — it must be a TVL domain effect (caption brevity, narrow
surface domain). This is the central cross-dataset sanity check on the WIT-1024 anchor.

Usage:
    python -m src.experiments.wit_anchor --output-dir experiments/wit_anchor

Output:
    experiments/wit_anchor/features/<encoder>.npy
    experiments/wit_anchor/results.csv
    experiments/wit_anchor/summary.txt
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

from src.encoders import get_encoder, list_encoders, LoadedEncoder
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
    unbiased_cka_alignment,
)


def load_wit_1024(data_dir: Path) -> tuple[list[Image.Image], list[str]]:
    """Load WIT 1024 image+caption pairs from minhuh/prh @ wit_1024."""
    parquets = sorted(data_dir.glob("data/train-*.parquet"))
    assert len(parquets) == 2, f"expected 2 parquet files, got {parquets}"
    images: list[Image.Image] = []
    captions: list[str] = []
    for p in parquets:
        t = pq.read_table(p)
        for img_dict, txt in zip(t["image"].to_pylist(), t["text"].to_pylist()):
            img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
            cap = txt[0] if (isinstance(txt, list) and txt) else (txt or "")
            images.append(img)
            captions.append(cap)
    return images, captions


def extract_features_v_or_l(
    enc: LoadedEncoder,
    images: list[Image.Image],
    captions: list[str],
) -> np.ndarray | None:
    """Extract features for V or L encoder. Returns None for tactile (skip)."""
    if enc.modality == "tactile":
        return None
    items = images if enc.modality == "vision" else captions
    feats = np.zeros((len(items), enc.feature_dim), dtype=np.float32)
    enc.model.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(items, desc=f"feat[{enc.name}]", leave=False)):
            if enc.modality == "vision":
                inp = enc.preprocess(x)
                out = enc.model(inp)
            else:
                batch = enc.preprocess([x])
                out = enc.model(batch)
            feats[i] = out.squeeze(0).cpu().numpy().astype(np.float32)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/wit_1024")
    ap.add_argument("--output-dir", default="experiments/wit_anchor")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument(
        "--encoders", nargs="+", default=None,
        help="If None, use V+L only (skip tactile, no touch view in WIT).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    feats_dir = out_dir / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading WIT 1024 from {args.data_dir} ...")
    images, captions = load_wit_1024(Path(args.data_dir))
    N = len(images)
    print(f"      N={N} pairs")
    assert N == 1024, f"expected 1024 pairs, got {N}"

    if args.encoders:
        encoder_ids = args.encoders
    else:
        # WIT-1024 has no tactile pairs, so we run vision + language encoders
        # only. Tactile encoders are listed explicitly (matches Table 1).
        TACTILE = {"sparsh_dino_base", "sparsh_ijepa_base", "anytouch", "tvl_vitb"}
        encoder_ids = [e for e in list_encoders() if e not in TACTILE]
    print(f"[2/4] Encoders: {encoder_ids}")

    features: dict[str, np.ndarray] = {}
    for eid in encoder_ids:
        cache = feats_dir / f"{eid}.npy"
        if cache.exists():
            features[eid] = np.load(cache)
            print(f"      [{eid}] cached: shape={features[eid].shape}")
            continue
        print(f"      [{eid}] loading + extracting ...")
        enc = get_encoder(eid)
        f = extract_features_v_or_l(enc, images, captions)
        if f is None:
            print(f"      [{eid}] skip (tactile)")
            del enc
            continue
        np.save(cache, f)
        features[eid] = f
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"      [{eid}] shape={f.shape}")

    names = sorted(features.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    print(f"[3/4] Computing {len(pairs)} pairs ...")
    records: list[dict] = []
    for a, b in tqdm(pairs, desc="pairs"):
        Za = torch.from_numpy(features[a])
        Zb = torch.from_numpy(features[b])
        knn = mutual_knn_alignment(Za, Zb, k=args.k)
        dcka = debiased_cka_alignment(Za, Zb)
        null = null_calibrated_alignment(
            Za, Zb, base_metric=mutual_knn_alignment, n_perms=args.n_perms,
        )
        ucka = unbiased_cka_alignment(Za, Zb)
        for metric, val in [
            ("mutual_knn", knn),
            ("debiased_cka", dcka),
            ("null_knn_z", float(null)),
            ("unbiased_cka", ucka),
        ]:
            records.append({
                "encoder_a": a, "encoder_b": b,
                "metric": metric, "value": float(val),
            })

    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=["encoder_a", "encoder_b", "metric", "value"],
        )
        w.writeheader()
        w.writerows(records)
    print(f"[4/4] Saved {csv_path}")

    L_set = {"clip_l_text", "siglip_base_text", "mpnet"}
    V_set = {"clip_l_vision", "siglip_base_vision",
             "dinov2_small", "dinov2_base", "dinov2_large"}
    summary_lines: list[str] = []
    summary_lines.append("=== V-L mutual-kNN summary (k={}) ===".format(args.k))
    vl_vals: list[float] = []
    vv_vals: list[float] = []
    ll_vals: list[float] = []
    for r in records:
        if r["metric"] != "mutual_knn":
            continue
        a, b = r["encoder_a"], r["encoder_b"]
        if (a in V_set and b in L_set) or (a in L_set and b in V_set):
            vl_vals.append(r["value"])
            summary_lines.append(
                f"  V-L  {a:25s} <> {b:25s} = {r['value']:.4f}"
            )
        elif a in V_set and b in V_set:
            vv_vals.append(r["value"])
        elif a in L_set and b in L_set:
            ll_vals.append(r["value"])
    summary_lines.append("")
    if vl_vals:
        summary_lines.append(f"  V-L  max  = {max(vl_vals):.4f}")
        summary_lines.append(f"  V-L  mean = {np.mean(vl_vals):.4f}")
        summary_lines.append(f"  V-L  min  = {min(vl_vals):.4f}")
    if vv_vals:
        summary_lines.append(
            f"  V-V  mean = {np.mean(vv_vals):.4f} (n={len(vv_vals)})"
        )
    if ll_vals:
        summary_lines.append(
            f"  L-L  mean = {np.mean(ll_vals):.4f} (n={len(ll_vals)})"
        )
    summary_lines.append("")
    summary_lines.append(
        "Huh ref: V-L max ~0.22 (best CLIP), 0.16 (llama-65b × DINOv2-G)."
    )
    summary_lines.append("Our TVL full-run: V-L mean = 0.027 (vs WIT here).")
    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n")

    meta = {
        "N": N, "k": args.k, "n_perms": args.n_perms,
        "encoders": encoder_ids,
        "data_dir": str(args.data_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
