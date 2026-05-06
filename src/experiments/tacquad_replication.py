"""Fig. 1 alignment matrix replicated on TacQuad (cross-dataset).

Paper binding: cross-dataset replication of the T-V > V-L pattern outside TVL.

Goal: Reproduce the T-V > V-L pattern observed on TVL using a different
vision-touch-language dataset (TacQuad / AnyTouch ICLR 2025). If T-V mutual-kNN
remains substantially > V-L mutual-kNN, the gap cannot be a TVL-specific
artifact (caption brevity, narrow surface domain, SSVTP-only annotation, etc.).

Protocol (same as alignment_matrix.py — only the dataset differs):
    1. Load 10 encoders (Sparsh-{DINO,IJEPA}, DINOv2 {S,B,L}, CLIP-L {V,T},
       SigLIP-Base {V,T}, mpnet).
    2. Build a TacQuadDataset (default: indoor + DIGIT sensor).
    3. Extract per-encoder features over the dataset.
    4. Compute pairwise mutual_knn / debiased_cka / null_knn_z / unbiased_cka.
    5. Write results.csv, summary.txt (T-V vs V-L breakdown), meta.json.

Usage (smoke test, ~100 samples):
    python -m src.experiments.tacquad_replication \\
        --subset indoor --sensor digit --max-samples 100 \\
        --output-dir experiments/tacquad_replication_smoke

Usage (full indoor run):
    python -m src.experiments.tacquad_replication \\
        --subset indoor --sensor digit --output-dir experiments/tacquad_replication_full
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.encoders import get_encoder, list_encoders, LoadedEncoder
from src.datasets.tacquad import TacQuadDataset
from src.alignment_metrics import (
    mutual_knn_alignment,
    debiased_cka_alignment,
    null_calibrated_alignment,
    unbiased_cka_alignment,
)


_VIEW_FOR_MODALITY = {
    "vision": "vision",
    "tactile": "tactile",
    "language": "text",
}

# Encoder modality groupings (mirrors alignment_matrix grouping).
_T_ENCODERS = {"sparsh_dino_base", "sparsh_ijepa_base"}
_V_ENCODERS = {
    "clip_l_vision", "siglip_base_vision",
    "dinov2_small", "dinov2_base", "dinov2_large",
}
_L_ENCODERS = {"clip_l_text", "siglip_base_text", "mpnet"}


def extract_features(enc: LoadedEncoder, dataset: TacQuadDataset) -> np.ndarray:
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
    records: list[dict] = []
    names = sorted(features.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

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
                "encoder_a": a, "encoder_b": b,
                "metric": metric, "value": float(val),
            })
    return records


def write_results_csv(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["encoder_a", "encoder_b", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(records)


def _classify_pair(a: str, b: str) -> str | None:
    """Return TV / VL / TL / TT / VV / LL or None."""
    sets = (
        ("T", _T_ENCODERS), ("V", _V_ENCODERS), ("L", _L_ENCODERS),
    )
    def _which(x):
        for tag, s in sets:
            if x in s:
                return tag
        return None
    ta, tb = _which(a), _which(b)
    if ta is None or tb is None:
        return None
    return "".join(sorted([ta, tb]))


def write_summary(records: list[dict], out_path: Path, meta: dict) -> str:
    """Writes summary.txt with T-V vs V-L breakdown.  Returns the text."""
    knn = [r for r in records if r["metric"] == "mutual_knn"]
    by_group: dict[str, list[float]] = {}
    for r in knn:
        g = _classify_pair(r["encoder_a"], r["encoder_b"])
        if g is None:
            continue
        by_group.setdefault(g, []).append(r["value"])

    lines: list[str] = []
    lines.append("=== TacQuad cross-dataset summary (mutual-kNN, k={}) ===".format(meta["k"]))
    lines.append(
        f"  dataset: TacQuad subset={meta['subset']} sensor={meta['sensor']} N={meta['N']}"
    )
    lines.append("")
    for group in ["TV", "VL", "TL", "TT", "VV", "LL"]:
        vals = by_group.get(group, [])
        if not vals:
            continue
        lines.append(
            f"  {group}  n={len(vals):<3d}  mean={np.mean(vals):.4f}"
            f"  max={max(vals):.4f}  min={min(vals):.4f}"
        )
    lines.append("")
    tv = by_group.get("TV", [])
    vl = by_group.get("VL", [])
    if tv and vl:
        ratio = float(np.mean(tv) / max(np.mean(vl), 1e-9))
        lines.append(f"  T-V / V-L mean ratio = {ratio:.2f}x")
        lines.append(
            "  TVL reference (full pipeline):  T-V = 0.391, V-L = 0.027 (14x)"
        )
        if np.mean(tv) > np.mean(vl):
            lines.append(
                "  PATTERN: TacQuad T-V > V-L (consistent with TVL → "
                "TVL-specific artifact NOT supported)"
            )
        else:
            lines.append(
                "  PATTERN: TacQuad T-V <= V-L (inconsistent → inconsistent with TVL — revisit cross-dataset interpretation)"
            )
    lines.append("")
    lines.append("=== Top-5 pairs by mutual_knn ===")
    knn_sorted = sorted(knn, key=lambda r: -r["value"])
    for r in knn_sorted[:5]:
        lines.append(
            f"  {r['encoder_a']:24s} <-> {r['encoder_b']:24s}  {r['value']:.4f}"
        )
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tacquad-root",
        default=os.environ.get("TACQUAD_ROOT", "data/tacquad/extracted"),
    )
    ap.add_argument("--subset", choices=["indoor", "outdoor", "all"], default="indoor")
    ap.add_argument("--sensor", choices=["digit", "gelsight", "duragel"], default="digit")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--per-object-max", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=100)
    ap.add_argument("--output-dir", default="experiments/tacquad_replication_full")
    ap.add_argument("--encoders", nargs="+", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    chosen = args.encoders or list_encoders()
    print(f"[tacquad_replication] encoders: {chosen}")
    print(
        f"[tacquad_replication] TacQuad subset={args.subset} sensor={args.sensor} "
        f"max_samples={args.max_samples} per_object_max={args.per_object_max}"
    )

    dataset = TacQuadDataset(
        root=Path(args.tacquad_root),
        subset=args.subset,
        sensor=args.sensor,
        max_samples=args.max_samples,
        per_object_max=args.per_object_max,
    )
    N = len(dataset)
    print(f"[tacquad_replication] dataset length: {N}")
    if N == 0:
        raise RuntimeError(
            "Empty dataset — check root, subset, sensor, or contact CSV ranges."
        )

    features: dict[str, np.ndarray] = {}
    for name in chosen:
        feat_cache = feat_dir / f"{name}.npy"
        if feat_cache.exists():
            cached = np.load(feat_cache)
            if cached.shape[0] != N:
                print(
                    f"[tacquad_replication] cache size mismatch for {name} "
                    f"(cached={cached.shape[0]}, expected={N}); re-extracting"
                )
            else:
                print(f"[tacquad_replication] load cached {name}")
                features[name] = cached
                continue
        print(f"[tacquad_replication] extract {name}")
        enc = get_encoder(name)
        feats = extract_features(enc, dataset)
        np.save(feat_cache, feats)
        features[name] = feats
        del enc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = compute_pairwise_metrics(features, k=args.k, n_perms=args.n_perms)
    write_results_csv(records, out_dir / "results.csv")
    print(f"[tacquad_replication] wrote {len(records)} records to {out_dir / 'results.csv'}")

    meta = {
        "N": N,
        "k": args.k,
        "n_perms": args.n_perms,
        "subset": args.subset,
        "sensor": args.sensor,
        "max_samples": args.max_samples,
        "per_object_max": args.per_object_max,
        "encoders": chosen,
        "tacquad_root": args.tacquad_root,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    summary = write_summary(records, out_dir / "summary.txt", meta)
    print()
    print(summary)


if __name__ == "__main__":
    main()
