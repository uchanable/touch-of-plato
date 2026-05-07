"""Layer-wise m-kNN probe.

Paper binding (LaTeX labels):
    Fig.~``fig:fig3`` (layer-wise probe);
    body §``sec:exp-fig3`` (Layer-wise);
    numerical detail in appendix §``sec:layerwise``
    (Tab.~``tab:layerwise``, Tab.~``tab:layerwise-z``).

Tests alternative hypothesis (b): surface-feature convergence.
If T-V alignment (m-kNN ~ 0.391, full-run) is a low-level effect, it
should be strongest at shallow transformer blocks (Q1) and decay with
depth (Q4).

Design:
  22 pairs:
    - 10 T-V (Sparsh-{dino,ijepa} x 5 vision)
    - 1 T-T (Sparsh-DINO ↔ Sparsh-IJEPA)
    - 10 V-V (C(5,2) of 5 vision encoders)
    - 1 control (ImageNet-supervised ResNet-50 ↔ DINOv2-Base)
  4 quartiles per pair: Q1=n/4, Q2=n/2, Q3=3n/4, Q4=n
  metric: mutual_knn k=10 (only — CKA noise-prone for layer-wise question)
  pool: patch mean (Sparsh-fair)

Usage:
    python -m src.experiments.layerwise_probe \
        --subset all \
        --output-dir experiments/layerwise_probe_full

Output:
    experiments/layerwise_probe_full/features/<encoder>.L<idx>.npy
    experiments/layerwise_probe_full/results.csv
    experiments/layerwise_probe_full/summary.txt
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


ENCODER_BLOCKS = {
    "sparsh_dino_base":    12,
    "sparsh_ijepa_base":   12,
    "dinov2_small":        12,
    "dinov2_base":         12,
    "dinov2_large":        24,
    "clip_l_vision":       24,
    "siglip_base_vision":  12,
}

# Whether the encoder's HF hidden_states[i] has a CLS-like prefix token
# at position 0 that should be dropped before patch mean-pool.
HAS_CLS_PREFIX = {
    "dinov2_small":        True,
    "dinov2_base":         True,
    "dinov2_large":        True,
    "clip_l_vision":       True,
    "siglip_base_vision":  False,  # SigLIP has no CLS
}

T_NAMES = ["sparsh_dino_base", "sparsh_ijepa_base"]
V_NAMES = ["dinov2_small", "dinov2_base", "dinov2_large",
           "clip_l_vision", "siglip_base_vision"]
CONTROL_NAME = "resnet50_in1k"


def quartile_to_abs(n_blocks: int, q: int) -> int:
    """Q1..Q4 -> n/4, n/2, 3n/4, n."""
    return int(round(n_blocks * q / 4))


# ---- HF backbone layer extractor ----------------------------------------

class HFLayerExtractor:
    """Extract patch-mean features at specified layers from a HF ViT backbone."""

    def __init__(self, backbone: nn.Module, layer_indices: list[int],
                 has_cls: bool):
        self.backbone = backbone
        self.layers = layer_indices
        self.has_cls = has_cls

    def __call__(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        # Force config-level setting
        if hasattr(self.backbone, "config"):
            self.backbone.config.output_hidden_states = True
        out = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            # SigLIP-style fallback: directly hook into encoder.layers
            return self._forward_hook_fallback(pixel_values)
        feats: dict[int, torch.Tensor] = {}
        for i in self.layers:
            h = hs[i]  # (B, S, D)
            if self.has_cls and h.shape[1] > 1:
                feats[i] = h[:, 1:, :].mean(dim=1)
            else:
                feats[i] = h.mean(dim=1)
        return feats

    def _forward_hook_fallback(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        """For models like SigLIP whose hidden_states return None even with config set.
        Hook into backbone.encoder.layers directly. Layer index convention:
        layer_idx=i (i>=1) → output of encoder.layers[i-1].
        """
        if not (hasattr(self.backbone, "encoder")
                and hasattr(self.backbone.encoder, "layers")):
            raise RuntimeError(
                f"backbone {type(self.backbone).__name__} returned hidden_states=None "
                f"and has no .encoder.layers for fallback"
            )
        cache: dict[int, torch.Tensor] = {}
        handles = []
        for i in self.layers:
            if i == 0:
                continue
            blk = self.backbone.encoder.layers[i - 1]

            def make_hook(idx):
                def hook(module, inp, out):
                    # SigLIP layer returns tuple (hidden, ...) — take [0]
                    cache[idx] = out[0] if isinstance(out, tuple) else out.detach()
                return hook
            handles.append(blk.register_forward_hook(make_hook(i)))
        try:
            _ = self.backbone(pixel_values=pixel_values, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        feats: dict[int, torch.Tensor] = {}
        for i, t in cache.items():
            # SigLIP has no CLS — patch mean over all tokens
            if self.has_cls and t.shape[1] > 1:
                feats[i] = t[:, 1:, :].mean(dim=1)
            else:
                feats[i] = t.mean(dim=1)
        return feats


# ---- Sparsh forward-hook extractor --------------------------------------

class SparshLayerExtractor:
    """Forward-hook extractor for Sparsh ViT (custom; no HF API).

    Sparsh blocks are 0-indexed in `vit.blocks`; our layer convention is
    1-indexed where layer_idx=1 == output of vit.blocks[0].
    Sparsh has 1 register token at index 0, no CLS. Drop register, mean over patches.
    """

    def __init__(self, vit: nn.Module, layer_indices: list[int]):
        self.vit = vit
        self.layers = [i for i in layer_indices if i >= 1]
        self.cache: dict[int, torch.Tensor] = {}
        self.handles: list = []
        for i in self.layers:
            blk = vit.blocks[i - 1]
            h = blk.register_forward_hook(self._make_hook(i))
            self.handles.append(h)

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, inp, out):
            self.cache[layer_idx] = out.detach()
        return hook

    def __call__(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        self.cache.clear()
        _ = self.vit(pixel_values)
        feats: dict[int, torch.Tensor] = {}
        for i, t in self.cache.items():
            # Drop 1 register token at idx 0 (Sparsh has no CLS).
            patch = t[:, 1:, :]
            feats[i] = patch.mean(dim=1)
        return feats

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---- ResNet-50 control --------------------------------------------------

class ResNet50LayerExtractor:
    """Extract spatial-pooled features at stage outputs of ImageNet-trained R50.

    Quartile mapping: Q1=after layer1, Q2=after layer2, Q3=after layer3,
    Q4=after layer4. Each output is (B, C, H, W); spatial mean -> (B, C).
    """

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        self.model = model
        self.layers = layer_indices
        self.cache: dict[int, torch.Tensor] = {}
        self.handles: list = []
        for q in layer_indices:
            stage = getattr(model, f"layer{q}")
            h = stage.register_forward_hook(self._make_hook(q))
            self.handles.append(h)

    def _make_hook(self, q: int) -> Callable:
        def hook(module, inp, out):
            self.cache[q] = out.detach()
        return hook

    def __call__(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        self.cache.clear()
        _ = self.model(pixel_values)
        feats: dict[int, torch.Tensor] = {}
        for q, t in self.cache.items():
            feats[q] = t.mean(dim=[2, 3])
        return feats

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def load_resnet50_in1k() -> tuple[nn.Module, Callable]:
    """Return (model.eval(), preprocess(PIL)->Tensor)."""
    import torchvision.models as tvm
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    model = tvm.resnet50(weights=weights)
    model.eval()
    tfm = weights.transforms()  # PIL -> Tensor (3, 224, 224), normalized

    def preprocess(pil_image):
        return tfm(pil_image).unsqueeze(0)

    return model, preprocess


# ---- Per-encoder driver -------------------------------------------------

def wrap_for_layers(enc: LoadedEncoder, layer_indices: list[int]):
    """Return extractor object + cleanup callable."""
    name = enc.name
    model = enc.model
    if name.startswith("sparsh_"):
        ext = SparshLayerExtractor(model.vit, layer_indices)
        return ext, ext.close
    if name.startswith("dinov2_"):
        ext = HFLayerExtractor(model.backbone, layer_indices,
                               has_cls=HAS_CLS_PREFIX[name])
        return ext, lambda: None
    if name == "clip_l_vision":
        # CLIPModel; vision_model is the ViT. Use raw 1024-d hidden states
        # (skip projection — it's only at the CLIPModel.get_image_features level).
        ext = HFLayerExtractor(model.clip.vision_model, layer_indices,
                               has_cls=HAS_CLS_PREFIX[name])
        return ext, lambda: None
    if name == "siglip_base_vision":
        ext = HFLayerExtractor(model.siglip.vision_model, layer_indices,
                               has_cls=HAS_CLS_PREFIX[name])
        return ext, lambda: None
    raise KeyError(f"unknown encoder: {name}")


def extract_layerwise(enc: LoadedEncoder, dataset: TVLDataset,
                      layer_indices: list[int]) -> dict[int, np.ndarray]:
    """Run encoder over dataset, returning {layer_idx: (N, d)}."""
    ext, cleanup = wrap_for_layers(enc, layer_indices)
    N = len(dataset)
    feats: dict[int, np.ndarray] | None = None  # lazy init after first sample

    enc.model.eval()
    view_key = "tactile" if enc.modality == "tactile" else "vision"
    try:
        with torch.no_grad():
            for n in tqdm(range(N), desc=f"feat[{enc.name}]", leave=False):
                sample = dataset[n][view_key]
                inp = enc.preprocess(sample)
                out_dict = ext(inp)  # {layer: (1, d)}
                if feats is None:
                    feats = {
                        L: np.zeros((N, t.shape[-1]), dtype=np.float32)
                        for L, t in out_dict.items()
                    }
                for L, v in out_dict.items():
                    feats[L][n] = v.squeeze(0).cpu().numpy().astype(np.float32)
    finally:
        cleanup()
    return feats or {}


def extract_resnet50(dataset: TVLDataset,
                     layer_indices: list[int]) -> dict[int, np.ndarray]:
    """ResNet-50 control on vision view of TVL."""
    model, preprocess = load_resnet50_in1k()
    ext = ResNet50LayerExtractor(model, layer_indices)
    N = len(dataset)
    feats: dict[int, np.ndarray] | None = None
    try:
        with torch.no_grad():
            for n in tqdm(range(N), desc=f"feat[{CONTROL_NAME}]", leave=False):
                pil = dataset[n]["vision"]
                inp = preprocess(pil)
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
    return feats or {}


# ---- Main ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all", choices=["all", "ssvtp", "hct"])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output-dir", default="experiments/layerwise_probe_full")
    ap.add_argument("--data-dir", default="data/tvl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--quartiles", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--skip-control", action="store_true",
                    help="Skip ResNet-50 ImageNet control pair.")
    ap.add_argument("--t-encoders", nargs="+", default=None,
                    help="Override T encoders (default: both Sparsh).")
    ap.add_argument("--v-encoders", nargs="+", default=None,
                    help="Override V encoders (default: 5 vision encoders).")
    ap.add_argument("--include-pair-types", nargs="+", default=None,
                    choices=["T-V", "T-T", "V-V", "Control"],
                    help="Limit pair types (default: all 4).")
    ap.add_argument("--n-perms", type=int, default=100,
                    help="If >0, also compute null-calibrated z-score with this many permutations.")
    ap.add_argument("--null-subsample-N", type=int, default=None,
                    help="If set, subsample to this N for null-calibration (raw too).")
    ap.add_argument("--null-seed", type=int, default=0,
                    help="Random seed for null-calibration permutations + subsample.")
    args = ap.parse_args()

    # Apply overrides
    t_names = args.t_encoders if args.t_encoders else T_NAMES
    v_names = args.v_encoders if args.v_encoders else V_NAMES
    allowed_ptypes = set(args.include_pair_types) if args.include_pair_types \
        else {"T-V", "T-T", "V-V", "Control"}

    out_dir = Path(args.output_dir)
    feats_dir = out_dir / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    print(f"[1/4] Loading TVL ({args.subset}) ...")
    dataset = TVLDataset(
        root=args.data_dir,
        subset=args.subset,
        max_samples=args.max_samples,
    )
    N = len(dataset)
    print(f"      N={N}")

    # 2. Per-encoder layer feature extraction
    # Always include encoders that appear in any allowed pair type
    encoders_to_run: list[str] = []
    if "T-V" in allowed_ptypes or "T-T" in allowed_ptypes:
        encoders_to_run += t_names
    if "T-V" in allowed_ptypes or "V-V" in allowed_ptypes or "Control" in allowed_ptypes:
        for n in v_names:
            if n not in encoders_to_run:
                encoders_to_run.append(n)
    features: dict[tuple[str, int], np.ndarray] = {}

    print(f"[2/4] Extracting features for {len(encoders_to_run)} encoders ...")
    for name in encoders_to_run:
        n_blocks = ENCODER_BLOCKS[name]
        layer_abs = [quartile_to_abs(n_blocks, q) for q in args.quartiles]
        todo = [L for L in layer_abs
                if not (feats_dir / f"{name}.L{L}.npy").exists()]
        if todo:
            print(f"      [{name}] extracting layers {todo} (n_blocks={n_blocks})")
            enc = get_encoder(name)
            extracted = extract_layerwise(enc, dataset, todo)
            for L, arr in extracted.items():
                np.save(feats_dir / f"{name}.L{L}.npy", arr)
                print(f"        saved L{L} shape={arr.shape}")
            del enc, extracted
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print(f"      [{name}] all layers cached")
        for L in layer_abs:
            features[(name, L)] = np.load(feats_dir / f"{name}.L{L}.npy")

    # 3. ResNet-50 control
    do_control = (not args.skip_control) and ("Control" in allowed_ptypes)
    if do_control:
        rn_layers = list(args.quartiles)  # Q1..Q4 == stage 1..4
        rn_todo = [L for L in rn_layers
                   if not (feats_dir / f"{CONTROL_NAME}.L{L}.npy").exists()]
        if rn_todo:
            print(f"[2.5/4] Extracting ResNet-50 control stages {rn_todo} ...")
            extracted = extract_resnet50(dataset, rn_todo)
            for L, arr in extracted.items():
                np.save(feats_dir / f"{CONTROL_NAME}.L{L}.npy", arr)
                print(f"        saved L{L} shape={arr.shape}")
            del extracted
            gc.collect()
        for L in rn_layers:
            features[(CONTROL_NAME, L)] = np.load(
                feats_dir / f"{CONTROL_NAME}.L{L}.npy"
            )

    # 4. Build pair list (filtered by allowed_ptypes)
    pairs: list[tuple[str, str, str]] = []
    if "T-V" in allowed_ptypes:
        pairs += [(t, v, "T-V") for t in t_names for v in v_names]
    if "T-T" in allowed_ptypes and len(t_names) >= 2:
        pairs += [(t_names[i], t_names[j], "T-T")
                  for i in range(len(t_names))
                  for j in range(i + 1, len(t_names))]
    if "V-V" in allowed_ptypes and len(v_names) >= 2:
        pairs += [(v_names[i], v_names[j], "V-V")
                  for i in range(len(v_names))
                  for j in range(i + 1, len(v_names))]
    if do_control:
        ctrl_partner = "dinov2_base" if "dinov2_base" in v_names else v_names[0]
        pairs += [(CONTROL_NAME, ctrl_partner, "Control")]
    print(f"[3/4] Computing layer-matched m-kNN for {len(pairs)} pairs x "
          f"{len(args.quartiles)} quartiles = {len(pairs)*len(args.quartiles)} runs")

    # Optional subsample for null-calibration
    sub_N = args.null_subsample_N
    sub_idx = None
    if sub_N is not None and sub_N < N:
        rng_sub = np.random.default_rng(args.null_seed)
        sub_idx = rng_sub.choice(N, size=sub_N, replace=False)
        sub_idx = np.sort(sub_idx)
        print(f"      [null-cal] subsample to N={sub_N} (seed={args.null_seed})")
    n_eff = sub_N if sub_idx is not None else N

    records: list[dict] = []
    for a, b, ptype in tqdm(pairs, desc="pairs"):
        nb_a = ENCODER_BLOCKS.get(a, 4)
        nb_b = ENCODER_BLOCKS.get(b, 4)
        for q in args.quartiles:
            la = quartile_to_abs(nb_a, q) if a in ENCODER_BLOCKS else q
            lb = quartile_to_abs(nb_b, q) if b in ENCODER_BLOCKS else q
            feat_a = features[(a, la)]
            feat_b = features[(b, lb)]
            if sub_idx is not None:
                feat_a = feat_a[sub_idx]
                feat_b = feat_b[sub_idx]
            Za = torch.from_numpy(feat_a)
            Zb = torch.from_numpy(feat_b)
            val = mutual_knn_alignment(Za, Zb, k=args.k)
            records.append({
                "encoder_a": a, "encoder_b": b, "pair_type": ptype,
                "quartile": q, "layer_a": la, "layer_b": lb,
                "metric": "mutual_knn", "value": float(val),
                "n_samples": n_eff, "k": args.k,
            })
            if args.n_perms > 0:
                z, raw_z, mu_z, sigma_z = null_calibrated_alignment(
                    Za, Zb,
                    base_metric=lambda x, y: mutual_knn_alignment(x, y, k=args.k),
                    n_perms=args.n_perms,
                    seed=args.null_seed,
                    return_raw=True,
                )
                records.append({
                    "encoder_a": a, "encoder_b": b, "pair_type": ptype,
                    "quartile": q, "layer_a": la, "layer_b": lb,
                    "metric": "null_knn_z", "value": float(z),
                    "n_samples": n_eff, "k": args.k,
                })

    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "encoder_a", "encoder_b", "pair_type",
            "quartile", "layer_a", "layer_b",
            "metric", "value", "n_samples", "k",
        ])
        w.writeheader()
        w.writerows(records)
    print(f"[4/4] Saved {csv_path}")

    # Summary by pair_type
    summary_lines: list[str] = []
    for ptype in ["T-V", "T-T", "V-V", "Control"]:
        sub = [r for r in records if r["pair_type"] == ptype]
        if not sub:
            continue
        summary_lines.append(f"\n=== {ptype} (n_pairs={len(sub)//len(args.quartiles)}) ===")
        for q in args.quartiles:
            qsub = [r["value"] for r in sub if r["quartile"] == q]
            summary_lines.append(
                f"  Q{q}: mean={np.mean(qsub):.4f}  "
                f"max={max(qsub):.4f}  min={min(qsub):.4f}  n={len(qsub)}"
            )
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n")

    meta = {
        "N": N, "k": args.k, "quartiles": args.quartiles,
        "encoders": encoders_to_run + ([CONTROL_NAME] if do_control else []),
        "pair_types": sorted(allowed_ptypes),
        "pairs_total": len(pairs),
        "runs_total": len(records),
        "subset": args.subset,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
