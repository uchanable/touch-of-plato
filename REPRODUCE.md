# Reproducing the paper

This document covers everything needed to regenerate
`data/results/*.csv` and `data/results/ground_truth.json` from
scratch. Released CSVs were produced on a Mac Studio (Apple Silicon
M2 Ultra, 64 GB unified memory). Wall-clock numbers below are
order-of-magnitude on that hardware; a single L4 / A100 should be
faster than the 32-bit Apple-Silicon baseline for everything except
the I/O-bound dataset scan.

## 0. Set up the environment (~10 min, one time)

```bash
# Python 3.12 is recommended (3.12.13 is what we tested with).
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Clone Sparsh (CC BY-NC 4.0) + platonic-rep (MIT) + correcting_CKA_alignment
# into third_party/. This also wires the .pth so `from metrics import
# AlignmentMetrics` works for the M4 unbiased-CKA cross-check.
bash scripts/setup_third_party.sh

# Encoder checkpoints (~several GB of HuggingFace + AnyTouch + TVL-ViT-B).
python scripts/download_checkpoints.py

# Optional but recommended: run the metric-level pytest suite.
pytest tests/
```

Expected sizes:

| Item | Disk |
|---|---:|
| Encoder checkpoints (`checkpoints/`) | ~6 GB |
| Third-party clones (`third_party/`) | ~50 MB |
| TVL dataset (`data/tvl/`)            | ~75 GB |
| WIT subset (`data/wit_1024/`)        | ~2 GB |
| TacQuad (`data/tacquad/`)            | ~5 GB |

## 1. Verify the released CSVs (~30 sec, no GPU)

This is the fastest possible reproduction: it just aggregates the 13
shipped CSVs into `ground_truth.json` and prints the headline numbers
that should match the paper.

```bash
python scripts/compute_ground_truth.py
```

Expected console tail:

```
m-kNN means:  V-V=0.5971  L-L=0.5582  T-T=0.3803
              V-L=0.0268  T-V=0.3749  L-T=0.0370
T-V / V-L  = 13.99x
cross/within (mean m-kNN) = 0.2857
```

If your numbers diverge from these by more than ~1e-4, please file an
issue — the released CSVs are the single source of truth.

## 2. Sanity test on a 500-sample TVL slice (~5 min)

```bash
# Requires data/tvl/ to be downloaded (see scripts/download_tvl.sh).
python scripts/sanity_test.py
```

This loads two encoders (one vision, one tactile), extracts features
on 500 TVL samples, and computes a single mutual-kNN value. The point
is to verify the encoder loaders and the dataset loader on a tiny
slice before you commit to a multi-hour full run.

## 3. Fig. 1 — full 12-encoder, 66-pair, 5-metric matrix

**Cost estimate:** ~8 h end-to-end on Mac Studio (the bulk is
feature extraction over the full TVL split, N=43,502).

### 3a. 10-encoder baseline (45 pairs × 4 metrics)

The Fig. 1 baseline uses the 10 encoders that existed before AnyTouch
and TVL-ViT-B were added; without the explicit `--encoders` filter the
runner uses all 12 encoders in the registry (which would produce a
66-pair CSV, not the 45-pair baseline expected by `compute_ground_truth.py`).

```bash
python -m src.experiments.alignment_matrix \
    --subset all \
    --k 10 --n-perms 100 \
    --output-dir experiments/alignment_matrix_full \
    --encoders dinov2_small dinov2_base dinov2_large \
               clip_l_vision clip_l_text \
               siglip_base_vision siglip_base_text mpnet \
               sparsh_dino_base sparsh_ijepa_base
```

Outputs `experiments/alignment_matrix_full/results.csv` (= `data/results/alignment_matrix_base.csv`)
plus the encoder feature `.npy`s under
`experiments/alignment_matrix_full/features/`. Re-runs are cheap once the features
are cached.

### 3b. AnyTouch additions (10 new pairs × 4 metrics)

`alignment_matrix_anytouch.py` does NOT extract features itself — it expects
each encoder's `(N, d)` matrix to already be cached as
`experiments/anytouch_full/features/<id>.npy`. Run the extraction step
first, which writes both the AnyTouch feature `.npy` and copies of the
10 baseline-encoder features into the same directory:

```bash
# (One-time) Extract AnyTouch features over full TVL.
python -m src.extract_anytouch_features \
    --tvl-root data/tvl \
    --subset all \
    --output-dir experiments/anytouch_full

# Then compute the 10 new pairs (AnyTouch × the 10 baseline encoders):
python -m src.experiments.alignment_matrix_anytouch \
    --features-dir experiments/anytouch_full/features \
    --output-dir experiments/anytouch_full
```

Outputs `experiments/anytouch_full/results.csv`
(= `data/results/alignment_matrix_anytouch.csv`).

### 3c. TVL-ViT-B additions (11 new pairs × 4 metrics)

Same two-step pattern as §3b: extract first, then pair.

```bash
# (One-time) Extract TVL-ViT-B features over full TVL.
python -m src.extract_tvl_vitb_features \
    --tvl-root data/tvl \
    --subset all \
    --output-dir experiments/tvl_vitb_full

# Then compute the 11 new pairs (TVL-ViT-B × {10 baseline + AnyTouch}):
python -m src.experiments.alignment_matrix_tvl_vitb \
    --features-dir experiments/tvl_vitb_full/features \
    --output-dir experiments/tvl_vitb_full
```

Outputs `experiments/tvl_vitb_full/results.csv`
(= `data/results/alignment_matrix_tvl_vitb.csv`).

### 3d. Procrustes M5 (66 pairs × 1 metric)

After 3a-c have produced their `features/*.npy`:

```bash
python scripts/procrustes_m5.py
```

Outputs `experiments/alignment_matrix_full/procrustes_m5.csv`
(= `data/results/alignment_matrix_procrustes_m5.csv`).

### 3e. Aggregate to ground_truth.json

```bash
# Copy the four CSVs into data/results/ if they were written elsewhere
cp experiments/alignment_matrix_full/results.csv         data/results/alignment_matrix_base.csv
cp experiments/anytouch_full/results.csv     data/results/alignment_matrix_anytouch.csv
cp experiments/tvl_vitb_full/results.csv     data/results/alignment_matrix_tvl_vitb.csv
cp experiments/alignment_matrix_full/procrustes_m5.csv   data/results/alignment_matrix_procrustes_m5.csv
python scripts/compute_ground_truth.py
```

## 4. Fig. 2 — scale curve (Sparsh size × data fraction, ~2 h)

```bash
python -m src.experiments.scale_curve \
    --subset all --k 10 --n-perms 100 \
    --output-dir experiments/scale_curve_full
```

Outputs `experiments/scale_curve_full/results.csv` (= `data/results/scale_curve.csv`).

The 14-pair extension that adds AnyTouch and TVL-ViT-B fractions is
in `src.experiments.scale_curve_extension`; its output maps to
`data/results/scale_curve_extension.csv`.

## 5. Fig. 3 — layer-wise probe (~3 h)

```bash
python -m src.experiments.layerwise_probe --subset all
```

Output maps to `data/results/layerwise_probe.csv` (Sparsh 22-pair) and
`data/results/layerwise_probe_extension.csv` (AnyTouch + TVL-ViT-B
extension), the latter via `src.experiments.layerwise_probe_extension`.

## 6. Fig. 4 — per-attribute alignment (~1 h, reuses Fig. 1 features)

```bash
python -m src.experiments.attribute_alignment \
    --subset all --k 10 \
    --features-from experiments/alignment_matrix_full/features \
    --output-dir experiments/attribute_alignment_full
```

Outputs `experiments/attribute_alignment_full/results.csv` (= `data/results/attribute_alignment.csv`).

## 7. WIT cross-check (~1 h)

```bash
python -m src.experiments.wit_anchor --data-dir data/wit_1024
```

Output -> `data/results/wit_anchor.csv`.

## 8. TacQuad cross-check (~1 h)

```bash
python -m src.experiments.tacquad_replication
```

Output -> `data/results/tacquad_replication.csv`.

## 9. Sparsh Mode-A vs Mode-B sensitivity (~30 min)

```bash
python -m src.experiments.sparsh_mode_sensitivity \
    --output-dir experiments/sparsh_mode_sensitivity
```

Output -> `data/results/sparsh_mode_sensitivity.csv`.

## 10. Metric consistency table (~10 sec, post-hoc)

After Fig. 1 has been computed:

```bash
python -m src.experiments.metric_consistency \
    --input experiments/alignment_matrix_full/results.csv \
    --output-dir experiments/metric_consistency_full
```

Output -> `data/results/metric_consistency.csv`.

## Troubleshooting

- **`ImportError: platonic-rep not available`**: `setup_third_party.sh`
  did not finish. Re-run it; the script writes a `.pth` file under
  `.venv/lib/python3.12/site-packages/` that makes
  `from metrics import AlignmentMetrics` resolvable.
- **`FileNotFoundError: data/tvl/...`**: TVL is not bundled; download
  it from `mlfu7/Touch-Vision-Language-Dataset` and place under
  `data/tvl/` (or override with `TVL_ROOT=...`).
- **AnyTouch checkpoint errors**: AnyTouch ships its weights as a
  Google Drive download. `scripts/download_checkpoints.py` handles
  this via `gdown`; on rate-limit hits, retry after a few minutes.
- **Different numbers**: most likely cause is partial TVL download
  (`N != 43502`). Check `len(dataset)` from a dry run.
