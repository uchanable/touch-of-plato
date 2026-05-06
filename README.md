# A Touch of Plato

Code, alignment-matrix CSVs, and reproducibility scripts for the
NeurIPS 2026 submission **"A Touch of Plato: Does the Platonic
Representation Hypothesis Extend to Tactile Modalities?"** (Submission
\#18965). Authors anonymized for double-blind review.

## What this repo contains

A 12-encoder, 66-pair, 5-metric alignment benchmark on the public
Touch-Vision-Language (TVL) corpus.

- **Encoders (12, all frozen, no fine-tuning):**
  5 vision (DINOv2-S/B/L, CLIP-L vision, SigLIP-B vision),
  3 language (CLIP-L text, SigLIP-B text, all-mpnet-base-v2),
  4 tactile (Sparsh-DINO, Sparsh-IJEPA, AnyTouch, TVL-ViT-B).
- **Metrics (5):**
  M1 mutual-kNN · M2 debiased CKA · M3 null-calibrated kNN-z ·
  M4 unbiased CKA (cross-check via [`platonic-rep`](https://github.com/minyoungg/platonic-rep)) ·
  M5 orthogonal Procrustes.
- **Released artefacts:**
  - 13 per-pair CSVs covering Fig. 1 / 2 / 3 / 4 of the paper plus the
    metric-consistency table, the WIT cross-check, the TacQuad
    cross-check, and the Sparsh Mode-A/B sensitivity (`data/results/`).
  - `data/results/ground_truth.json` — paper headline numbers
    aggregated from those CSVs (T-T 0.380, T-V 0.375, V-L 0.027,
    T-V / V-L = 14×).

## Quickstart

```bash
# 1. Install pinned deps (Python 3.12 recommended)
pip install -r requirements.txt

# 2. Clone Sparsh / platonic-rep / correcting_CKA_alignment into third_party/
bash scripts/setup_third_party.sh

# 3. Download encoder checkpoints (~several GB)
python scripts/download_checkpoints.py

# 4. Verify the released CSVs aggregate to the paper headline numbers
python scripts/compute_ground_truth.py
# -> writes data/results/ground_truth.json

# 5. (optional) Quick sanity check on a 500-sample TVL slice (~5 min on a laptop CPU)
python scripts/sanity_test.py
```

To re-extract features from scratch and recompute the full 66-pair
matrix, see [`REPRODUCE.md`](./REPRODUCE.md).

## Repo layout

```
src/
  alignment_metrics/   # M1 mutual-kNN, M2 dCKA, M3 null-cal z, M4 uCKA
  encoders/            # 12 frozen encoder loaders (Table 1)
  datasets/            # TVL + TacQuad
  experiments/         # Per-experiment runners (alignment_matrix, scale_curve, layerwise_probe,
                       #   attribute_alignment, tacquad_replication, wit_anchor,
                       #   sparsh_mode_sensitivity, encoders_table, metric_consistency)
scripts/
  compute_ground_truth.py
  procrustes_m5.py
  download_checkpoints.py
  setup_third_party.sh
configs/
  encoders.yaml        # Table 1 registry (HF id, d, modality, loader path)
  hyperparams.yaml     # k=10, B=100, seed=0, ...
data/
  results/             # 13 per-pair CSVs + ground_truth.json
tests/                 # pytest sanity tests for metrics + encoders
```

## Hardware note

The released CSVs were produced on a Mac Studio (Apple Silicon M2
Ultra). For the camera-ready version, the same pipeline will be
re-run on cloud GPUs (RunPod L4) and the CSVs will be regenerated.
Anyone reproducing this work on a CUDA box should expect identical
results up to floating-point determinism.

## License

- **Code:** MIT (see [`LICENSE`](./LICENSE)).
- **Released CSVs (`data/results/`):** CC BY 4.0 (see
  [`LICENSE-DATA`](./LICENSE-DATA)).
- **Sparsh and AnyTouch checkpoints** are downloaded from their
  upstream sources; they are not redistributed here and remain under
  their own licenses ([`LICENSE-Sparsh`](./LICENSE-Sparsh),
  [`LICENSE-AnyTouch`](./LICENSE-AnyTouch)).
