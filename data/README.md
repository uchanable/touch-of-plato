# `data/`

This directory hosts only the small, derivative artefacts that map
directly onto numbers in the paper. Large raw inputs (TVL, WIT,
TacQuad) are downloaded via `scripts/download_*.sh` to `data/tvl/`,
`data/wit/`, `data/tacquad/` (all gitignored).

```
data/
├── results/              # 12 CSVs + ground_truth.json (CC-BY-4.0)
├── tvl/                  # gitignored, ~75 GB, see scripts/download_tvl.sh
├── wit/                  # gitignored, see scripts/download_wit.sh
└── tacquad/              # gitignored, see scripts/download_tacquad.sh
```

## Tier policy

- **Tier 2 release (this repo).** All per-pair × per-metric results are
  shipped as CSVs under `data/results/`, plus a `ground_truth.json`
  summary that aggregates the 12-encoder / 66-pair × 5-metric matrix
  into the 6 modality blocks (V-V / L-L / T-T / V-L / T-V / L-T).
- **Tier 3 (deferred).** Cached `.npy` feature tensors (~tens of GB on
  the full TVL split) are NOT shipped here. They will be uploaded to a
  long-term host post-acceptance. Until then, features must be
  extracted locally; see `REPRODUCE.md`.
