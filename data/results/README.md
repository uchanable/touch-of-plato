# `data/results/`

Per-pair and per-block alignment CSVs, plus a single
`ground_truth.json` of paper-headline numbers. Released under
**CC BY 4.0** (see `LICENSE-DATA`).

| File | Source experiment | Paper binding | Rows |
|---|---|---|---:|
| `alignment_matrix_base.csv` | `src.experiments.alignment_matrix` | Fig. 1 — 10-encoder baseline (45 pairs × 4 metrics) | 180 |
| `alignment_matrix_anytouch.csv` | `src.experiments.alignment_matrix_anytouch` | Fig. 1 — AnyTouch additions (10 pairs × 6 columns; 4 standard metrics + 2 raw null intermediates) | 60 |
| `alignment_matrix_tvl_vitb.csv` | `src.experiments.alignment_matrix_tvl_vitb` | Fig. 1 — TVL-ViT-B additions (11 pairs × 6 columns) | 66 |
| `alignment_matrix_procrustes_m5.csv` | `scripts.procrustes_m5` | Fig. 1 — Procrustes M5 (66 pairs × 1 metric) | 66 |
| `metric_consistency.csv` | `src.experiments.metric_consistency` | Table tab:metric-consistency (Spearman ρ between metrics) | 4 |
| `scale_curve.csv` | `src.experiments.scale_curve` | Fig. 2 — main scale curve (6 pairs × 4 fractions × 3 metrics) | 72 |
| `scale_curve_extension.csv` | `src.experiments.scale_curve_extension` | Fig. 2 — 14-pair extension (14 × 4 × 2) | 112 |
| `layerwise_probe.csv` | `src.experiments.layerwise_probe` | Fig. 3 — Sparsh 22-pair layer-wise probe (22 × 4 × 2) | 176 |
| `layerwise_probe_extension.csv` | `src.experiments.layerwise_probe_extension` | Fig. 3 — AnyTouch + TVL-ViT-B extension (15 × 4 × 2) | 120 |
| `attribute_alignment.csv` | `src.experiments.attribute_alignment` | Fig. 4 — per-attribute alignment | 60 |
| `wit_anchor.csv` | `src.experiments.wit_anchor` | Appendix WIT-1024 cross-check (28 V+L pairs × 4 metrics) | 112 |
| `tacquad_replication.csv` | `src.experiments.tacquad_replication` | Appendix TacQuad cross-check (45 pairs × 4 metrics) | 180 |
| `sparsh_mode_sensitivity.csv` | `src.experiments.sparsh_mode_sensitivity` | Appendix Sparsh Mode A vs B sensitivity | 24 |
| `ground_truth.json` | `scripts.compute_ground_truth` | All headline numbers cited in the paper | — |

Aggregated headline numbers in `ground_truth.json` should match the
paper to the last quoted decimal — any drift is a bug. Re-run
`python scripts/compute_ground_truth.py` to regenerate; the script is
idempotent and writes byte-identical output if inputs are unchanged.
