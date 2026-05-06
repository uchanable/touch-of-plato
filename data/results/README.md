# `data/results/`

Per-pair and per-block alignment CSVs, plus a single
`ground_truth.json` of paper-headline numbers. Released under
**CC BY 4.0** (see `LICENSE-DATA`).

| File | Source experiment | Paper binding | Rows |
|---|---|---|---:|
| `fig1_perpair_base.csv` | `src.experiments.fig1_alignment_matrix` | Fig. 1 — 10-encoder baseline (45 pairs × 4 metrics) | 180 |
| `fig1_perpair_anytouch.csv` | `src.experiments.anytouch_pairwise` | Fig. 1 — AnyTouch additions (10 pairs × 6 columns; 4 standard metrics + 2 raw null intermediates) | 60 |
| `fig1_perpair_tvl_vitb.csv` | `src.experiments.tvl_vitb_pairwise` | Fig. 1 — TVL-ViT-B additions (11 pairs × 6 columns) | 66 |
| `fig1_procrustes_m5.csv` | `scripts.procrustes_m5` | Fig. 1 — Procrustes M5 (66 pairs × 1 metric) | 66 |
| `metric_consistency.csv` | `src.experiments.tbl2_metric_consistency` | Table tab:metric-consistency (Spearman ρ between metrics) | 4 |
| `fig2_scale.csv` | `src.experiments.fig2_scale_curve` | Fig. 2 — main scale curve (6 pairs × 4 fractions × 3 metrics) | 72 |
| `fig2_scale_extended.csv` | `src.experiments.fig2_extension` | Fig. 2 — 14-pair extension (14 × 4 × 2) | 112 |
| `fig3_layerwise.csv` | `src.experiments.fig1_layerwise` | Fig. 3 — Sparsh 22-pair layer-wise probe (22 × 4 × 2) | 176 |
| `fig3_layerwise_extended.csv` | `src.experiments.fig1_layerwise_extension` | Fig. 3 — AnyTouch + TVL-ViT-B extension (15 × 4 × 2) | 120 |
| `fig4_attribute.csv` | `src.experiments.fig4_attribute` | Fig. 4 — per-attribute alignment | 60 |
| `wit_anchor.csv` | `src.experiments.wit_anchor` | Appendix WIT-1024 cross-check (28 V+L pairs × 4 metrics) | 112 |
| `tacquad.csv` | `src.experiments.fig1_tacquad` | Appendix TacQuad cross-check (45 pairs × 4 metrics) | 180 |
| `sparsh_mode_a_b.csv` | `src.experiments.sparsh_sensitivity` | Appendix Sparsh Mode A vs B sensitivity | 24 |
| `ground_truth.json` | `scripts.compute_ground_truth` | All headline numbers cited in the paper | — |

Aggregated headline numbers in `ground_truth.json` should match the
paper to the last quoted decimal — any drift is a bug. Re-run
`python scripts/compute_ground_truth.py` to regenerate; the script is
idempotent and writes byte-identical output if inputs are unchanged.
