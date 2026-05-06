# `data/results/`

Per-pair and per-block alignment CSVs, plus a single
`ground_truth.json` of paper-headline numbers. Released under
**CC BY 4.0** (see `LICENSE-DATA`).

| File | Source experiment | Paper binding | Rows |
|---|---|---|---:|
| `fig1_perpair_base.csv` | `src.experiments.fig1_alignment_matrix` | Fig. 1 — 10-encoder baseline (45 pairs × 4 metrics) | TBD |
| `fig1_perpair_anytouch.csv` | `src.experiments.anytouch_pairwise` | Fig. 1 — AnyTouch additions (15 pairs) | TBD |
| `fig1_perpair_tvl_vitb.csv` | `src.experiments.tvl_vitb_pairwise` | Fig. 1 — TVL-ViT-B additions (11 pairs) | TBD |
| `fig1_procrustes_m5.csv` | `scripts.procrustes_m5` | Fig. 1 — Procrustes M5 (66 pairs) | TBD |
| `metric_consistency.csv` | `src.experiments.tbl2_metric_consistency` | Table tab:metric-consistency (Spearman ρ between metrics) | TBD |
| `fig2_scale.csv` | `src.experiments.fig2_scale_curve` | Fig. 2 — main scale curve | TBD |
| `fig2_scale_extended.csv` | `src.experiments.fig2_extension` | Fig. 2 — 14-pair extension | TBD |
| `fig3_layerwise.csv` | `src.experiments.fig1_layerwise` | Fig. 3 — Sparsh 22-pair layer-wise probe | TBD |
| `fig3_layerwise_extended.csv` | `src.experiments.fig1_layerwise_extension` | Fig. 3 — AnyTouch + TVL-ViT-B extension | TBD |
| `fig4_attribute.csv` | `src.experiments.fig4_attribute` | Fig. 4 — per-attribute alignment | TBD |
| `wit_anchor.csv` | `src.experiments.wit_anchor` | Appendix WIT cross-check | TBD |
| `tacquad.csv` | `src.experiments.fig1_tacquad` | Appendix TacQuad cross-check | TBD |
| `sparsh_mode_a_b.csv` | `src.experiments.sparsh_sensitivity` | Appendix Sparsh Mode A vs B | TBD |
| `ground_truth.json` | `scripts.compute_ground_truth` | All headline numbers cited in the paper | — |

Row counts and headline numbers are populated by
`scripts/compute_ground_truth.py`. After running it,
the values in `ground_truth.json` should match the paper to the last
quoted decimal — any drift is a bug.
