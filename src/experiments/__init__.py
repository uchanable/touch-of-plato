"""Experiment runners for paper figures and tables.

Each runner module's docstring states its paper binding. The 14 runners
here cover the full reproducibility pipeline (see REPRODUCE.md for the
canonical command sequence and per-figure wall-clock estimates).

Main figures:
- fig1_alignment_matrix.py     -> Fig. 1 (12x12 pairwise alignment heatmap, 10-encoder baseline)
- anytouch_pairwise.py         -> Fig. 1 additions (AnyTouch x 10 partners)
- tvl_vitb_pairwise.py         -> Fig. 1 additions (TVL-ViT-B x 11 partners)
- fig2_scale_curve.py          -> Fig. 2 (Sparsh size x data fraction)
- fig2_extension.py            -> Fig. 2 (14-pair AnyTouch + TVL-ViT-B extension)
- fig1_layerwise.py            -> Fig. 3 (layer-wise probe, 22-pair Sparsh-only)
- fig1_layerwise_extension.py  -> Fig. 3 (layer-wise probe, AnyTouch + TVL-ViT-B 15-pair extension)
- fig4_attribute.py            -> Fig. 4 (per-attribute alignment)

Cross-checks and tables:
- wit_anchor.py                -> Appendix WIT-1024 cross-dataset replication
- fig1_tacquad.py              -> Appendix TacQuad cross-dataset replication
- sparsh_sensitivity.py        -> Appendix Sparsh Mode A vs Mode B sensitivity
- tbl1_encoders.py             -> Table 1 (encoder registry, auto-generated LaTeX)
- tbl2_metric_consistency.py   -> Table 2 (metric rank correlation)
"""
