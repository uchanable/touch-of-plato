"""Experiment runners for paper figures and tables.

Each runner module's docstring states its paper binding. The 14 runners
here cover the full reproducibility pipeline (see REPRODUCE.md for the
canonical command sequence and per-figure wall-clock estimates).

Main figures:
- alignment_matrix.py     -> Fig. 1 (12x12 pairwise alignment heatmap, 10-encoder baseline)
- alignment_matrix_anytouch.py         -> Fig. 1 additions (AnyTouch x 10 partners)
- alignment_matrix_tvl_vitb.py         -> Fig. 1 additions (TVL-ViT-B x 11 partners)
- scale_curve.py          -> Fig. 2 (Sparsh size x data fraction)
- scale_curve_extension.py            -> Fig. 2 (14-pair AnyTouch + TVL-ViT-B extension)
- layerwise_probe.py            -> Fig. 3 (layer-wise probe, 22-pair Sparsh-only)
- layerwise_probe_extension.py  -> Fig. 3 (layer-wise probe, AnyTouch + TVL-ViT-B 15-pair extension)
- attribute_alignment.py            -> Fig. 4 (per-attribute alignment)

Cross-checks and tables:
- wit_anchor.py                -> Appendix WIT-1024 cross-dataset replication
- tacquad_replication.py              -> Appendix TacQuad cross-dataset replication
- sparsh_mode_sensitivity.py        -> Appendix Sparsh Mode A vs Mode B sensitivity
- encoders_table.py             -> Tab.~``tab:encoders`` (encoder registry, auto-generated LaTeX)
- metric_consistency.py   -> Table 2 (metric rank correlation)
"""
