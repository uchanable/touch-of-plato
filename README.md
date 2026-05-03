# A Touch of Plato

Code, data references, and reproducibility scripts for the paper:

> **A Touch of Plato: Does the Platonic Representation Hypothesis Extend to Tactile Modalities?**

## Status

🚧 Code release in progress. The full pipeline — feature extraction, the
66-pair five-metric alignment matrix, the layer-wise probe, the scale
curve, and figure-generation scripts — will be uploaded here ahead of the
NeurIPS 2026 review period.

## What this repository will contain

- Five-metric alignment benchmark (M1–M5: mutual-$k$NN, debiased CKA,
  null-calibrated $z$, unbiased CKA, orthogonal Procrustes)
- Feature-extraction scripts for 12 pretrained encoders
  (5 vision · 3 language · 4 touch)
- Cached `.npy` feature arrays on the full TVL dataset
  ($N = 43{,}502$) so the alignment matrix can be reproduced
  without re-running the ~8 h feature-extraction step
- Layer-wise probe and scale-curve scripts
- Figure-generation scripts for all main figures and the appendix

## Datasets

All input data are publicly available and used under their original
licenses:

- **TVL** — `mlfu7/Touch-Vision-Language-Dataset` (Apache 2.0)
- **WIT-1024** — `minhuh/prh::wit_1024` subset
- **TacQuad** — released with the AnyTouch paper

## Encoders

Twelve frozen pretrained encoders are evaluated; full registry, sources,
and feature dimensions are in the paper appendix. None of them are
trained or modified in this work.

## License

Planned release license:
- **Code:** MIT
- **Precomputed alignment matrix (CSV):** CC-BY-4.0

## Citation

Citation information will be added upon acceptance.
