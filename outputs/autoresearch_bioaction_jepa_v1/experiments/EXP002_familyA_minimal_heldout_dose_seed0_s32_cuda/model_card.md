# BioAction-JEPA Model Card

- Model name: `BioAction-JEPA minimal synthetic synth_dose_extrapolation_lite`
- Dataset: `synth_dose_extrapolation_lite`
- Eval split: `test_heldout_dose`
- Training data: synthetic only
- JEPA tasks active: intra-modal, cross-modal, and transition latent prediction
- Teacher/target design: EMA RNA and image target encoders with detached latent targets
- Raw reconstruction weight: `0.0`
- PLS usage: not used in the main BioAction-JEPA path
- Tier reached: implementation smoke/Tier 0 candidate until gates are run
- RNA->image recall@1: `0.0625`
- Image->RNA recall@1: `0.0312`
