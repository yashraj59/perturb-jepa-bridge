# BioTech-JEPA Model Card

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Device: `cuda`
- Steps: `20`
- Model role: diagnostic Tier 1 candidate only; protected PLS remains model of record
- Latent contract: `z_bio` is used for JEPA retrieval/cross-modal/transition losses; `z_tech` is used for technical batch allocation when batch labels exist.
- Norman metadata amendment: batch and chemical dose are ignored for Norman because the processed h5ad does not expose batch and `dose_val` is guide-count notation.
- Transition cosine improvement: `0.0161`
- RNA->image recall@1: `0.1875`
- RNA-only diagnostic: `0`
