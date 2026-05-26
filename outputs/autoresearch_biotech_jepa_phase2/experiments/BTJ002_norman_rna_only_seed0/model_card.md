# BioTech-JEPA Model Card

- Dataset: `norman`
- Eval split: `test`
- Device: `cuda`
- Steps: `10`
- Model role: diagnostic Tier 1 candidate only; protected PLS remains model of record
- Latent contract: `z_bio` is used for JEPA retrieval/cross-modal/transition losses; `z_tech` is used for technical batch allocation when batch labels exist.
- Norman metadata amendment: batch and chemical dose are ignored for Norman because the processed h5ad does not expose batch and `dose_val` is guide-count notation.
- Transition cosine improvement: `0.0313`
- RNA->image recall@1: `nan`
- RNA-only diagnostic: `1`
