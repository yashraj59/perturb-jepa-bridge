# PLS Distilled Head Report

- Dataset: `synth_micro`
- Seed: `1`
- Rank: `3`
- Steps: `10`
- Student head: `linear_clone`
- Frozen readout max abs drift: `0.00000000`
- Protected geometry preserved: `True`

## Student Head Before

- pass: `True`
- recall@1: `0.2188`
- RNA latent R2: `0.7563`
- batch balanced accuracy: `0.5625`

## Student Head After

- pass: `True`
- recall@1: `0.2188`
- recall@5: `0.6875`
- RNA latent R2: `0.7563`
- Image latent R2: `0.9379`
- RNA min std: `6.2555`
- Image min std: `17.0461`
- batch balanced accuracy: `0.5625`

## Protected PLS Retrieval Path

- recall@1: `0.2188`
- RNA latent R2: `0.7563`
- batch balanced accuracy: `0.5625`

## Protected Deltas

- `model_rna_to_image_recall@1`: `0.00000000`
- `model_rna_to_image_recall@5`: `0.00000000`
- `model_bio_latent_r2_rna_shared`: `0.00000000`
- `model_bio_latent_r2_image_shared`: `0.00000000`
- `model_rna_shared_min_std`: `0.00000000`
- `model_image_shared_min_std`: `0.00000000`
- `model_batch_probe_balanced_accuracy`: `0.00000000`

## Artifacts

- `STUDENT_BEFORE_METRICS.json`
- `STUDENT_AFTER_METRICS.json`
- `PROTECTED_BEFORE_METRICS.json`
- `PROTECTED_AFTER_METRICS.json`
- `TRAIN_HISTORY.json`
- `prefit_pls_readout.json`
- `pls_distilled_head.pt`
