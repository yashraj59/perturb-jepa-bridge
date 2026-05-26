# Frozen PLS JEPA Report

- Dataset: `synth_micro`
- Seed: `1`
- Rank: `3`
- Steps: `50`
- Protected geometry preserved: `True`
- Frozen readout max abs drift: `0.00000000`

## Before Training

- pass: `True`
- recall@1: `0.2188`
- RNA latent R2: `0.7563`
- batch balanced accuracy: `0.5625`

## After Training

- pass: `True`
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

- `BEFORE_METRICS.json`
- `AFTER_METRICS.json`
- `TRAIN_HISTORY.json`
- `prefit_pls_readout.json`
- `frozen_pls_jepa.pt`
