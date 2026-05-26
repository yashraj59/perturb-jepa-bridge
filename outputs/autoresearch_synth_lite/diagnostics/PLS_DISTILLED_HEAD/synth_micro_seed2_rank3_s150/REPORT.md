# PLS Distilled Head Report

- Dataset: `synth_micro`
- Seed: `2`
- Rank: `3`
- Steps: `150`
- Frozen readout max abs drift: `0.00000000`
- Protected geometry preserved: `True`

## Student Head Before

- pass: `False`
- recall@1: `0.0938`
- RNA latent R2: `-0.1675`
- batch balanced accuracy: `0.4375`

## Student Head After

- pass: `False`
- recall@1: `0.0625`
- recall@5: `0.5312`
- RNA latent R2: `0.4903`
- Image latent R2: `0.4810`
- RNA min std: `0.6046`
- Image min std: `0.4415`
- batch balanced accuracy: `0.3750`

## Protected PLS Retrieval Path

- recall@1: `0.2812`
- RNA latent R2: `0.6069`
- batch balanced accuracy: `0.3750`

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
