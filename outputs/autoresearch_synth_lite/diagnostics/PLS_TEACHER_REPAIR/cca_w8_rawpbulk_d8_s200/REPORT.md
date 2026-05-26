# PLS Teacher Repair Report

- Dataset: `synth_micro`
- Seed: `0`
- Teacher: `regularized_cca`
- Teacher weight: `8.0`
- Model dim: `8`
- Steps: `200`
- Tier 1 pass: `False`

## Teacher Probe

- recall@1: `0.1562`
- RNA latent R2: `0.3392`
- Image latent R2: `0.7738`

## Model Result

- collapse flag: `False`
- recall@1: `0.0625`
- recall@5: `0.3750`
- RNA latent R2: `0.0174`
- Image latent R2: `0.7393`
- RNA min std: `0.2821`
- Image min std: `0.3487`
- Batch balanced accuracy: `0.3125`

## Artifacts

- `MODEL_METRICS.json`
- `TEACHER_PROBE_METRICS.json`
- `TRAIN_HISTORY.json`
- `bridge_config.json`
