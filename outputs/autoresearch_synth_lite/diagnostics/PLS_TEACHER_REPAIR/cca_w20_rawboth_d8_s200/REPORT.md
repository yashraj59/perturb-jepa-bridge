# PLS Teacher Repair Report

- Dataset: `synth_micro`
- Seed: `0`
- Teacher: `regularized_cca`
- Teacher weight: `20.0`
- Model dim: `8`
- Steps: `200`
- Tier 1 pass: `False`

## Teacher Probe

- recall@1: `0.1562`
- RNA latent R2: `0.3392`
- Image latent R2: `0.7738`

## Model Result

- collapse flag: `False`
- recall@1: `0.0938`
- recall@5: `0.3438`
- RNA latent R2: `-0.0405`
- Image latent R2: `0.7092`
- RNA min std: `0.2032`
- Image min std: `0.2556`
- Batch balanced accuracy: `0.5000`

## Artifacts

- `MODEL_METRICS.json`
- `TEACHER_PROBE_METRICS.json`
- `TRAIN_HISTORY.json`
- `bridge_config.json`
