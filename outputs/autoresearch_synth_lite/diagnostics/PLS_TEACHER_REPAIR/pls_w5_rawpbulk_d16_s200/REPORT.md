# PLS Teacher Repair Report

- Dataset: `synth_micro`
- Seed: `0`
- Teacher: `pls`
- Teacher weight: `5.0`
- Model dim: `16`
- Steps: `200`
- Tier 1 pass: `False`

## Teacher Probe

- recall@1: `0.2188`
- RNA latent R2: `0.3094`
- Image latent R2: `0.9827`

## Model Result

- collapse flag: `True`
- recall@1: `0.2500`
- recall@5: `0.3750`
- RNA latent R2: `-0.3037`
- Image latent R2: `0.7259`
- RNA min std: `0.1514`
- Image min std: `0.0010`
- Batch balanced accuracy: `0.5625`

## Artifacts

- `MODEL_METRICS.json`
- `TEACHER_PROBE_METRICS.json`
- `TRAIN_HISTORY.json`
- `bridge_config.json`
