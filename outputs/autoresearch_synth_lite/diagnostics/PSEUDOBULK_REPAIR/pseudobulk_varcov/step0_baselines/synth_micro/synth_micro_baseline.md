# synth_micro Step 0 Baseline

Seed: `0`

Training steps completed: `30`
Device: `cpu`
Wallclock minutes: `0.756`

## Key Metrics

- Model RNA->image recall@1: `0.0625`
- Random RNA->image recall@1: `0.0625`
- Batch-only recall@1: `0.0625`
- Model counterfactual direction accuracy: `0.6218`
- Source-as-target direction accuracy: `0.0000`
- RNA shared biological latent R2: `0.1088`
- Batch probe balanced accuracy: `0.4375`
- Embedding rank: `5.0`
- Delta norm ratio: `1.5553`
- Collapse flag: `True`

Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.
