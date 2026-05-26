# synth_micro Step 0 Baseline

Seed: `0`

Training steps completed: `30`
Device: `cpu`
Wallclock minutes: `0.741`

## Key Metrics

- Model RNA->image recall@1: `0.0625`
- Random RNA->image recall@1: `0.0625`
- Batch-only recall@1: `0.0625`
- Model counterfactual direction accuracy: `0.6055`
- Source-as-target direction accuracy: `0.0000`
- RNA shared biological latent R2: `-0.6616`
- Batch probe balanced accuracy: `0.4062`
- Embedding rank: `25.0`
- Delta norm ratio: `0.9965`
- Collapse flag: `False`

Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.
