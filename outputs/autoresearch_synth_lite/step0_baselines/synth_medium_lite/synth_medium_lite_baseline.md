# synth_medium_lite Step 0 Baseline

Seed: `0`

Training steps completed: `5`
Device: `cpu`
Wallclock minutes: `2.355`

## Key Metrics

- Model RNA->image recall@1: `0.0154`
- Random RNA->image recall@1: `0.0123`
- Batch-only recall@1: `0.0093`
- Model counterfactual direction accuracy: `0.4999`
- Source-as-target direction accuracy: `0.0000`
- RNA shared biological latent R2: `0.3797`
- Batch probe balanced accuracy: `0.2593`
- Embedding rank: `26.0`
- Delta norm ratio: `0.5923`
- Collapse flag: `True`

Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.
