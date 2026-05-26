# synth_dose_extrapolation_lite Step 0 Baseline

Seed: `0`

Training steps completed: `5`
Device: `cpu`
Wallclock minutes: `2.291`

## Key Metrics

- Model RNA->image recall@1: `0.0185`
- Random RNA->image recall@1: `0.0123`
- Batch-only recall@1: `0.0093`
- Model counterfactual direction accuracy: `0.5198`
- Source-as-target direction accuracy: `0.0000`
- RNA shared biological latent R2: `0.3820`
- Batch probe balanced accuracy: `0.2840`
- Embedding rank: `26.0`
- Delta norm ratio: `0.5903`
- Collapse flag: `True`

Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.
