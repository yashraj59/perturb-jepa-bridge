# synth_easy_lite Step 0 Baseline

Seed: `0`

Training steps completed: `5`
Device: `cpu`
Wallclock minutes: `0.710`

## Key Metrics

- Model RNA->image recall@1: `0.0208`
- Random RNA->image recall@1: `0.0104`
- Batch-only recall@1: `0.0208`
- Model counterfactual direction accuracy: `0.5278`
- Source-as-target direction accuracy: `0.0000`
- RNA shared biological latent R2: `0.5158`
- Batch probe balanced accuracy: `0.4167`
- Embedding rank: `23.0`
- Delta norm ratio: `0.6264`
- Collapse flag: `True`

Plain autoencoder baseline: skipped in Step 0 runner because adding and validating a separate trainable baseline would consume time better spent on the protocol-mandated JEPA baseline and negative controls.
