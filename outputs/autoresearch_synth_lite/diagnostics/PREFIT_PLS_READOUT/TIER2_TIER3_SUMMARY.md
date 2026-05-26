# Prefit PLS Linear Readout Tier 2/Tier 3 Summary

## Candidate

- Mechanism: train-split-only closed-form PLS cross-covariance readout.
- Rank: `3`.
- RNA readout: `raw_linear_pseudobulk`.
- Image readout: `raw_linear_pooled`.
- Training steps: `0`; this is a fitted linear initializer/evaluator, not an SGD-trained JEPA candidate.
- Real data used: `false`.
- GPU used: `false`.

The rank-3 setting was chosen after rank-4 passed seed 0/1 but failed the seed-2 batch-leakage gate. Rank 3 preserved retrieval and biological latent recovery while keeping the batch probe inside the gate.

## Tier 2: `synth_micro`, Seeds 0/1/2

| seed | pass | collapse | RNA->image R@1 | RNA->image R@5 | RNA latent R2 | image latent R2 | RNA min std | image min std | batch bal acc | batch majority |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0.2188 | 0.5000 | 0.4155 | 0.8876 | 8.8061 | 23.2078 | 0.5000 | 0.5000 |
| 1 | 1 | 0 | 0.2188 | 0.6875 | 0.7563 | 0.9379 | 6.2555 | 17.0461 | 0.5625 | 0.5000 |
| 2 | 1 | 0 | 0.2812 | 0.8125 | 0.6069 | 0.9146 | 12.1573 | 19.2276 | 0.3750 | 0.5000 |

Tier 2 means:

- RNA->image recall@1: `0.2396 +/- 0.0295`
- RNA->image recall@5: `0.6667 +/- 0.1284`
- RNA latent R2: `0.5929 +/- 0.1395`
- Image latent R2: `0.9134 +/- 0.0206`
- Batch balanced accuracy: `0.4792 +/- 0.0780`

## Tier 3: No-Regression Validators

| dataset | seed | pass | collapse | RNA->image R@1 | RNA latent R2 | image latent R2 | batch bal acc | batch majority |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `synth_easy_lite` | 0 | 1 | 0 | 0.2292 | 0.7821 | 0.8712 | 0.3750 | 0.5000 |
| `synth_batch_confound_lite` | 0 | 1 | 0 | 0.1746 | 0.6945 | 0.7538 | 0.3296 | 0.3571 |

## Decision

This candidate passes Tier 2 and the focused Tier 3 synthetic validators. It should become the current synthetic model-of-record mechanism for the low-compute loop.

Important caveat: this is not a trained JEPA improvement yet. It is a closed-form low-rank shared readout that fixes the condition-level bridge geometry. The next architectural work should preserve this fitted readout as an initializer or frozen safety baseline while reintroducing JEPA training around it.

## Artifacts

- `seed0_dim3_unscaled/`
- `seed1_dim3_unscaled/`
- `seed2_dim3_unscaled/`
- `tier3_synth_easy_seed0_dim3/`
- `tier3_synth_batch_confound_seed0_dim3/`
