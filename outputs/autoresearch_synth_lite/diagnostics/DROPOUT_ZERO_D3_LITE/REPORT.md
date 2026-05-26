# D3-Lite No-Dropout Anti-Collapse Report

## Summary

Four CPU-only `synth_micro` checks were run for 30 steps each:

| Run | Collapse | RNA min std | Image min std | Recall@1 | RNA latent R2 | Image latent R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attention_dropout0 | 1 | 0.009886 | 0.004790 | 0.0625 | -0.8553 | 0.8895 |
| mean_dropout0 | 1 | 0.010609 | 0.005219 | 0.0625 | -0.2498 | 0.9045 |
| attention_dropout0_varcov | 1 | 0.066800 | 0.008129 | 0.09375 | -3.0627 | 0.9209 |
| mean_dropout0_varcov | 0 | 0.021106 | 0.011776 | 0.0625 | -0.6616 | 0.9202 |

## Interpretation

Disabling dropout does not fix the failure. Explicit variance/covariance regularization can remove the hard collapse flag in the mean-pooling variant, but retrieval remains random and RNA biological latent R2 remains negative.

This means collapse prevention is necessary but not sufficient. The RNA global condition embedding is not learning useful biological structure, while the image side consistently recovers the synthetic biological latent.

## Decision

Do not promote. Do not run a broad sweep. The next repair must target RNA condition readout / biological signal extraction, not only variance geometry.
