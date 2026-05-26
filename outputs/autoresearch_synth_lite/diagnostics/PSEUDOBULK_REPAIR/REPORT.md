# RNA Pseudobulk Condition Readout Repair Report

## Setup

Two focused CPU-only checks were run on `synth_micro`, seed `0`, 30 steps:

- `pseudobulk_varcov`: replace RNA shared condition embedding with a pseudobulk gene/value embedding readout.
- `encoder_plus_pseudobulk_varcov`: residual combination of encoder shared embedding and pseudobulk readout.

Both used mean bag aggregation, dropout `0.0`, shared variance weight `0.2`, and shared covariance weight `0.01`.

## Results

| Run | Collapse | RNA min std | Image min std | Recall@1 | RNA latent R2 | Image latent R2 | RNA rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_varcov_baseline | 0 | 0.021106 | 0.011776 | 0.0625 | -0.6616 | 0.9202 | 25 |
| token_pool_mean_varcov | 1 | 0.031627 | 0.009962 | 0.0625 | -0.3632 | 0.9210 | 29 |
| pseudobulk_varcov | 1 | 0.000242 | 0.005780 | 0.0625 | 0.1088 | 0.9112 | 5 |
| encoder_plus_pseudobulk_varcov | 1 | 0.000417 | 0.012845 | 0.0625 | 0.1522 | 0.9146 | 14 |
| pseudobulk_var2_cov001 | 1 | 0.000314 | 0.013171 | 0.0625 | 0.0763 | 0.9234 | 5 |
| pseudobulk_unnorm_varcov | 1 | 0.001053 | 0.006286 | 0.0625 | -0.9428 | 0.9111 | 5 |

## Interpretation

The pseudobulk readout is biologically better for RNA latent recovery, moving RNA latent R2 from negative to positive. But it collapses the RNA representation severely and does not improve retrieval. The residual version preserves image variance but still collapses RNA variance.

So the next useful mechanism is not simply replacing the RNA condition readout. Stronger variance pressure and disabling final pseudobulk L2 normalization both failed. We need to preserve the pseudobulk biological signal while preventing the projection/readout from compressing it into a low-rank narrow cone.

## Decision

Do not promote. Keep the implementation as a focused diagnostic/repair option. The next step should be a different RNA condition architecture, likely direct pseudobulk-to-shared whitening or a supervised synthetic latent probe, before spending more runs on JEPA alignment.
