# PLS Distilled Head Tier 2 Summary

## Candidates

Two protected student-head variants were evaluated after the frozen PLS JEPA baseline:

- `raw_mlp`: separate nonlinear raw RNA/image heads trained from random initialization against the frozen PLS readout for 150 CPU steps.
- `linear_clone`: separate trainable linear RNA/image heads initialized exactly from the train-split-only rank-3 PLS readout and trained for 10 CPU steps with no auxiliary drift pressure.

The protected retrieval path remained the frozen raw-linear PLS readout in both candidates.

## Raw MLP Student Result

| seed | student pass | protected preserved | RNA->image R@1 | RNA->image R@5 | RNA latent R2 | image latent R2 | RNA min std | image min std | batch bal acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0.1250 | 0.3438 | 0.0441 | 0.1751 | 0.7075 | 0.1447 | 0.5938 |
| 1 | 0 | 1 | 0.0938 | 0.5000 | 0.2842 | 0.8099 | 0.2628 | 0.0886 | 0.4375 |
| 2 | 0 | 1 | 0.0625 | 0.5312 | 0.4903 | 0.4810 | 0.6046 | 0.4415 | 0.3750 |

Tier 2 means:

- RNA->image recall@1: `0.0938 +/- 0.0255`
- RNA->image recall@5: `0.4583 +/- 0.0820`
- RNA latent R2: `0.2729 +/- 0.1824`
- Image latent R2: `0.4887 +/- 0.2592`
- Batch balanced accuracy: `0.4688 +/- 0.0920`

Decision: `TIER2_FAIL_SIGNAL_INCONSISTENT`. The raw MLP student is noncollapsed and often biologically informative, but retrieval does not clear the random-plus-margin gate across seeds.

## Linear Clone Student Result

| seed | student pass | protected preserved | readout drift | RNA->image R@1 | RNA->image R@5 | RNA latent R2 | image latent R2 | batch bal acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0.00000000 | 0.2188 | 0.5000 | 0.4155 | 0.8876 | 0.5000 |
| 1 | 1 | 1 | 0.00000000 | 0.2188 | 0.6875 | 0.7563 | 0.9379 | 0.5625 |
| 2 | 1 | 1 | 0.00000000 | 0.2812 | 0.8125 | 0.6069 | 0.9146 | 0.3750 |

Tier 2 means:

- RNA->image recall@1: `0.2396 +/- 0.0295`
- RNA->image recall@5: `0.6667 +/- 0.1284`
- RNA latent R2: `0.5929 +/- 0.1395`
- Image latent R2: `0.9134 +/- 0.0206`
- Batch balanced accuracy: `0.4792 +/- 0.0780`
- Student-to-teacher MSE: `0.0` for RNA and image in all seeds

Decision: `TIER2_PASS_CLEAN_ENGINEERING_BASELINE`. This is not a new biological improvement over PLS. It establishes a separate trainable student head that can be serialized, evaluated independently, and protected from the frozen retrieval path.

## Interpretation

The raw nonlinear student confirms that direct neural distillation is underconstrained at this low data scale: it preserves variance and often learns biological signal, but seed-to-seed retrieval alignment is unstable. The linear clone confirms the safe handoff boundary: closed-form PLS geometry should enter trainable code as an exact initialized linear student, not as a randomly initialized MLP.

## Zero-Initialized Residual Adapter

| seed | student pass | protected preserved | RNA residual scale | image residual scale | RNA->image R@1 | RNA->image R@5 | RNA latent R2 | image latent R2 | batch bal acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0.0118 | 0.0326 | 0.2188 | 0.5000 | 0.4155 | 0.8875 | 0.5000 |
| 1 | 1 | 1 | -0.0050 | 0.0273 | 0.2188 | 0.6875 | 0.7563 | 0.9379 | 0.5625 |
| 2 | 1 | 1 | 0.0063 | -0.0054 | 0.2812 | 0.8125 | 0.6069 | 0.9146 | 0.3750 |

Tier 2 means:

- RNA->image recall@1: `0.2396 +/- 0.0295`
- RNA->image recall@5: `0.6667 +/- 0.1284`
- RNA latent R2: `0.5929 +/- 0.1395`
- Image latent R2: `0.9134 +/- 0.0206`
- Batch balanced accuracy: `0.4792 +/- 0.0780`
- Mean residual scale: RNA `0.0044 +/- 0.0070`, image `0.0182 +/- 0.0168`
- Mean student-teacher MSE: RNA `0.0000289`, image `0.0002950`

Decision: `TIER2_PASS_CLEAN_NO_IMPROVEMENT_DO_NOT_PROMOTE`. The residual adapter is safe under this tiny trust-region setting but behaves as a near-clone and does not improve the primary metric.

## Next Step

Use the linear clone as the student model of record for trainable shared geometry. The residual adapter should stay available as a controlled extension point, but it should not replace the clone unless a future residual objective improves recall or counterfactual behavior without moving the PLS trust region.
