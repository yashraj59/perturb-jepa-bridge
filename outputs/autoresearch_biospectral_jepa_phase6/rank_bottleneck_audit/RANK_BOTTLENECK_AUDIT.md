# Rank Bottleneck Audit

## Scope

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Latent cache: `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit`
- No BioSpectral architecture was implemented before this audit.
- Leakage check: train-only ridge/basis fits; eval latents used only for scoring; condition labels used only for retrieval labels.

## Stage A Floor Reproduction

- Full action ridge eval improvement: `0.0057`
- Low-rank action ridge eval improvement: `0.0046`

## Reduced-Rank Sweep

| rank | transition_improvement | delta_cosine | recall@1 | median_rank | delta_rank | magnitude_ratio | floor_gap_vs_full_ridge |
| ---- | ---------------------- | ------------ | -------- | ----------- | ---------- | --------------- | ----------------------- |
| 1    | -0.0071                | 0.1823       | 0.3704   | 4.0000      | 1.0000     | 0.5162          | -0.0128                 |
| 2    | -0.0068                | 0.2451       | 0.3333   | 3.0000      | 1.9858     | 0.6267          | -0.0125                 |
| 4    | -0.0008                | 0.3323       | 0.4074   | 3.0000      | 3.7777     | 0.7123          | -0.0065                 |
| 6    | 0.0046                 | 0.3822       | 0.3704   | 2.0000      | 5.3637     | 0.7433          | -0.0011                 |
| 8    | 0.0046                 | 0.3877       | 0.4074   | 2.0000      | 6.7681     | 0.7585          | -0.0011                 |
| 10   | 0.0058                 | 0.3953       | 0.4815   | 2.0000      | 7.9803     | 0.7680          | 0.0001                  |
| 12   | 0.0057                 | 0.3952       | 0.4815   | 2.0000      | 8.6711     | 0.7708          | -0.0000                 |
| 16   | 0.0057                 | 0.3977       | 0.4815   | 2.0000      | 9.8775     | 0.7741          | -0.0000                 |
| 24   | 0.0057                 | 0.3980       | 0.4815   | 2.0000      | 10.2835    | 0.7744          | -0.0000                 |
| full | 0.0057                 | 0.3980       | 0.4815   | 2.0000      | 10.2835    | 0.7744          | -0.0000                 |

## Neural Low-Rank Equivalence

- pass: `1.0`
- eval improvement: `0.0046`
- eval delta cosine: `0.3877`
- eval recall@1: `0.4074`
- eval delta rank: `6.7681`
- max abs eval delta diff: `0.00000010`

## BOJ002 Control-Affine Discrepancy

| item                                    | status  | detail                                                                                                                                             |
| --------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| train_prediction_vs_analytical_low_rank | diff    | rmse=0.024167; max_abs=0.127171                                                                                                                    |
| eval_prediction_vs_analytical_low_rank  | diff    | rmse=0.022674; max_abs=0.085180                                                                                                                    |
| same_action_features                    | pass    | action_dim=20                                                                                                                                      |
| same_train_rows_only                    | pass    | train_rows=72                                                                                                                                      |
| same_ridge_regularization               | pass    | alpha=1e-2                                                                                                                                         |
| same_output_normalization               | warning | operator metrics use source + delta without endpoint normalization; old BOJ002 code trained raw delta then evaluated common metrics                |
| same_rank_orientation                   | fail    | BOJ002 decomposes source block and keeps full action shift; analytical low-rank ridge reduces full predicted delta in teacher-delta spectral basis |
| same_bias_term                          | pass    | delta mean/source/action mean retained                                                                                                             |
| same_feature_standardization            | pass    | mean-centering only                                                                                                                                |
| same_action_descriptor_dimensions       | pass    | 20                                                                                                                                                 |

## Decision

`PHASE6_REOPEN_BIOSPECTRAL_APPROVED`

## Interpretation

BOJ002 failed for both reasons considered by the prompt: its source-block factorization does not reproduce analytical low-rank ridge, and rank-8 reduced-rank ridge itself loses held-out recall/rank relative to the full action-ridge floor. A floor-preserving, rank-adaptive spectral residual has plausible headroom because the full-ridge floor is reproducible and the reduced-rank sweep shows higher ranks recover floor geometry.
