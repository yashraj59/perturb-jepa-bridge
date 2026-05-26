# Phase 4 Delta Operator Audit

## Scope

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Model used for latent cache: BTJ001 BioTech-JEPA checkpoint; protected PLS remains audit-only.
- No BioFlow model code or model training was used to make this decision.
- Forbidden shortcuts checked: no condition-key input, no biological-key one-hot input, no test target means, no pooled train+test targets.

## Teacher Delta Statistics

### train

| metric | value |
| --- | ---: |
| `n_samples` | 72.0000 |
| `delta_mean_norm` | 0.4310 |
| `delta_std_mean` | 0.0846 |
| `delta_effective_rank` | 13.5627 |
| `delta_perturbation_nn_recall@1` | 0.3056 |
| `delta_batch_probe_accuracy` | 0.3611 |
| `delta_batch_probe_majority_accuracy` | 0.3333 |
| `delta_perturbation_probe_accuracy` | 0.4583 |
| `source_to_target_cosine_mean` | 0.8977 |
| `delta_near_zero_fraction_norm_lt_1e-3` | 0.0000 |

### eval

| metric | value |
| --- | ---: |
| `n_samples` | 27.0000 |
| `delta_mean_norm` | 0.4252 |
| `delta_std_mean` | 0.0832 |
| `delta_effective_rank` | 11.7819 |
| `delta_perturbation_nn_recall@1` | 0.5926 |
| `delta_batch_probe_accuracy` | 0.2963 |
| `delta_batch_probe_majority_accuracy` | 0.3333 |
| `delta_perturbation_probe_accuracy` | 0.4815 |
| `source_to_target_cosine_mean` | 0.9031 |
| `delta_near_zero_fraction_norm_lt_1e-3` | 0.0000 |

## Simple Transition Baselines

| split | baseline                       | transition_source_cosine_improvement | transition_to_target_recall@1 | transition_to_target_median_rank | delta_cosine | delta_magnitude_ratio | delta_prediction_effective_rank |
| ----- | ------------------------------ | ------------------------------------ | ----------------------------- | -------------------------------- | ------------ | --------------------- | ------------------------------- |
| eval  | action_ridge_delta             | 0.0057                               | 0.4815                        | 2.0000                           | 0.3980       | 0.7744                | 10.2835                         |
| eval  | action_low_rank_ridge          | 0.0046                               | 0.4074                        | 2.0000                           | 0.3877       | 0.7585                | 6.7681                          |
| eval  | source_as_target_null          | 0.0000                               | 0.2963                        | 4.0000                           | 0.0000       | 0.0000                | 0.0000                          |
| eval  | mean_delta_null                | -0.0188                              | 0.3333                        | 5.0000                           | -0.0761      | 0.3932                | 0.0000                          |
| eval  | action_mean_delta              | -0.0188                              | 0.3333                        | 5.0000                           | -0.0761      | 0.3932                | 0.0000                          |
| eval  | action_knn_delta               | -0.0692                              | 0.4074                        | 2.0000                           | 0.1518       | 1.1210                | 10.5857                         |
| train | oracle_train_delta_upper_bound | 0.1023                               | 1.0000                        | 1.0000                           | 1.0000       | 1.0000                | 13.5627                         |
| train | action_ridge_delta             | 0.0769                               | 0.7778                        | 1.0000                           | 0.8483       | 0.8535                | 11.6312                         |
| train | action_low_rank_ridge          | 0.0740                               | 0.6111                        | 1.0000                           | 0.8230       | 0.8318                | 7.0452                          |
| train | action_mean_delta              | 0.0396                               | 0.4722                        | 2.0000                           | 0.5930       | 0.6707                | 5.5697                          |
| train | mean_delta_null                | 0.0108                               | 0.1806                        | 8.0000                           | 0.3430       | 0.3995                | 0.0000                          |
| train | source_as_target_null          | 0.0000                               | 0.0972                        | 8.5000                           | 0.0000       | 0.0000                | 0.0000                          |
| train | action_knn_delta               | -0.0014                              | 0.5694                        | 1.0000                           | 0.3839       | 0.9884                | 12.9641                         |

## Frozen-Latent Gradient/Sign Audit

| audit                    | step_count | transition_source_cosine_improvement | absolute_target_cosine | delta_cosine | delta_magnitude_ratio | delta_prediction_effective_rank | source_improvement_hinge_violation_fraction |
| ------------------------ | ---------- | ------------------------------------ | ---------------------- | ------------ | --------------------- | ------------------------------- | ------------------------------------------- |
| zero_init                | 0.0000     | -0.0000                              | 0.8977                 | 0.0000       | 0.0000                | 0.0000                          | 1.0000                                      |
| one_step_raw_delta_mse   | 1.0000     | 0.0006                               | 0.8983                 | 0.4108       | 0.0090                | 6.6136                          | 1.0000                                      |
| raw_delta_mse            | 20.0000    | 0.0403                               | 0.9380                 | 0.6699       | 1.1511                | 9.7235                          | 0.4722                                      |
| endpoint_cosine          | 20.0000    | 0.0607                               | 0.9584                 | 0.1229       | 16.3994               | 8.3893                          | 0.2917                                      |
| whitened_delta_mse       | 20.0000    | 0.0386                               | 0.9363                 | 0.6032       | 0.8577                | 9.6204                          | 0.4722                                      |
| whitened_hinge_direction | 20.0000    | -0.7517                              | 0.1460                 | 0.4179       | 17.7108               | 8.6785                          | 1.0000                                      |

## Reopening Decision

Decision label: `PHASE4_DELTA_OPERATOR_REOPEN_APPROVED`

- teacher_delta_measurable=True (eval_rank=11.7819, eval_std_mean=0.0832, eval_mean_norm=0.4252)
- positive_simple_baseline=True (action_low_rank_ridge/eval, action_low_rank_ridge/train, action_mean_delta/train, action_ridge_delta/eval, action_ridge_delta/train, mean_delta_null/train)
- train_optimization_improves_first20=True
- targeted_fix_available=True (delta whitening + source-improvement hinge + direction loss + vector field)
- leakage_check=pass (train-only deltas for fit; no condition_key/test target means/pooled targets used)

## Conclusion

- Best eval simple baseline: `action_ridge_delta` with transition improvement `0.0057`.
- Best frozen-latent optimization audit: `endpoint_cosine` with train transition improvement `0.0607`.
- Targeted Phase 4 mechanism, if reopened: delta whitening, source-improvement hinge, delta direction loss, endpoint latent JEPA, and a controlled vector-field transition in `z_bio` space.
