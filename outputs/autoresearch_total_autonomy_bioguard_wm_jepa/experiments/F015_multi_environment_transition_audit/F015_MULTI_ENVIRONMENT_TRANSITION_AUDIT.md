# F015 Multi-Environment Transition Audit

## Decision
`F015_POOLED_ENVIRONMENT_TRANSITION_PASSES_GATES_BELOW_PER_SEED_FLOOR`

## Baseline Summary
| baseline | seeds | mean transition improvement | mean delta cosine | mean condition recall@1 | mean delta rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| pooled_source_only_ridge | 5 | 0.014092 | 0.450511 | 0.222222 | 9.905101 |
| pooled_full_ridge | 5 | 0.014049 | 0.449757 | 0.222222 | 10.271098 |
| pooled_environment_centered_ridge | 5 | 0.011833 | 0.431519 | 0.325926 | 10.476280 |
| per_seed_action_ridge_floor | 5 | 0.011281 | 0.451515 | 0.333333 | 10.245809 |
| pooled_action_only_ridge | 5 | 0.000830 | 0.080440 | 0.318519 | 1.860602 |
| leave_one_seed_out_full_ridge | 5 | -0.017283 | 0.334519 | 0.185185 | 8.490332 |

## Interpretation
This diagnostic tests whether multi-seed/environment pooling or train-only environment centering gives transition headroom beyond the per-seed action-ridge floor. It is not a JEPA candidate and cannot promote the model of record.
