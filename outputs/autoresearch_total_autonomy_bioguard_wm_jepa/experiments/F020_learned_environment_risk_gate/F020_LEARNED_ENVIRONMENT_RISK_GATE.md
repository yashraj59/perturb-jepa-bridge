# F020 Learned Environment Risk Gate

## Decision
`F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK`

## Selected Train-Only Learned Rule
- candidate: `floor_fallback`
- alpha: `0.000000`
- threshold: `0.000000`
- selection label: `F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK`
- features: `action_support, blend_delta_support, blend_minus_floor_norm, blend_to_floor_norm_ratio, floor_blend_delta_cosine, floor_delta_support, source_support, support_gain`

## Top Out-Of-Fold Calibration Rules
| candidate | alpha | threshold | active fraction | mean transition gap | min transition gap | mean delta gap | mean recall gap | min recall gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled_environment_centered_ridge | 0.350 | 0.546010 | 0.750000 | 0.001895 | -0.002338 | 0.013584 | -0.005556 | -0.111111 |
| pooled_environment_centered_ridge | 0.750 | 0.501531 | 0.500000 | 0.001860 | -0.004278 | 0.012725 | -0.008333 | -0.111111 |
| pooled_environment_centered_ridge | 0.500 | 0.558608 | 0.500000 | 0.001812 | -0.001840 | 0.014186 | 0.002778 | -0.111111 |
| pooled_environment_centered_ridge | 0.750 | 0.534561 | 0.350000 | 0.001708 | -0.002257 | 0.014307 | 0.000000 | -0.111111 |
| pooled_environment_centered_ridge | 0.500 | 0.506016 | 0.750000 | 0.001705 | -0.004493 | 0.012416 | -0.005556 | -0.222222 |
| pooled_full_ridge | 0.500 | 0.554125 | 0.750000 | 0.001682 | -0.004383 | 0.010982 | -0.011111 | -0.222222 |
| pooled_environment_centered_ridge | 0.350 | 0.561380 | 0.650000 | 0.001627 | -0.002515 | 0.012031 | -0.008333 | -0.111111 |
| pooled_environment_centered_ridge | 0.500 | 0.533435 | 0.650000 | 0.001603 | -0.004493 | 0.013366 | -0.005556 | -0.222222 |
| pooled_environment_centered_ridge | 0.500 | 0.593689 | 0.350000 | 0.001540 | -0.001663 | 0.011539 | -0.002778 | -0.111111 |
| pooled_environment_centered_ridge | 0.750 | 0.478560 | 0.650000 | 0.001513 | -0.004697 | 0.011179 | -0.008333 | -0.111111 |

## Held-Out Seed Scoring
| seed | active fraction | transition improvement | delta cosine | condition recall@1 | transition gap vs floor | recall gap vs floor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000000 | 0.005662 | 0.397963 | 0.481481 | 0.000000 | 0.000000 |
| 1 | 0.000000 | 0.021756 | 0.540428 | 0.333333 | 0.000000 | 0.000000 |
| 2 | 0.000000 | 0.009402 | 0.401885 | 0.296296 | 0.000000 | 0.000000 |
| 3 | 0.000000 | 0.000424 | 0.420055 | 0.259259 | 0.000000 | 0.000000 |
| 4 | 0.000000 | 0.019162 | 0.497245 | 0.296296 | 0.000000 | 0.000000 |

## Interpretation
F020 is the final low-compute test of the environment-blend branch before cooling it: a learned linear risk score is fit only from train perturbation folds and then frozen for held-out scoring. It cannot promote the model of record.
