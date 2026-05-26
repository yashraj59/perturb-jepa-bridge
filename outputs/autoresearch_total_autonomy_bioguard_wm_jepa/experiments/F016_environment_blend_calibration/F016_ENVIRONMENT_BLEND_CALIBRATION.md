# F016 Environment Blend Calibration

## Decision
`F016_TRAIN_ONLY_BLEND_SELECTS_FLOOR_FALLBACK`

## Selected Train-Only Rule
- candidate: `floor_fallback`
- alpha: `0.000000`
- selection label: `F016_TRAIN_ONLY_CALIBRATION_SELECTS_FLOOR_FALLBACK`

## Top Calibration Rules By Mean Transition Gap
| candidate | alpha | mean transition gap | min transition gap | mean recall gap | min recall gap | mean delta gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled_full_ridge | 0.500 | 0.002410 | -0.005678 | -0.019444 | -0.333333 | 0.013637 |
| pooled_source_only_ridge | 0.500 | 0.002377 | -0.005326 | -0.019444 | -0.333333 | 0.013424 |
| pooled_full_ridge | 0.350 | 0.002290 | -0.003218 | -0.000000 | -0.222222 | 0.014805 |
| pooled_source_only_ridge | 0.350 | 0.002270 | -0.003057 | -0.002778 | -0.222222 | 0.014714 |
| pooled_environment_centered_ridge | 0.500 | 0.001967 | -0.003123 | -0.016667 | -0.333333 | 0.014097 |
| pooled_environment_centered_ridge | 0.350 | 0.001824 | -0.001686 | -0.011111 | -0.111111 | 0.013541 |
| pooled_full_ridge | 0.200 | 0.001648 | -0.001475 | -0.008333 | -0.111111 | 0.011016 |
| pooled_source_only_ridge | 0.200 | 0.001638 | -0.001470 | -0.008333 | -0.111111 | 0.010996 |

## Held-Out Seed Scoring
| seed | transition improvement | delta cosine | condition recall@1 | transition gap vs floor | recall gap vs floor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.005662 | 0.397963 | 0.481481 | 0.000000 | 0.000000 |
| 1 | 0.021756 | 0.540428 | 0.333333 | 0.000000 | 0.000000 |
| 2 | 0.009402 | 0.401885 | 0.296296 | 0.000000 | 0.000000 |
| 3 | 0.000424 | 0.420055 | 0.259259 | 0.000000 | 0.000000 |
| 4 | 0.019162 | 0.497245 | 0.296296 | 0.000000 | 0.000000 |

## Interpretation
F015 found pooled transition headroom but recall loss. F016 tests whether that headroom can be safely used by a convex blend selected only with train-split, perturbation-grouped cross-fitting. It is diagnostic only and cannot promote the protected model of record.
