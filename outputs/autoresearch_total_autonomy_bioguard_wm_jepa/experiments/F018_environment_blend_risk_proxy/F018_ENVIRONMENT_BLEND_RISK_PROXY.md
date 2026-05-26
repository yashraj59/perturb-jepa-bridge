# F018 Environment Blend Risk Proxy

## Decision
`F018_TRAIN_ONLY_RISK_PROXY_DISCARDED_HELDOUT_BELOW_FLOOR`

## Selected Train-Only Risk Rule
- candidate: `pooled_full_ridge`
- alpha: `0.350000`
- feature: `support_gain`
- direction: `ge`
- threshold: `-0.024371`
- selection label: `F018_TRAIN_ONLY_RISK_PROXY_SELECTS_NONZERO_RULE`

## Top Train-Only Calibration Rules
| candidate | alpha | feature | direction | threshold | active fraction | transition gap | delta gap | recall gap |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pooled_full_ridge | 0.500 | action_support | le | 0.000000 | 1.000000 | 0.002410 | 0.013637 | -0.019444 |
| pooled_full_ridge | 0.500 | action_support | ge | 0.000000 | 1.000000 | 0.002410 | 0.013637 | -0.019444 |
| pooled_source_only_ridge | 0.500 | action_support | ge | 0.000000 | 1.000000 | 0.002377 | 0.013424 | -0.019444 |
| pooled_source_only_ridge | 0.500 | action_support | le | 0.000000 | 1.000000 | 0.002377 | 0.013424 | -0.019444 |
| pooled_full_ridge | 0.500 | blend_to_floor_norm_ratio | le | 0.984217 | 0.800000 | 0.002313 | 0.012293 | -0.013889 |
| pooled_full_ridge | 0.500 | source_support | le | 0.997165 | 0.802778 | 0.002300 | 0.012393 | -0.011111 |
| pooled_full_ridge | 0.350 | action_support | ge | 0.000000 | 1.000000 | 0.002290 | 0.014805 | -0.000000 |
| pooled_full_ridge | 0.350 | action_support | le | 0.000000 | 1.000000 | 0.002290 | 0.014805 | -0.000000 |
| pooled_source_only_ridge | 0.500 | blend_to_floor_norm_ratio | le | 0.984465 | 0.800000 | 0.002275 | 0.012015 | -0.013889 |
| pooled_source_only_ridge | 0.500 | source_support | le | 0.997165 | 0.802778 | 0.002271 | 0.012199 | -0.011111 |

## Held-Out Seed Scoring
| seed | active fraction | transition improvement | delta cosine | condition recall@1 | transition gap vs floor | recall gap vs floor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.851852 | 0.010769 | 0.405189 | 0.370370 | 0.005107 | -0.111111 |
| 1 | 0.592593 | 0.023446 | 0.545434 | 0.333333 | 0.001690 | 0.000000 |
| 2 | 0.814815 | 0.013132 | 0.456717 | 0.333333 | 0.003730 | 0.037037 |
| 3 | 0.666667 | 0.003888 | 0.437369 | 0.296296 | 0.003464 | 0.037037 |
| 4 | 0.703704 | 0.021992 | 0.512489 | 0.333333 | 0.002829 | 0.037037 |

## Interpretation
F017 proved eval-oracle capacity exists. F018 tests whether a simple train-only proxy can recover safe nonzero activation without held-out labels. It is still diagnostic and cannot promote the model of record.
