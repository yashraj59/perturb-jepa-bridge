# BGWM003 Multiseed v1 Action-AdaLN Sanity

## Decision
`BGWM003_PASS_FLOOR_PRESERVING_SANITY_ZERO_RESIDUAL`

## Active Gates
- transition improvement >= `0.003221`
- delta cosine >= `0.394388`
- secondary condition recall@1 >= `0.255644`

## Seed Results
| seed | transition improvement | delta cosine | recall@1 | residual scale | floor-gap transition | train decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.005662 | 0.397963 | 0.481481 | 0.000 | 0.000000 | BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN |
| 1 | 0.021756 | 0.540428 | 0.333333 | 0.000 | 0.000000 | BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN |
| 2 | 0.009402 | 0.401885 | 0.296296 | 0.000 | 0.000000 | BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN |
| 3 | 0.000424 | 0.420055 | 0.259259 | 0.000 | 0.000000 | BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN |
| 4 | 0.019162 | 0.497245 | 0.296296 | 0.000 | 0.000000 | BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN |

## Interpretation
This is a low-compute Tier 1 sanity run for the activated multi-seed benchmark. It verifies whether the existing floor-preserving Action-AdaLN + RoPE residual path can run across seeds without regressing the new gates. It cannot promote the model of record and is not a full cross-modal JEPA validation.
