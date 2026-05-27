# Phase Closure Report 123

## Trigger
`FAIL_EXTERNAL_NO_PROMOTION` from `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`

## Interpretation
F094 ran on CUDA and completed as a non-promoting scientific failure. The split-safe gate correctly refused harmful calibration and selected raw JEPA output for all three seeds, preserving held-out delta cosine relative to the protected floor. That did not clear the external transition floor on `alternate_test` or `test`, so calibration repair alone is insufficient.

## F094 Summary
| split | transition improvement | delta cosine | recall@1 | floor gap transition | floor gap delta cosine | floor gap recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternate_test | 0.440256 | 0.338808 | 0.115152 | -0.090531 | 0.026219 | -0.012121 |
| test | 0.573212 | 0.458449 | 0.321429 | -0.029148 | 0.069711 | 0.071429 |
| validation | 0.324853 | 0.350708 | 0.283951 | 0.012280 | 0.056553 | 0.135802 |

## Repair Direction
Proceed to non-exact action descriptor repair. The current 12-dimensional scalar PubChem descriptor is leakage-safe but too low-capacity for perturbation mechanism transfer. Do not train a new architecture before the descriptor branch is tested.
