# Phase Closure Report 048: Environment-Blend Risk Branch

## Trigger
`F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK`

## Interpretation
The environment-blend/risk-gate branch is cooled. F017 showed oracle capacity, but F016 selected floor fallback, F018 violated seed0 recall, F019 found no simple second-gate repair, and F020 selected floor fallback under out-of-fold learned risk calibration. This is an ordinary scientific branch closure, not a project halt.

## Protected Model Status
The protected rank-3 train-split-only PLS raw-linear readout remains the model of record. No JEPA candidate is promoted.

## Rows
| experiment | decision | transition improvement | transition gap | recall gap | residual/blend scale |
| --- | --- | ---: | ---: | ---: | ---: |
| F015 | F015_POOLED_ENVIRONMENT_TRANSITION_PASSES_GATES_BELOW_PER_SEED_FLOOR | 0.014092 | 0.002811 | -0.111111 | 0.0 |
| F016 | F016_TRAIN_ONLY_BLEND_SELECTS_FLOOR_FALLBACK | 0.011281 | 0.0 | 0.0 | 0.0 |
| F017 | F017_ORACLE_SAFE_ENVIRONMENT_BLEND_CAPACITY_EXISTS | 0.021782 | 0.010501 | 0.051852 | 1.0 |
| F018 | F018_TRAIN_ONLY_RISK_PROXY_DISCARDED_HELDOUT_BELOW_FLOOR | 0.014645 | 0.003364 | 0.0 | 0.35 |
| F019 | F019_NO_SIMPLE_SECOND_GATE_REPAIRS_F018_FAILURE | 0.003364 | 0.003364 | 0.0 | 0.35 |
| F020 | F020_OOF_LEARNED_RISK_SELECTS_FLOOR_FALLBACK | 0.011281 | 0.0 | 0.0 | 0.0 |
