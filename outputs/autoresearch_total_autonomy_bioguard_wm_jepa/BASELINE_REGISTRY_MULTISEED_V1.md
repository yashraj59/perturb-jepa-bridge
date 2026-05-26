# Synthetic Genetic Anchor Lite Multiseed v1 Baseline Registry

## Decision
`F013_MULTISEED_STEP0_REGISTRY_COMPLETE_ARCHITECTURE_STILL_LOCKED`

## Scope
This is a new named Step 0 registry for `synthetic_genetic_anchor_lite_multiseed_v1`. It does not alter the locked seed0 benchmark, the protected model of record, or any promotion gate.

## Leakage Contract
- Fit data: each seed's train split only.
- Eval data: scoring only.
- Forbidden model inputs: `condition_key`, `biological_key`, exact target-key one-hot features, eval/test target means, pooled train+test statistics, and PLS raw-linear heads as candidate path.
- Metadata labels are used only for retrieval reporting.

## Baseline Summary
| baseline | seeds | mean transition improvement | mean delta cosine | mean condition recall@1 | mean perturbation recall@1 | mean cell-line recall@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| action_ridge_floor | 5 | 0.011281 | 0.451515 | 0.333333 | 0.377778 | 0.896296 |
| source_only_ridge | 5 | 0.009866 | 0.427645 | 0.259259 | 0.281481 | 0.911111 |
| source_as_target | 5 | 0.000000 | 0.000000 | 0.333333 | 0.370370 | 0.918519 |
| action_mean_delta | 5 | -0.004957 | 0.045609 | 0.296296 | 0.318519 | 0.903704 |
| mean_delta | 5 | -0.004957 | 0.045609 | 0.296296 | 0.318519 | 0.903704 |
| action_only_ridge | 5 | -0.005422 | 0.108696 | 0.362963 | 0.392593 | 0.911111 |
| seed_randomized_full_ridge_null | 5 | -0.014996 | 0.053779 | 0.303704 | 0.355556 | 0.874074 |

## Interpretation
The old seed0 condition-recall floor remains locked for historical comparison, but future architecture should not be optimized against that single-seed threshold until a separate amendment defines new multi-seed Tier 1/Tier 2/Tier 3 gates from this registry.
