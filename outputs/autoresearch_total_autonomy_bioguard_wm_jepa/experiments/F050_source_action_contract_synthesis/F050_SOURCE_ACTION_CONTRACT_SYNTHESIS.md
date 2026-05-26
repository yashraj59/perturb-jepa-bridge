# F050 Source/Action Contract Synthesis

## Decision
`F050_PROTECTED_FLOOR_IS_COVARIATE_ADJUSTED_SOURCE_DOMINATED`

## Purpose
F047-F049 collectively test whether the active failure is an architecture problem, an exact-action support problem, or a benchmark/action-contract problem.

## Summary Table
| model/audit view | transition improvement | delta cosine | recall@1 |
| --- | ---: | ---: | ---: |
| full protected floor | 0.011281 | 0.451515 | 0.333333 |
| source-only floor | 0.009866 | 0.427645 | 0.259259 |
| program-action-only floor | 0.002384 | 0.414301 | 0.355556 |
| full-fit no-action contribution | 0.012101 | 0.442556 | 0.000000 |
| full-fit no-exact contribution | 0.011281 | 0.451515 | 0.000000 |

## Key Gaps
- program-action transition gap vs full: `-0.008897`
- program-action delta gap vs full: `-0.037215`
- source-only transition gap vs full: `-0.001415`
- adjusted-source transition gap vs full: `0.000819`
- exact train-action direct contribution fraction: `0.000000`
- program direct contribution fraction: `0.337415`

## Interpretation
The full protected floor is not directly using held-out exact perturbation one-hots. Its advantage over the candidate-legal program-action floor is mainly a covariate-adjusted source-state fit learned with train action covariates. This makes the protected floor a valid audit baseline but a poor identity target for a JEPA action contract.

## Next Recommendation
Pivot to a biological action-target contract with shared descriptors or a descriptor-aligned benchmark before another residual/operator head.

## Promotion Status
No model is promoted. The protected model of record and protected full-ridge floor remain active audit references.
