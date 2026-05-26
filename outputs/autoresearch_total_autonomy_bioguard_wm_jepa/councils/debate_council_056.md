# Debate Council 056

## Trigger
`F026_DESCRIPTOR_ALIGNED_BENCHMARK_APPROVES_STEP0_REDESIGN`

## Evidence Summary
The current trigger is an ordinary scientific pivot event, not a hard escalation. The protected model of record and protected full-ridge transition floor remain active while the council selects the smallest falsifiable next step.

## Independent Proposals
- Architect: try population-level or safer gated residual mechanisms rather than forcing residual magnitude.
- Skeptic: repeated residual failures suggest benchmark noise or representation/floor saturation, not only predictor weakness.
- Methodologist: audit evaluation variance and held-out perturbation difficulty before spending more training compute.
- Biologist: inspect whether perturbation effects are too small or mean-like in current synthetic anchors.
- Monitor: no hard escalation trigger is present; continue through documented amendment.

## Steelman Of Opposing Proposals
- A stronger predictor could still help if the residual target is learnable but calibration is too conservative.
- Population transport could address mean-like deterministic endpoints, but it is premature before noise/difficulty auditing.

## Three-Round Debate Summary
Round 1 favored predictor repair; Round 2 noted repeated residual overfit; Round 3 selected metric/data redesign as the smallest falsifiable next step.

## Scoring Table
| proposal | average | minimum | next action |
| --- | ---: | ---: | --- |
| F027 Cell-JEPA warmstart on descriptor-aligned benchmark | 0.869 | 0.700 | Train a tiny Cell-JEPA RNA warmstart on the repaired benchmark and compare its program-action floor with true z_bio and observed RNA PCA. |
| Train full cross-modal JEPA immediately | 0.754 | 0.520 | Defer one step because representation extraction should be checked on the repaired benchmark before spending training compute. |
| Return to old benchmark residuals | 0.669 | 0.200 | Reject because F026 proved the old action-target contract is not suitable for non-exact descriptor learning. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F027 Cell-JEPA warmstart on descriptor-aligned benchmark`.

## Next Experiment Command Or Target
`Train a tiny Cell-JEPA RNA warmstart on the repaired benchmark and compare its program-action floor with true z_bio and observed RNA PCA.`
