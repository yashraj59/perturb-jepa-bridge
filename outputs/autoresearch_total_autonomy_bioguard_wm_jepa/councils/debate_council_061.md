# Debate Council 061

## Trigger
`F031_PROTECTED_TIER2_PASS_LOCAL_RIDGE_FLOOR_NOT_PRESERVED`

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
| F032 train-only floor-preserving JEPA residual calibration | 0.840 | 0.620 | Default to the train-only same-representation ridge floor and select a JEPA residual scale only if train-only calibration preserves transition, delta cosine, and recall. |
| Proceed to Tier 3 despite local floor miss | 0.627 | 0.440 | Reject because F031 did not preserve the stronger local train-only floor required for no-regression confidence. |
| Train longer with the same direct head | 0.623 | 0.180 | Reject because the failure is local floor preservation, not lack of positive protected-floor signal. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F032 train-only floor-preserving JEPA residual calibration`.

## Next Experiment Command Or Target
`Default to the train-only same-representation ridge floor and select a JEPA residual scale only if train-only calibration preserves transition, delta cosine, and recall.`
