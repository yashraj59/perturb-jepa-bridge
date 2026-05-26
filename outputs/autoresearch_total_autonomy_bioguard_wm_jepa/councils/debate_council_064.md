# Debate Council 064

## Trigger
`F034_OPERATOR_INITIALIZED_RESIDUAL_DISCARDED_HELDOUT_BELOW_FLOOR`

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
| F035 inner-validation operator residual gate | 0.817 | 0.640 | Select residual scale on a train-only inner validation split, then deploy a full-train operator-initialized head with that fixed scale. |
| Relax heldout recall no-regression | 0.610 | 0.180 | Reject because the protocol requires no-regression, not post-hoc metric relaxation. |
| Abandon operator-initialized residuals immediately | 0.740 | 0.340 | Defer one train-only validation-gate test because F034 improved mean transition, delta cosine, and recall with small localized failures. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F035 inner-validation operator residual gate`.

## Next Experiment Command Or Target
`Select residual scale on a train-only inner validation split, then deploy a full-train operator-initialized head with that fixed scale.`
