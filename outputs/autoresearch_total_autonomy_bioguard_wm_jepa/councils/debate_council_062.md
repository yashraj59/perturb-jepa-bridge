# Debate Council 062

## Trigger
`F032_NONZERO_RESIDUAL_DISCARDED_HELDOUT_BELOW_LOCAL_FLOOR`

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
| F033 conservative train-only residual gate | 0.831 | 0.580 | Use a smaller residual scale grid and require positive train-side transition, delta, and recall slack before deploying a nonzero residual. |
| Tier 3 despite one heldout recall miss | 0.611 | 0.420 | Reject because F032 explicitly violated the heldout local floor on recall. |
| Discard all residuals immediately | 0.744 | 0.300 | Defer one stricter calibration check because F032 had positive mean local floor gaps and only a small recall failure. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F033 conservative train-only residual gate`.

## Next Experiment Command Or Target
`Use a smaller residual scale grid and require positive train-side transition, delta, and recall slack before deploying a nonzero residual.`
