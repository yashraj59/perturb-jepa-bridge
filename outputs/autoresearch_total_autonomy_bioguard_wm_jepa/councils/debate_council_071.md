# Debate Council 071

## Trigger
`F041_SIMPLE_TRAIN_PROXY_ABLATION_RECOVERS_SAFE_NONZERO_DIAGNOSTIC`

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
| F042 fresh-seed tiny-cap validation | 0.814 | 0.540 | Pre-register the F041 tiny-cap rule and validate it on fresh synthetic seeds 5-9 with held-out scoring only after train-only scale selection. |
| Immediately use small-cap rule as candidate | 0.697 | 0.340 | Reject because F041 discovered the rule with held-out diagnostic outcomes on seeds 0-4. |
| Train another retrieval-aware objective | 0.706 | 0.480 | Defer until the low-compute fresh-seed validation tells whether a tiny cap generalizes. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F042 fresh-seed tiny-cap validation`.

## Next Experiment Command Or Target
`Pre-register the F041 tiny-cap rule and validate it on fresh synthetic seeds 5-9 with held-out scoring only after train-only scale selection.`
