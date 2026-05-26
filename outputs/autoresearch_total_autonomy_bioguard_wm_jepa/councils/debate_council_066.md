# Debate Council 066

## Trigger
`F036_RECALL_FAILURE_PRIMARILY_MARGIN_INSTABILITY`

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
| F037 train-only retrieval-margin gate | 0.823 | 0.660 | Select residual scale on inner validation only if it preserves continuous metrics and creates zero broken floor-correct retrieval rows. |
| Endpoint+delta composite metric replacement | 0.734 | 0.620 | Defer because metric replacement needs train-only calibration evidence and cannot rescue F035 post hoc. |
| Ignore near-tie recall failures | 0.597 | 0.120 | Reject because no-regression gates remain active even when the failure looks like near-tie instability. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F037 train-only retrieval-margin gate`.

## Next Experiment Command Or Target
`Select residual scale on inner validation only if it preserves continuous metrics and creates zero broken floor-correct retrieval rows.`
