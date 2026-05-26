# Debate Council 041

## Trigger
`F014_MULTISEED_V1_BENCHMARK_ACTIVATED_FOR_SEARCH`

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
| BGWM003 multiseed Action-AdaLN sanity | 0.854 | 0.660 | Run BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY on seeds 0-4 under active multiseed gates. |
| New architecture family immediately | 0.726 | 0.500 | Defer until the existing floor-preserving Action-AdaLN path is sanity-checked under the new benchmark. |
| Return to old seed0 benchmark | 0.607 | 0.200 | Reject because F014 activated the named multiseed benchmark without mutating the old benchmark. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `BGWM003 multiseed Action-AdaLN sanity`.

## Next Experiment Command Or Target
`Run BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY on seeds 0-4 under active multiseed gates.`
