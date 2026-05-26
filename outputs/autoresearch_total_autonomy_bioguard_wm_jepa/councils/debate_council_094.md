# Debate Council 094

## Trigger
`F064_TRAIN_SELECTOR_OVERFITS_CONTINUOUS_TRANSITION_GAINS`

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
| F065 train-only action-heldout calibration | 0.800 | 0.580 | Replace in-sample train residual selection with leave-perturbation-out train calibration before scoring held-out perturbations. |
| Try a smaller residual cap immediately | 0.700 | 0.220 | Reject because F064 shows the core issue is train-to-heldout optimism, not just cap size. |
| Jump to a new representation family | 0.711 | 0.420 | Defer one calibration gate so the representation pivot is justified by a split-matched train-only failure. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F065 train-only action-heldout calibration`.

## Next Experiment Command Or Target
`Replace in-sample train residual selection with leave-perturbation-out train calibration before scoring held-out perturbations.`
