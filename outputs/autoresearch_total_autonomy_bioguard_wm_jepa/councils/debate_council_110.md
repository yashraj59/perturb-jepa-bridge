# Debate Council 110

## Trigger
`F080_FULL_JEPA_WRAPPER_MIXED_OR_INCONCLUSIVE`

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
| F081 train-only JEPA delta calibrator | 0.854 | 0.720 | Fit a train-only linear correction from F080 predicted deltas to train image-PCA deltas, then score the fixed calibrator on heldout perturbations. |
| Increase F080 delta-direction loss and rerun | 0.706 | 0.380 | Defer because the first question is whether the JEPA output contains a linearly recoverable delta direction before retuning losses. |
| Richer action descriptors | 0.757 | 0.680 | Defer until delta-direction recoverability is audited; F080 already improved transition with coarse program actions. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F081 train-only JEPA delta calibrator`.

## Next Experiment Command Or Target
`Fit a train-only linear correction from F080 predicted deltas to train image-PCA deltas, then score the fixed calibrator on heldout perturbations.`
