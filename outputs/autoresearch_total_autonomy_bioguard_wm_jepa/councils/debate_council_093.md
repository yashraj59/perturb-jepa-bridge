# Debate Council 093

## Trigger
`F063_TRAIN_CONE_SELECTOR_HELDOUT_BELOW_FLOOR`

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
| F064 train-heldout selector mismatch audit | 0.880 | 0.500 | Read F063 artifacts and quantify whether train-only residual gains invert on held-out scoring before changing the model again. |
| Lower residual cap and rerun | 0.674 | 0.280 | Reject for now because F063 already used row-wise abstention and still selected below-floor held-out residuals. |
| Jump directly to representation warmstart | 0.700 | 0.440 | Defer one read-only audit so the pivot is based on the exact train-held-out failure signature. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `F064 train-heldout selector mismatch audit`.

## Next Experiment Command Or Target
`Read F063 artifacts and quantify whether train-only residual gains invert on held-out scoring before changing the model again.`
