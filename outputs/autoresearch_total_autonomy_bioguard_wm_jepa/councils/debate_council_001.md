# Debate Council 001

## Trigger
`BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN`

## Evidence Summary
BGWM002 selected zero residual across the requested seeds, so the action-AdaLN + RoPE residual did not pass train-only calibration above the protected full-ridge floor.

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
| Family F: bootstrap floor/noise audit | 0.843 | 0.550 | Run F001_BOOTSTRAP_FLOOR_NOISE_AUDIT before more residual architecture. |
| Family B: stricter abstention gate | 0.736 | 0.450 | Implement per-action abstention only after noise audit identifies enough signal. |
| Family D: population transport JEPA | 0.671 | 0.500 | Defer until benchmark noise and prototype coverage are audited. |

## Monitor Decision
`COUNCIL_EXECUTE`

## Exact Next Amendment
Execute `Family F: bootstrap floor/noise audit`.

## Next Experiment Command Or Target
`F001_BOOTSTRAP_FLOOR_NOISE_AUDIT`
