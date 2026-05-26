# Session Amendment 096: Close Residual-Selector Family Under Current Data Contract

## Trigger
`F066_TRAIN_ACTION_HELDOUT_STILL_OPTIMISTIC_FOR_REAL_HELDOUT`

## Evidence
F055-F066 tested row-wise abstention, Tier 2 validation, projection, residual cones, train-only cone selection, leave-perturbation-out train calibration, and mismatch audits. Oracle residual capacity exists, but train-only selectors remain optimistic and fall below the protected floor on real held-out perturbation scoring.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F067 is synthesis/closure work and cannot promote.

## Family Status
Residual selector variants under the current descriptor-aligned data contract are closed/cooled:
- scalar cap search;
- row-wise abstention variants;
- pure floor-direction projection;
- residual-cone selectors;
- train action-heldout calibration gates.

## Next Family To Open
`F068_DATA_CONTRACT_CALIBRATION_REDIRECT`

## Implementation Tasks For Next Family
- Design a read-only benchmark/data-contract diagnostic before new model training.
- Determine why train-heldout calibration is optimistic for real heldout perturbations despite similar program-direction support.
- Compare random perturbation holdout, stratified program holdout, and extrapolative perturbation-index holdout on the synthetic generator.
- Keep all existing held-out results locked; do not mutate old benchmarks.
- If redesign is needed, create a new named synthetic benchmark rather than rewriting prior data.
- Do not promote any model from F067/F068.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F067 residual-selector family closure
