# Session Amendment 110: Train-Only Delta-Calibrated JEPA Wrapper

## Trigger
`F080_FULL_JEPA_WRAPPER_MIXED_OR_INCONCLUSIVE`

## Evidence
F080 was not a clean pass: it improved image-teacher transition and kept recall, but delta cosine remained below the gate. The magnitude ratio was also high, suggesting a transition endpoint that is useful but geometrically miscalibrated in delta space.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F081 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F081_DELTA_CALIBRATED_JEPA_WRAPPER`

## Implementation Tasks
- Reuse the F080 real ProgramBootstrapJEPA path.
- Fit a train-only ridge correction from JEPA predicted delta to train image-PCA teacher delta.
- Apply the fixed calibrator to heldout perturbation JEPA predicted deltas.
- Report raw versus calibrated transition, delta cosine, recall, rank, magnitude, action-negative gap, and train/eval calibration gap.
- Do not use `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F081 repairs heldout delta without losing transition/recall, convert the calibrator into an internal JEPA delta-head or loss. If it overfits train direction, pivot to action descriptor adequacy or environment-stable delta targets.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F081 train-only JEPA delta calibrator
