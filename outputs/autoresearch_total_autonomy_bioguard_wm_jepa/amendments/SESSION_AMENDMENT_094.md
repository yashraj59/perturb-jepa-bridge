# Session Amendment 094: Train-Only Action-Heldout Calibration

## Trigger
`F064_TRAIN_SELECTOR_OVERFITS_CONTINUOUS_TRANSITION_GAINS`

## Evidence
F064 showed that train-selected residual gains are optimistic: train transition gap was positive while held-out transition and delta gaps were negative. The failure was continuous transition/delta generalization, not retrieval-row breakage.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F065 is calibration diagnostic work and cannot promote.

## New Diagnostic Branch
`F065_ACTION_HELDOUT_CALIBRATION`

## Implementation Tasks
- Reuse the descriptor-aligned synthetic benchmark and low-compute real JEPA path.
- Within train rows only, hold out one perturbation ID at a time.
- Fit ridge/head/rules on the remaining train perturbations.
- Select a residual cone only if it preserves transition, delta, recall, and retrieval on the held-out train perturbation folds.
- If no cone passes, default exactly to the protected floor.
- Score the real held-out perturbation split once after train-only calibration.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F065 train-only action-heldout calibration
