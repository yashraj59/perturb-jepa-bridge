# Session Amendment 062: Conservative Floor Gate

## Trigger
`F032_NONZERO_RESIDUAL_DISCARDED_HELDOUT_BELOW_LOCAL_FLOOR`

## Evidence
F032 selected nonzero train-only residual scales and improved mean held-out transition, delta cosine, and recall over the local floor, but one seed fell below the local recall floor. The next experiment should not train longer; it should make calibration stricter and default to the exact local floor unless train evidence has slack.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F033_CONSERVATIVE_FLOOR_GATE`

## Implementation Tasks
- Reuse the F032 deterministic training setup.
- Use a smaller residual scale grid.
- Require positive train-side transition, delta cosine, and recall slack for nonzero residual deployment.
- Evaluate held-out rows only after train-only selection.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F033 preserves the held-out local floor with nonzero residuals, design a Tier 3/no-regression amendment. If it falls back to zero, pivot to operator-initialized direct predictor training. If it still violates local floor, discard this calibration family.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F033 conservative train-only residual gate
