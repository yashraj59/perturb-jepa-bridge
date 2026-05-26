# Session Amendment 063: Operator-Initialized Transition Head

## Trigger
`F033_CONSERVATIVE_GATE_ZERO_FALLBACK`

## Evidence
F033 proved that conservative post-hoc residual deployment is safe only by selecting zero residual. The failure mode is architectural: the JEPA transition head is not initialized at the local train-only operator floor. F034 therefore makes floor preservation part of the predictor, not a later rescue step.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Branch
`F034_OPERATOR_INITIALIZED_TRANSITION_HEAD`

## Implementation Tasks
- Train the same low-compute real JEPA representation path.
- Fit the same-representation train-only ridge floor on frozen train latents.
- Initialize a transition head exactly to that ridge floor.
- Add a zero-initialized action-conditioned residual branch.
- Train only the residual branch with endpoint, delta-direction, and floor-preservation losses.
- Select residual scale using train rows only, then score held-out rows.
- Keep action input as non-exact program descriptors only.

## Decision Use
If F034 preserves the held-out local floor with nonzero residuals, design Tier 3/no-regression validation. If it still falls back to zero or violates the local floor, pivot away from residual scaling toward a different operator family or representation target.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F034 operator-initialized floor-preserving transition head
