# Session Amendment 095: Action-Heldout Calibration Mismatch Audit

## Trigger
`F065_ACTION_HELDOUT_GATE_HELDOUT_BELOW_FLOOR`

## Evidence
F065 passed train-only leave-perturbation-out inner calibration but still fell below the protected floor on real held-out perturbations. That means even split-aware train calibration is not currently sufficient.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F066 is read-only diagnostic work and cannot promote.

## New Diagnostic Branch
`F066_ACTION_HELDOUT_MISMATCH_AUDIT`

## Implementation Tasks
- Read F065 seed, inner-fold, and cone-summary artifacts.
- Compare inner train-heldout gaps against real heldout gaps.
- Count train-action-positive but real-negative seeds.
- Regenerate the same synthetic benchmark only for direction-support diagnostics.
- Compare train inner perturbation support with real heldout perturbation support.
- Decide whether the next family should pivot away from residual selectors.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F066 action-heldout mismatch audit
