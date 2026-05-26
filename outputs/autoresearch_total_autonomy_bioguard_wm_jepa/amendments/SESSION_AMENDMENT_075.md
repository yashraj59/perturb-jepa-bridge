# Session Amendment 075: Operator Train-Heldout Mismatch Audit

## Trigger
`F045_ACTIVE_OPERATOR_WRAPPER_HELDOUT_BELOW_FLOOR`

## Evidence
F045 tested the exact floor-initialized operator wrapper on the active multiseed latent benchmark. It selected tiny nonzero residuals on every seed using train-only metrics, but held-out transition, delta, recall, and retrieval no-regression failed.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F046_OPERATOR_MISMATCH_AUDIT`

## Implementation Tasks
- Read F045 seed rows without retraining.
- Compare train-selected transition/delta/recall gaps against held-out gaps.
- Quantify train-positive/eval-negative mismatch rates.
- Use the result to decide whether the next branch should be a target/representation redesign rather than another residual-cap search.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F046 operator train-heldout mismatch audit
