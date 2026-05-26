# Session Amendment 068: Train-Only Safe-Scale Proxy

## Trigger
`F038_ORACLE_SAFE_NONZERO_SCALE_CAPACITY_EXISTS`

## Evidence
F038 showed nonzero safe residual scales exist for every seed under oracle held-out selection. The remaining problem is not representation identity or operator signal; it is selecting a safe scale without held-out leakage.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Calibration Branch
`F039_TRAIN_PROXY_SAFE_SCALE`

## Implementation Tasks
- Reuse the F038 floor-initialized transition head and low-compute setting.
- Select scale using train rows only.
- Require no aggregate train metric regression, zero broken train retrieval rows, no erosion on floor-correct near-tie train rows, and nonnegative lower-tail margin change.
- Score held-out rows only after train-only scale selection.
- Do not use held-out oracle scales for fitting, selection, or promotion.

## Decision Use
If F039 preserves held-out floor with nonzero residuals, design Tier 3/no-regression validation. If it falls back to zero or still violates held-out recall, pivot to a retrieval-aware transition loss rather than more scale heuristics.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F039 train-only safe-scale proxy
