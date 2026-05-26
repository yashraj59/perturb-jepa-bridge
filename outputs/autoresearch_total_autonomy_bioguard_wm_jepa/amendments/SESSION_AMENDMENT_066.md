# Session Amendment 066: Train-Only Retrieval-Margin Gate

## Trigger
`F036_RECALL_FAILURE_PRIMARILY_MARGIN_INSTABILITY`

## Evidence
F036 localized the F035 recall failure: all broken rows were near-tie nearest-neighbor flips, while transition gain and delta-cosine change stayed positive. This supports a train-only retrieval-margin safety gate rather than longer training or a post-hoc recall relaxation.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Calibration Branch
`F037_RETRIEVAL_MARGIN_GATE`

## Implementation Tasks
- Reuse the F035 floor-initialized transition head.
- Split train rows into inner-fit and inner-validation sets.
- Select residual scale on inner-validation only.
- Require transition, delta cosine, aggregate recall, and zero broken floor-correct retrieval rows on inner validation.
- Refit the head on full train and score held-out rows without using held-out labels for selection.

## Decision Use
If F037 preserves held-out local floor with nonzero residuals, design a stricter Tier 3/no-regression validation. If it falls back to zero, pivot to a retrieval-aware transition objective. If it still violates held-out recall, the inner validation gate is insufficient and should be retired.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F037 train-only retrieval-margin gate
