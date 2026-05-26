# Session Amendment 067: Oracle Safe-Scale Capacity Audit

## Trigger
`F037_RETRIEVAL_MARGIN_GATE_STILL_HELDOUT_BELOW_FLOOR`

## Evidence
F037 improved mean transition, delta cosine, and recall, but still violated a per-seed held-out recall gate. Because the train-only gate had zero inner-validation broken rows, the next question is whether safe nonzero residual deployment exists at all on held-out rows.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F038_ORACLE_SAFE_SCALE_CAPACITY`

## Implementation Tasks
- Rerun the same floor-initialized transition head.
- Evaluate a fixed residual scale grid on held-out rows for diagnosis only.
- Mark a scale safe only if transition, delta cosine, recall, and zero broken floor-correct retrieval rows all preserve the floor.
- Do not use oracle scales for promotion, calibration, or model selection.

## Decision Use
If nonzero safe scales exist, the next branch should learn a train-only proxy for that capacity. If safe scales are zero or absent, retire residual scaling and pivot to a retrieval-aware transition objective or representation redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F038 held-out oracle safe-scale capacity audit
