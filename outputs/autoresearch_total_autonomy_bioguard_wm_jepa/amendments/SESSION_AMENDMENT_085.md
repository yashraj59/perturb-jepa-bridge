# Session Amendment 085: Row-Wise Abstention Replication

## Trigger
`F055_ROWWISE_ABSTENTION_SAFE_NONZERO_DIAGNOSTIC`

## Evidence
F055 selected nonzero row-wise residuals with train-only abstention, improved held-out transition and delta metrics, preserved recall, and had zero held-out broken retrieval rows. This is promising but not promotable.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F056 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F056_ROWWISE_ABSTENTION_REPLICATION`

## Implementation Tasks
- Use fresh seeds 19, 20, and 21.
- Repeat the exact F055 train-only row-wise abstention mechanism.
- Score held-out rows only after the train-only rule is fixed.
- Preserve the full JEPA identity and leakage constraints.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F056 independent row-wise abstention replication
