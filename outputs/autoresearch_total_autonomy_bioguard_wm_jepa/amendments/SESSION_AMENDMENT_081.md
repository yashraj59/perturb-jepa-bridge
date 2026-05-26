# Session Amendment 081: Descriptor Near-Tie Retrieval Audit

## Trigger
`F051_DESCRIPTOR_ALIGNED_ACTION_CONTRACT_HELDOUT_BELOW_FLOOR`

## Evidence
F051 used non-exact program actions and a real small JEPA path on the descriptor-aligned benchmark. It passed the active continuous gates and selected nonzero residual scale on all fresh seeds, but failed exact local floor preservation because one held-out retrieval row flipped under a tiny margin.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F052 is diagnostic-only and cannot promote.

## New Diagnostic Branch
`F052_DESCRIPTOR_NEAR_TIE_AUDIT`

## Implementation Tasks
- Read F051 metrics, seed rows, train scale grid, and query-level retrieval rows.
- Do not train, refit, recalibrate, or choose a new residual scale.
- Quantify whether broken held-out rows are near-ties and whether continuous transition/delta metrics still improved on those rows.
- Decide whether the next branch should be a train-only retrieval-margin safety gate rather than a larger JEPA.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F052 descriptor near-tie retrieval audit
