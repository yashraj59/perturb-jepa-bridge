# Session Amendment 086: Row-Wise Abstention Synthesis

## Trigger
`F056_ROWWISE_ABSTENTION_REPLICATES_SAFE_NONZERO`

## Evidence
F055 and F056 both selected train-only row-wise nonzero residuals, improved held-out transition and delta metrics, preserved recall, and produced zero held-out retrieval breaks. The evidence is still diagnostic and synthetic-only.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F057 is synthesis only and cannot promote.

## New Diagnostic Synthesis
`F057_ROWWISE_SYNTHESIS`

## Implementation Tasks
- Aggregate F055 and F056 seed-level rows.
- Compute six-seed mean, standard deviation, minimum floor gaps, active fraction, and held-out retrieval-break count.
- Decide whether row-wise abstention is ready for a formal non-promoting Tier 2 design.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F057 six-seed row-wise synthesis
