# Session Amendment 082: Descriptor Margin Gate

## Trigger
`F052_DESCRIPTOR_FAILURE_IS_NEAR_TIE_RETRIEVAL_MARGIN`

## Evidence
F052 localized the F051 failure to one held-out near-tie retrieval row. The nonzero residual slightly improved continuous transition and delta metrics on that row but moved a tiny floor-correct retrieval margin below zero.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F053 is calibration-diagnostic only and cannot promote.

## New Calibration Diagnostic
`F053_DESCRIPTOR_MARGIN_GATE`

## Implementation Tasks
- Read the F051 train-only scale grid.
- Certify a nonzero residual only if the train grid proves continuous floor preservation, no train retrieval breaks, nonnegative train margin movement, and explicit near-tie lower-tail safety diagnostics.
- Default to exact floor fallback if the certificate is missing or fails.
- Do not use F051 held-out broken-row behavior to tune a new residual scale.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F053 train-only descriptor margin gate
