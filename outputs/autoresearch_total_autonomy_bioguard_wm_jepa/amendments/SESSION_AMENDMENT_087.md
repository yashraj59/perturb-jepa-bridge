# Session Amendment 087: Row-Wise Abstention Tier 2 Validation

## Trigger
`F057_ROWWISE_ABSTENTION_READY_FOR_TIER2_DESIGN`

## Evidence
F057 aggregated F055/F056 into six fresh seeds with nonzero row-wise residual abstention, positive transition and delta gaps, preserved recall, and zero held-out retrieval breaks. The effect is small, so the next step must be a promotion-ineligible Tier 2 validation with stricter seed and stability gates.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F058 is Tier 2 validation only and cannot promote.

## New Validation Branch
`F058_ROWWISE_TIER2`

## Implementation Tasks
- Use fresh seeds 22, 23, 24, 25, 26, and 27.
- Use `synth_program_aligned_genetic_lite` with non-exact program action descriptors only.
- Repeat the exact train-only row-wise risk abstention mechanism from F055/F056.
- Export per-seed metrics, train selection grids, held-out query rows, and train traces.
- Require exact held-out floor preservation, no retrieval breaks, positive transition and delta gaps, and gap standard deviation not exceeding the mean gap.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F058 row-wise abstention Tier 2 validation
