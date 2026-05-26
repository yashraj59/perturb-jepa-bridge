# Session Amendment 088: Row-Wise Transition Stability Audit

## Trigger
`F058_ROWWISE_TIER2_SAFE_BUT_WEAK`

## Evidence
F058 selected nonzero row-wise residuals for every fresh seed, preserved recall, and produced zero held-out retrieval breaks. However, it did not pass Tier 2 because transition-source cosine improvement was too small and unstable: the mean gap was positive but below the seed-level standard deviation, and at least one seed had a negative transition gap.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F059 is a read-only diagnostic and cannot promote.

## New Diagnostic Branch
`F059_ROWWISE_STABILITY_AUDIT`

## Implementation Tasks
- Read F055, F056, and F058 seed rows.
- Isolate whether F058 weakness comes from retrieval breaks, delta collapse, recall regression, or transition-source cosine instability.
- Compare train-selected transition gains to held-out transition gaps.
- Recommend the next mechanism only after localization.
- Do not fit models and do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F059 row-wise transition stability audit
