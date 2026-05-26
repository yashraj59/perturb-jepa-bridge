# Session Amendment 065: Retrieval Failure Localization

## Trigger
`F035_INNER_VAL_GATE_STILL_HELDOUT_BELOW_FLOOR`

## Evidence
F035 selected nonzero residual scales using train-only inner validation and improved held-out transition and delta cosine over the local ridge floor, but it still violated held-out recall no-regression. The strongest failure was a seed-level condition recall drop, so the next move must inspect retrieval ranks and margins before changing the JEPA architecture.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F036_RETRIEVAL_FAILURE_LOCALIZATION`

## Implementation Tasks
- Rerun the deterministic F035 low-compute setup on seeds 0-4.
- Keep action input as non-exact program descriptors only.
- For each held-out query, compare floor endpoint retrieval and selected residual endpoint retrieval.
- Report top-1 labels, ranks, correct-vs-best-wrong margins, rank changes, residual norms, transition-gain changes, and delta-cosine changes.
- Use held-out labels for diagnostic localization only; do not use them for residual selection or promotion.

## Decision Use
If failures are mostly near-tie margin instability, design a train-only retrieval-margin risk gate or dual endpoint+delta objective. If failures are real endpoint boundary crossings, pivot to retrieval-aware/prototype-aligned transition constraints. Do not train longer by default.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F036 retrieval-failure localization
