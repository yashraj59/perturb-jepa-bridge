# Session Amendment 020: Support-Threshold Split Redesign Audit

## Trigger
`F008_SPLIT_GEOMETRY_UNSTABLE_REDESIGN_BENCHMARK_BEFORE_MORE_CANDIDATES`

## Evidence
F008 found train-only pseudoheldout geometry instability: several pseudoheldout triplets had negative transition and nearly all triplets fell below the protected recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending and residual-only operator searches remain cooled until benchmark geometry is clarified.

## New Or Reopened Family
Family F: Metric And Data Redesign, support-threshold split redesign audit.

## Exact Next Experiment
`F009_SUPPORT_THRESHOLD_SPLIT_REDESIGN_AUDIT`

## Implementation Tasks
- Sweep train-only support-score thresholds over F008 pseudoheldout triplets.
- Identify whether support thresholds can eliminate negative transition without destroying recall.
- Use the result to decide between benchmark redesign and retrieval-label metric audit.

## Gates
Diagnostic only; no model promotion and no mutation of the locked held-out split.

## Do-Not-Run List
Do not use these thresholds to select a candidate model on the existing test split.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F009 support-threshold split redesign audit
