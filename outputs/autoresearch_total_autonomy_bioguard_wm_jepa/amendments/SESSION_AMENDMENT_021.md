# Session Amendment 021: Retrieval Label Granularity Audit

## Trigger
`F009_SUPPORT_THRESHOLDS_REMOVE_NEGATIVE_TRANSITION_NOT_RECALL`

## Evidence
F009 showed train-only support thresholds can eliminate negative transition pseudoheldout triplets but cannot preserve the current condition-level recall floor. This suggests the retrieval metric or representation dominance may be limiting progress.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending and residual-only transition heads remain cooled.

## New Or Reopened Family
Family R: Retrieval And Metric Redesign, label granularity audit.

## Exact Next Experiment
`R001_RETRIEVAL_LABEL_GRANULARITY_AUDIT`

## Implementation Tasks
- Recompute train-only pseudoheldout retrieval under condition, condition+batch, perturbation, cell-line, and batch labels.
- Determine whether recall failure is label-granularity, perturbation discrimination, or source/cell-line dominance.
- Do not replace protected gates; document diagnostic evidence only.

## Gates
Diagnostic only; no model promotion and no benchmark mutation.

## Do-Not-Run List
Do not promote on alternative retrieval labels.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: R001 retrieval label granularity audit
