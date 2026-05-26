# Session Amendment 024: Endpoint+Delta Composite Retrieval Audit

## Trigger
`C005_DELTA_SPACE_PARTIALLY_REPAIRS_PERTURBATION_RECALL_BUT_BELOW_FLOOR`

## Evidence
C005 showed delta-space retrieval improves perturbation recall relative to endpoint retrieval, but remains below the protected condition-level recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Endpoint-only, residual-only, and support-gated transition modifications remain cooled.

## New Or Reopened Family
Family C: Representation Repair, endpoint+delta composite retrieval audit.

## Exact Next Experiment
`C006_ENDPOINT_DELTA_COMPOSITE_RETRIEVAL_AUDIT`

## Implementation Tasks
- Combine endpoint similarity and delta similarity with fixed diagnostic weights.
- Score condition and perturbation retrieval on train-only pseudoheldout triplets.
- Decide whether a future dual-objective JEPA target is justified.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not use composite scoring to claim a promoted model.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C006 endpoint+delta composite retrieval audit
