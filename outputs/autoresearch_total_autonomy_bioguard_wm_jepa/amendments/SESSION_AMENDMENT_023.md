# Session Amendment 023: Perturbation-Centered Delta Retrieval Audit

## Trigger
`C004_CELL_LINE_SOURCE_DOMINANCE_CONFIRMED_PERTURBATION_DELTA_UNDERREPRESENTED`

## Evidence
C004 confirmed predicted endpoints are dominated by cell-line/source structure, while true deltas are more perturbation-structured than predicted deltas.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Support-gated blending, metric relabeling, and endpoint-only residual corrections remain cooled.

## New Or Reopened Family
Family C: Representation Repair, perturbation-centered delta retrieval audit.

## Exact Next Experiment
`C005_PERTURBATION_CENTERED_DELTA_RETRIEVAL_AUDIT`

## Implementation Tasks
- Compare endpoint retrieval with delta-space retrieval on train-only pseudoheldout triplets.
- Score perturbation, condition, and cell-line labels without replacing protected gates.
- Decide whether a future candidate should add a perturbation-centered delta retrieval objective.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a new JEPA head until delta-space retrieval evidence is recorded.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C005 perturbation-centered delta retrieval audit
