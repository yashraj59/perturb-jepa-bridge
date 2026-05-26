# Session Amendment 025: Nested Composite Retrieval Calibration Audit

## Trigger
`C006_COMPOSITE_RETRIEVAL_PARTIALLY_REPAIRS_CONDITION_RECALL_BELOW_FLOOR`

## Evidence
C006 showed that endpoint+delta composite scoring partially repairs condition-level retrieval, but the best diagnostic weight was selected on the same pseudoheldout triplets it scored and remained below the protected recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Endpoint-only, residual-only, support-gated transition, and uncalibrated composite scoring remain cooled.

## New Or Reopened Family
Family C: Representation Repair, train-only nested composite calibration.

## Exact Next Experiment
`C007_NESTED_COMPOSITE_RETRIEVAL_CALIBRATION_AUDIT`

## Implementation Tasks
- Select the endpoint/delta retrieval weight using only inner train-action folds.
- Score the selected rule on outer train-only pseudoheldout perturbation triplets.
- Compare selected composite scoring with endpoint-only and delta-only scoring without changing protected gates.

## Gates
Diagnostic only; no model promotion and no metric replacement. A future dual-objective JEPA candidate requires the nested rule to beat endpoint-only scoring and approach the protected condition-recall floor without held-out peeking.

## Do-Not-Run List
Do not launch a dual-objective JEPA candidate until nested composite calibration is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C007 nested composite retrieval calibration audit
