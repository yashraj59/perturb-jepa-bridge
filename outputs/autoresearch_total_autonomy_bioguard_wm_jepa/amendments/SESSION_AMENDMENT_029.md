# Session Amendment 029: Nested Source-State Retrieval Calibration Audit

## Trigger
`C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_REACHES_RECALL_FLOOR_DIAGNOSTIC`

## Evidence
C008 reached the protected current-seed condition-recall floor only under same-source-state constrained scoring, while fresh seed scoring remained below floor. The C008 weight was selected oracle-style from the scored triplets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state JEPA training remains cooled until the constrained retrieval rule survives train-only nested calibration.

## New Or Reopened Family
Family C: Representation Repair, nested source-state constrained retrieval calibration.

## Exact Next Experiment
`C009_NESTED_SOURCE_STATE_RETRIEVAL_CALIBRATION_AUDIT`

## Implementation Tasks
- Select endpoint/delta source-state constrained retrieval weights using inner train-action folds.
- Score the selected rule on outer train-only heldout perturbation triplets.
- Repeat the same nested audit on the existing fresh synthetic seed cache when available.
- Keep source-state metadata as an evaluation constraint only, not as a model input.

## Gates
Diagnostic only; no model promotion and no metric replacement. A source-state JEPA candidate requires the nested current-seed result to preserve the protected recall floor and fresh-seed diagnostics to avoid clear collapse.

## Do-Not-Run List
Do not train a source-state JEPA candidate before C009 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C009 nested source-state retrieval calibration audit
