# Session Amendment 034: Online-Teacher Source Geometry Audit

## Trigger
`C013_ONLINE_SOURCE_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C013 showed online/context source neighborhoods improve retrieval but remain below floor. C012 showed online/context z_bio has much stronger source-state purity than teacher z_bio, so the next question is whether the teacher target representation attenuates source-state structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preserving JEPA training remains cooled until online-vs-teacher source geometry is documented.

## New Or Reopened Family
Family C: Representation Repair, online-vs-teacher source geometry audit.

## Exact Next Experiment
`C014_ONLINE_TEACHER_SOURCE_GEOMETRY_AUDIT`

## Implementation Tasks
- Compare online/context z_bio with teacher z_bio for source-state purity, eta-squared, centroid accuracy, rank, norm, and pairwise separation.
- Measure online-teacher row alignment and whether the online-minus-teacher residual carries source-state information.
- Repeat the same diagnostic on the existing fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA candidate before C014 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C014 online-vs-teacher source geometry audit
