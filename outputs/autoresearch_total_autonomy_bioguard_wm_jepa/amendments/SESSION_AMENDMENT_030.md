# Session Amendment 030: Source-Latent Neighborhood Retrieval Audit

## Trigger
`C009_NESTED_SOURCE_STATE_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C009 showed exact source-state constrained retrieval survives nested calibration only as a near miss on the current seed and remains below floor on the fresh seed. Exact source-state metadata should not become a model input shortcut.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state metadata-conditioned JEPA training remains cooled.

## New Or Reopened Family
Family C: Representation Repair, source-latent neighborhood retrieval.

## Exact Next Experiment
`C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_AUDIT`

## Implementation Tasks
- Build retrieval masks from source latent nearest-neighborhood fractions, not cell-line metadata.
- Select neighborhood fraction and endpoint/delta score weight using inner train-action folds.
- Score selected rules on outer train-only heldout perturbation triplets.
- Repeat on the existing fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement. A future JEPA candidate requires the source-latent proxy to approach or preserve the protected recall floor without exact metadata inputs.

## Do-Not-Run List
Do not train a source-state JEPA candidate before C010 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C010 source-latent neighborhood retrieval audit
