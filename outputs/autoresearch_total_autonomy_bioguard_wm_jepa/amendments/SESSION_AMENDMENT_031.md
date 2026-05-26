# Session Amendment 031: Source-Neighborhood Purity Audit

## Trigger
`C010_SOURCE_LATENT_NEIGHBORHOOD_RETRIEVAL_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
C010 showed that source-latent nearest-neighborhood retrieval is a partial repair but substantially underperforms exact source-state constrained retrieval. The failure could come from impure source latent neighborhoods or from perturbation-delta insufficiency even with good neighborhoods.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-latent neighborhood JEPA training remains cooled until the source-neighborhood failure mode is localized.

## New Or Reopened Family
Family C: Representation Repair, source-neighborhood purity audit.

## Exact Next Experiment
`C011_SOURCE_NEIGHBORHOOD_PURITY_AUDIT`

## Implementation Tasks
- Measure same-cell-line/source-state purity and coverage of source latent nearest-neighborhoods.
- Compare exact source-state constrained nested recall with source-latent neighborhood nested recall.
- Run the same purity summary on the fresh synthetic seed cache when available.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state preserving JEPA objective before C011 is documented.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C011 source-neighborhood purity audit
