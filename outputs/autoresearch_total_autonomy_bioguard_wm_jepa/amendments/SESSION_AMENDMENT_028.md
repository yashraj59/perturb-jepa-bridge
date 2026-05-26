# Session Amendment 028: Source-State Constrained Retrieval Audit

## Trigger
`F010_RECALL_GATE_SEED_UNSTABLE_CONTINUOUS_METRICS_DISAGREE`

## Evidence
F010 showed that recall@1 is unstable relative to continuous transition quality, while endpoint+delta composite gains replicate across the original and fresh synthetic seeds.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate heavy JEPA training remains cooled until retrieval failure is separated into cross-source-state competition versus perturbation-level separation failure.

## New Or Reopened Family
Family C: Representation Repair, source-state constrained perturbation retrieval audit.

## Exact Next Experiment
`C008_SOURCE_STATE_CONSTRAINED_RETRIEVAL_AUDIT`

## Implementation Tasks
- Restrict diagnostic retrieval scoring to candidates with the same source-state/cell-line metadata.
- Compare constrained endpoint, delta, and endpoint+delta composite retrieval.
- Run the same diagnostic on the fresh seed cache when available.
- Do not use the metadata constraint as a model input or promotion path.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a source-state JEPA candidate until this diagnostic shows the reachable ceiling.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: C008 source-state constrained perturbation retrieval audit
