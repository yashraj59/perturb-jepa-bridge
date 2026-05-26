# Session Amendment 027: Protected Metric Disagreement Audit

## Trigger
`G001_FRESH_SYNTHETIC_SEED_CONFIRMS_COMPOSITE_PARTIAL_REPAIR_BELOW_FLOOR`

## Evidence
G001 confirmed that endpoint+delta composite retrieval gives a repeatable partial repair on a fresh synthetic seed, but recall@1 remains below the protected floor even when continuous transition and delta-cosine metrics improve.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate JEPA architecture reopening remains cooled until the metric disagreement is explicitly documented. This does not alter protected gates.

## New Or Reopened Family
Family F: Metric And Data Redesign, protected metric disagreement audit.

## Exact Next Experiment
`F010_METRIC_DISAGREEMENT_AUDIT`

## Implementation Tasks
- Compare recall@1 with transition improvement and delta-cosine across current and fresh synthetic seeds.
- Quantify whether endpoint+delta composite gains replicate without reaching the protected recall floor.
- Write a report that preserves current gates but explains whether recall is acting as an unstable proxy.

## Gates
Diagnostic only; no model promotion and no metric replacement.

## Do-Not-Run List
Do not train a heavier model or relax the protected floor based on this audit alone.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F010 protected metric disagreement audit
