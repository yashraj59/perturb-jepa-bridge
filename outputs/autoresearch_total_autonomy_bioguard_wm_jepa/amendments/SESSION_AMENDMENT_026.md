# Session Amendment 026: Fresh Synthetic Seed Latent-Cache Audit

## Trigger
`C007_NESTED_COMPOSITE_CALIBRATION_PARTIAL_GAIN_BELOW_FLOOR`

## Evidence
C007 showed the endpoint+delta composite rule is train-internally calibratable and improves over endpoint-only retrieval, but it remains materially below the protected recall floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Uncalibrated composite scoring, residual-only predictors, and immediate dual-objective JEPA reopening are cooled until seed-specific geometry is checked.

## New Or Reopened Family
Family G: Fresh Synthetic Seed Geometry.

## Exact Next Experiment
`G001_FRESH_SYNTHETIC_SEED_LATENT_CACHE_AUDIT`

## Implementation Tasks
- Build a tiny CPU latent cache for synthetic seed 1 using the existing Phase 4 cache script.
- Reproduce the action-ridge floor on that fresh synthetic seed.
- Re-run the train-only endpoint+delta composite geometry audit on that fresh seed.
- Decide whether the current failure is seed-specific geometry, general representation saturation, or enough evidence for a JEPA target change.

## Gates
Diagnostic only; no model promotion and no metric replacement. The protected model of record and protected transition floor remain unchanged.

## Do-Not-Run List
Do not train a heavier model or use GPU for this audit.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: G001 fresh synthetic seed latent-cache audit
