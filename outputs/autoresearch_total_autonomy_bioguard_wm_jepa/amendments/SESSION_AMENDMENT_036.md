# Session Amendment 036: Multi-Seed Source-State Stability Audit

## Trigger
`C015_DO_NOT_REOPEN_TRAINING_SOURCE_STATE_PRESERVATION_INSUFFICIENT`

## Evidence
C015 decided not to reopen source-state preservation training: the current seed nearly reached the protected recall floor under exact source-state diagnostics, but the learnable online-source-neighborhood proxy stayed below floor and fresh seed support was weaker.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Immediate source-state preservation JEPA training is cooled unless multi-seed diagnostics show stable, learnable, floor-preserving source-state retrieval support.

## New Or Reopened Family
Family G: fresh synthetic seed geometry, source-state stability.

## Exact Next Experiment
`G002_MULTI_SEED_SOURCE_STATE_STABILITY_AUDIT`

## Implementation Tasks
- Reuse current seed and existing seed-1 cache.
- Generate additional tiny CPU latent caches for synthetic seeds 2 and 3.
- For each seed, run nested exact source-state retrieval, teacher-source-neighborhood retrieval, online-source-neighborhood retrieval, and online/teacher source geometry.
- Do not use source-state metadata, condition keys, or exact target keys as model inputs.
- Use metadata only as diagnostic scoring labels/constraints and document that limitation.

## Gates
Diagnostic only; no model promotion and no metric replacement. Reopen training only if online-source-neighborhood retrieval is stable across seeds and preserves the protected floor.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F011 recall-gate redesign memo
