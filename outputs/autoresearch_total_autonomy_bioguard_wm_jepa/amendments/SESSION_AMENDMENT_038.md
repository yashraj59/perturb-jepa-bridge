# Session Amendment 038: Benchmark Redesign Brief Before More Architecture

## Trigger
`F011_RECALL_FLOOR_SEED_SPECIFIC_CONTINUOUS_METRICS_STABLE`

## Evidence
F011 showed the protected seed0 recall floor is seed-specific: seed0 recall@1 is `0.481481`, while fresh seeds average `0.296296` and max out at `0.333333`. Continuous transition and delta-cosine metrics do not show the same seed0 advantage.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it. No gate changes are authorized by this amendment alone.

## Retired / Cooled Families
Architecture search against the current seed0 recall floor is cooled until a benchmark redesign is documented. Source-state preservation remains cooled.

## New Or Reopened Family
Family F: metric/data redesign documentation.

## Exact Next Experiment
`F012_BENCHMARK_REDESIGN_BRIEF`

## Implementation Tasks
- Write a self-contained redesign brief that separates locked current benchmark facts from proposed future benchmark changes.
- Preserve the old protected model and old protected gate as historical references.
- Propose any new benchmark under a new name, with multi-seed baseline registry and no silent split/evaluator mutation.

## Gates
Documentation only. Do not promote any model. Do not change current gates/evaluators.

## Do-Not-Run List
Do not run new architecture under a changed gate. Do not overwrite current split semantics. Do not use eval targets for fitting or selection.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F012 benchmark redesign brief
