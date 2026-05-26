# Session Amendment 039: Multi-Seed Benchmark Registry

## Trigger
`F012_BENCHMARK_REDESIGN_REQUIRED_BEFORE_MORE_ARCHITECTURE`

## Evidence
F012 requires a new named multi-seed benchmark before more architecture search. The old seed0 benchmark remains locked as historical reference, but should not be silently modified.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## New Benchmark Name
`synthetic_genetic_anchor_lite_multiseed_v1`

## Exact Next Experiment
`F013_MULTISEED_BENCHMARK_REGISTRY`

## Implementation Tasks
- Use seed0-4 latent caches.
- Compute train-only baselines per seed: full action-ridge floor, source-only ridge, action-only ridge, mean-delta null, and source-as-target null.
- Aggregate mean/std and propose future gates without applying them to the old benchmark.
- Preserve leakage controls: fit only on train split per seed.

## Gates
Registry only. No model promotion. No old-gate mutation.

## Do-Not-Run List
Do not run new architecture until registry is complete.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F013 multiseed benchmark registry
