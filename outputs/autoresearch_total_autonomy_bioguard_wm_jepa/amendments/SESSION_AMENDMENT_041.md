# Session Amendment 041: BGWM003 Multiseed Action-AdaLN Sanity

## Trigger
`F014_MULTISEED_V1_BENCHMARK_ACTIVATED_FOR_SEARCH`

## Evidence
F014 explicitly activated `synthetic_genetic_anchor_lite_multiseed_v1` for future architecture search and named `BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY` as the next experiment. The old seed0 benchmark remains locked and historical.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Active Benchmark Gates
- Mean transition improvement must meet the active F014 primary transition gate.
- Mean delta cosine must meet the active F014 primary delta-cosine gate.
- Mean condition recall@1 is secondary/no-regression under the active F014 gate.
- No Tier 1 result can promote the model of record.

## New Or Reopened Family
Phase 8 v3 action-AdaLN + RoPE floor-preserving residual sanity under multiseed v1.

## Exact Next Experiment
`BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY`

## Implementation Tasks
- Run the existing floor-preserving Action-AdaLN + RoPE residual path on seeds 0-4.
- Use each seed's train split only for floor and residual calibration.
- Score eval rows only after train-only calibration.
- Report per-seed and mean metrics under the F014 gates.
- Keep the result non-promotional.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: BGWM003 multiseed Action-AdaLN sanity
