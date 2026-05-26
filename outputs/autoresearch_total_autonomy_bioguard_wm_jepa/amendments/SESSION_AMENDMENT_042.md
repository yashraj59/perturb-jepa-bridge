# Session Amendment 042: F015 Multi-Environment Transition Audit

## Trigger
`BGWM003_NO_IMPROVEMENT_ZERO_RESIDUAL_FLOOR_FALLBACK`

## Evidence
BGWM003 passed the active multiseed gates only through exact floor fallback. All seeds selected residual scale zero, and fold traces showed negligible residual signal with zero action-negative separation.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Active Benchmark
`synthetic_genetic_anchor_lite_multiseed_v1` remains active for search. The old seed0 benchmark remains locked and historical.

## New Or Reopened Family
Family F / environment-stable transition diagnostics.

## Exact Next Experiment
`F015_MULTI_ENVIRONMENT_TRANSITION_AUDIT`

## Implementation Tasks
- Compare per-seed action-ridge floor against pooled full ridge, pooled source-only ridge, pooled action-only ridge, train-only environment-centered ridge, and leave-one-seed-out full ridge.
- Use train splits only for all fitting and centering.
- Use eval rows only for scoring.
- Treat seed/environment labels as diagnostic grouping variables, not biological shortcuts.

## Gates
Diagnostic only. A pooled/environment baseline cannot promote the model of record. It can justify a future environment-stable JEPA amendment if it shows train-only headroom.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F015 multi-environment transition audit
