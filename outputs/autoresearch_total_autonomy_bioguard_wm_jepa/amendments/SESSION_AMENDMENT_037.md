# Session Amendment 037: Protected-Floor Seed-Stability Audit

## Trigger
`G002_FRESH_REPLICATION_CONFIRMS_SOURCE_PROXY_INSTABILITY_BELOW_FLOOR`

## Evidence
G002 showed the deployable online-source proxy stays below the protected recall floor on seeds 1-4. It also showed the train-only floor recall itself is far lower on fresh synthetic seeds than on seed0. Before more architecture work, determine whether the recall gate is a stable protected target or a seed-specific artifact.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it. No model is promoted by this diagnostic.

## Retired / Cooled Families
Source-state preservation training remains cooled. Fresh-cache diagnostics have shown insufficient deployable proxy stability.

## New Or Reopened Family
Family F: metric/data redesign, protected-floor seed-stability audit.

## Exact Next Experiment
`F011_PROTECTED_FLOOR_SEED_STABILITY_AUDIT`

## Implementation Tasks
- Compare seed0 protected floor metrics against fresh-seed floor metrics from G002.
- Quantify recall, transition, delta-cosine, rank, and magnitude variability across synthetic seeds.
- Decide whether the architecture loop should continue under the current recall gate or pivot to benchmark/metric redesign documentation.

## Gates
Diagnostic only. Do not change the protected floor, split, evaluator, or promotion criteria in this experiment.

## Do-Not-Run List
Do not train new architecture. Do not replace recall gate. Do not use eval target rows for fitting or selection.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F011 protected-floor seed-stability audit
