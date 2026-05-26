# Session Amendment 003: Action Support Distance Audit

## Trigger
`F003_ACTION_HETEROGENEITY_IDENTIFIED`

## Evidence
F003 showed one held-out perturbation group is negative under the protected floor while two held-out groups are positive.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Do not force a nonzero residual scale.

## New Or Reopened Family
Family F: Metric And Data Redesign, action descriptor support branch.

## Exact Next Experiment
`F004_ACTION_SUPPORT_DISTANCE_AUDIT`

## Implementation Tasks
- Compute nearest train action-descriptor support for each held-out perturbation.
- Relate train-support distance to floor transition performance.
- Use action labels only for diagnostics.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not launch GEARS-style descriptors or per-action gates until support distance is measured.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F004 action support distance audit
