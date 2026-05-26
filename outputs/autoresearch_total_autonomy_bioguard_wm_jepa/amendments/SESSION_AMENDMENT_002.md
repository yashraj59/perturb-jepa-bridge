# Session Amendment 002: Action/Condition Heterogeneity Audit

## Trigger
`F001_F002_METRIC_HETEROGENEITY_AFTER_PHASE8_FAILURE`

## Evidence
F001 showed the protected floor transition estimate is noisy, and F002 showed floor performance differs sharply by effect-size tertile.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only predictor families remain cooled. Do not force a nonzero residual scale.

## New Or Reopened Family
Family F: Metric And Data Redesign, action/condition heterogeneity branch.

## Exact Next Experiment
`F003_ACTION_CONDITION_HETEROGENEITY_AUDIT`

## Implementation Tasks
- Stratify held-out scoring by perturbation, condition, cell line, and batch labels.
- Use labels only for diagnostics, never as model inputs.
- Identify whether failures concentrate in specific perturbations or support cells.

## Gates
Diagnostic only; no model promotion. Leakage and identity reports remain required.

## Do-Not-Run List
Do not launch more residual architecture until heterogeneity is explained.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F003 action/condition heterogeneity audit
