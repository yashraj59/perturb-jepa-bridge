# Session Amendment 001: Metric/Data Redesign Before More Residuals

## Trigger
`BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN`

## Evidence
Action-AdaLN + RoPE preserved the floor but train-only calibration selected zero residual.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only spectral, kernel, program, and action-AdaLN predictors are cooled until a diagnostic identifies residual headroom.

## New Or Reopened Family
Family F: Metric And Data Redesign.

## Exact Next Experiment
`F001_BOOTSTRAP_FLOOR_NOISE_AUDIT`

Continuation already executed under this amendment:
`F002_HELDOUT_DIFFICULTY_STRATIFICATION`

## Implementation Tasks
- Bootstrap the protected floor metrics on held-out perturbation scoring.
- Estimate whether claimed residual effects are below evaluation noise.
- Do not modify locked splits or use eval targets for selection.

## Gates
Diagnostic only; no model promotion. Leakage and identity reports remain required.

## Do-Not-Run List
Do not force nonzero residual scale. Do not launch full JEPA wrapper until a frozen-latent residual passes calibration.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: Family F: bootstrap floor/noise audit
