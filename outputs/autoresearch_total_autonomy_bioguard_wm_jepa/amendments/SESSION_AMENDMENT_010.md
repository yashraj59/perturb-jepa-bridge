# Session Amendment 010: Action Prior Sufficiency Audit

## Trigger
`C003_TRAIN_ONLY_DELTA_SUPPORT_PREDICTS_FAILURE`

## Evidence
C003 validated train-only that weak latent delta support predicts action-ridge floor failures.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, uncertainty-gated, and simple support-distance fixes remain cooled.

## New Or Reopened Family
Family E: Program And Graph Action Priors, diagnostic baseline branch.

## Exact Next Experiment
`E001_ACTION_PRIOR_SUFFICIENCY_AUDIT`

## Implementation Tasks
- Compare source-only, action-only, and source+action ridge baselines.
- Use train-only leave-action-out scoring for model-choice evidence.
- Also score held-out perturbations as diagnostics only.

## Gates
Diagnostic only; no model promotion.

## Do-Not-Run List
Do not implement graph/program descriptors until this audit shows action-prior insufficiency or source-only dominance.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E001 action-prior sufficiency audit
