# Session Amendment 047: F020 Learned Environment Risk Gate

## Trigger
`F019_NO_SIMPLE_SECOND_GATE_REPAIRS_F018_FAILURE`

## Evidence
F017 showed oracle safe-blend capacity, F018 showed a train-only simple proxy can improve mean transition while violating seed0 recall, and F019 showed no simple second feature threshold repairs that failure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Or Reopened Family
Family F: Metric And Data Redesign, learned risk-gate branch.

## Exact Next Experiment
`F020_LEARNED_ENVIRONMENT_RISK_GATE`

## Implementation Tasks
- Train a tiny linear risk score on train-only perturbation-fold safe/unsafe labels.
- Use out-of-fold train scores to select candidate, alpha, and threshold.
- Freeze the selected rule and score held-out eval rows.
- Enforce per-seed floor preservation for transition, delta cosine, and condition recall.

## Gates
Diagnostic only. If F020 cannot preserve the per-seed floor, cool environment blending and pivot away from risk-gated environment residuals.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F020 learned environment risk gate
