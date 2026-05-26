# Session Amendment 016: Support Threshold Oracle Audit

## Trigger
`E005_PREDICTED_SUPPORT_OVERCONFIDENT_ON_BAD_ACTION`

## Evidence
E005 found predicted support is correlated with held-out transition success but remains overconfident on the worst action. Before designing a new calibration mechanism, test whether this support-gated source/action family has any held-out capacity at all.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families
Residual-only, global blending, condition-level blending, and the current support-aware selected candidate remain non-promotable.

## New Or Reopened Family
Family E: Program And Graph Action Priors, support-threshold oracle capacity audit.

## Exact Next Experiment
`E006_HELDOUT_SUPPORT_THRESHOLD_ORACLE_AUDIT`

## Implementation Tasks
- Sweep support thresholds and source-only blend weights on held-out rows for audit only.
- Report whether any candidate could preserve the protected floor on transition, recall, and delta cosine.
- Do not use the oracle-selected candidate as a promoted model or calibration result.

## Gates
Diagnostic only. If no oracle candidate preserves the floor, retire this support-gated blend family.

## Do-Not-Run List
Do not use this held-out oracle for model selection, training, or promotion.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: E006 held-out support-threshold oracle audit
