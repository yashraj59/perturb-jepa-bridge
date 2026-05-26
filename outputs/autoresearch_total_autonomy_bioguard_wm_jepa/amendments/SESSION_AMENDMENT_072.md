# Session Amendment 072: Action-AdaLN Tiny-Cap Calibration

## Trigger
`F042_FRESH_SMALL_CAP_VALIDATION_PASS_DIAGNOSTIC`

## Evidence
F042 validated the tiny-cap residual rule on fresh synthetic seeds, but it was still an operator-style diagnostic. The next step must test whether the same rule is useful in the named BioGuard-WM Action-AdaLN + RoPE path.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Candidate Diagnostic
`F043_ACTION_ADALN_SMALL_CAP`

## Implementation Tasks
- Add a `small_cap_continuous` calibration mode to the Action-AdaLN + RoPE BioGuard-WM frozen-latent training path.
- Restrict candidate residual scales to `<= 0.05`.
- Select scale using train-only crossfit continuous transition, delta, and recall metrics.
- Score held-out rows only after train-only selection.
- Do not promote this result; a pass only permits designing stricter Tier 2/Tier 3 validation.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F043 Action-AdaLN tiny-cap calibration
