# Session Amendment 054: F025 Program-Only Action Descriptor Audit

## Trigger
`F024_HELDOUT_ACTION_DESCRIPTOR_SUPPORT_LIMITS_ACTION_GENERALIZATION`

## Evidence
F024 showed held-out exact perturbation action dimensions are unsupported in train rows, while one program-level descriptor dimension remains supported. Correct action gives a small but inconsistent gain. Before implementing graph/program action JEPA, test whether supported non-exact action descriptors preserve the old floor.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F025_PROGRAM_ONLY_ACTION_DESCRIPTOR_AUDIT`

## Implementation Tasks
- Compare full exact+program action descriptors with source-only, supported-action-only, program-only, and supported-program-only descriptors.
- Fit all floors on train rows only.
- Use eval rows only for scoring.
- Do not use `condition_key`, `biological_key`, held-out target means, or exact target-key one-hot features as candidate model inputs.

## Decision Use
If program descriptors preserve the old floor, reopen non-exact biological action representation work. If they fail, pivot to benchmark/action-target redesign before more JEPA architecture.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F025 program-only action descriptor audit
