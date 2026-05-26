# Session Amendment 053: F024 Heldout Action Descriptor Support Audit

## Trigger
`F023_OLD_LATENT_ACTION_CONTRACT_AMBIGUOUS`

## Evidence
F023 found that source-only latents recover most of the old floor, while correct action still beats wrong action by a small amount. Action-only does not pass the active gates. The action descriptor support must be localized before training graph/program action priors.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F024_HELDOUT_ACTION_DESCRIPTOR_SUPPORT_AUDIT`

## Implementation Tasks
- For every held-out perturbation and seed, list active action descriptor dimensions.
- Record which active dimensions were seen in train rows.
- Score full vs source-only vs wrong-action gains by held-out perturbation.
- Do not use held-out target means, `condition_key`, or `biological_key` as model inputs.

## Decision Use
If held-out action dimensions are unsupported, pivot to non-exact-key biological action descriptors or benchmark redesign. If support is sufficient and action gains are consistent, reopen action-representation repair.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F024 heldout action descriptor support audit
