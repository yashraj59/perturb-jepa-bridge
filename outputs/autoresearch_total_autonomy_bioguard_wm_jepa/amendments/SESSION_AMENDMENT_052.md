# Session Amendment 052: F023 Old Latent Action Contract Audit

## Trigger
`F022_SYNTHETIC_Z_BIO_CEILING_DOES_NOT_SUPPORT_TRANSITION_SEARCH`

## Evidence
F022 showed the old positive BioFlow/BioTech teacher latent floor is not reproduced by true synthetic `z_bio` or train-only RNA PCA representations. Before changing architecture, the old floor's action/source contract must be audited.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F023_OLD_LATENT_ACTION_CONTRACT_AUDIT`

## Implementation Tasks
- Reproduce the old full action-ridge floor across seeds 0-4.
- Compare source-only ridge, action-only ridge, mean delta, wrong eval action, permuted train action, and metadata group-mean audit references.
- Treat metadata group-mean rows as diagnostic references only, never as candidate model inputs.
- Do not use eval target rows for fitting.

## Decision Use
If wrong/permuted action performs like the full floor, cool residual/operator work and pivot to target/benchmark redesign. If action ablation materially hurts, reopen action representation work.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F023 old latent action/source contract audit
