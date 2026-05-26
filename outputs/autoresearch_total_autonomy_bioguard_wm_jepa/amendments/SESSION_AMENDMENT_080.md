# Session Amendment 080: Descriptor-Aligned Action Contract Replication

## Trigger
`F050_PROTECTED_FLOOR_IS_COVARIATE_ADJUSTED_SOURCE_DOMINATED`

## Evidence
F050 showed that the active protected full-ridge floor is source/covariate dominated and is a poor identity target for a biological action-transition JEPA. Earlier F026 approved the descriptor-aligned synthetic benchmark, and F042 showed fresh-seed nonzero residual safety on that benchmark. The next step is a low-compute replication before designing another full JEPA phase.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. No Phase 8/F051 diagnostic can promote.

## New Diagnostic Branch
`F051_DESCRIPTOR_ALIGNED_REPLICATION`

## Implementation Tasks
- Use `synth_program_aligned_genetic_lite`.
- Use non-exact program action descriptors only.
- Train the existing small real JEPA path with RNA online/context encoder, image online/context encoder, EMA target encoders, stop-gradient latent targets, query predictors, cross-modal RNA/image losses, and action-conditioned transition JEPA.
- Fit ridge floor and residual head on train rows only.
- Select residual scale using the F042 train-only tiny-cap rule.
- Score held-out rows only after train-only selection.

## Decision Use
If F051 replicates safe nonzero behavior, design the next full descriptor-aligned BioGuard-WM/JEPAction branch. If it fails, close residual/operator heads and pivot to representation or action-descriptor redesign.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F051 descriptor-aligned action-contract replication
