# Session Amendment 114: Current Image-Teacher Latent Floor Registry

## Trigger
`F084_LOCKED_RANK_FLOOR_EXCEEDS_IMAGE_TEACHER_CEILING`

## Evidence
F084 showed the locked delta-rank floor is above the current image-teacher target ceiling. This indicates a cross-representation baseline mismatch. The protocol allows a fresh baseline discrepancy to be documented rather than silently picking a convenient number.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F085_CURRENT_LATENT_FLOOR_REGISTRY`

## Implementation Tasks
- Rebuild the current train-only image-teacher latent tables for F082 seeds and split policies.
- Fit train-only action ridge floors in that current latent registry.
- Compute true delta effective-rank ceilings in the same registry.
- Compare F082 calibrated JEPA against those current-registry floors.
- Do not train and do not promote.

## Decision Use
If F085 shows the candidate preserves the current-registry floor and rank ceiling, design Tier 3 with an explicit representation-specific rank registry plus external-validator plan. If it shows a current-registry rank gap, implement a rank-preserving calibrator.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F085 current-latent floor registry
