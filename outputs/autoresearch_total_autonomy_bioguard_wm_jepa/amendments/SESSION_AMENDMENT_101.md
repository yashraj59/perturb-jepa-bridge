# Session Amendment 101: Non-Oracle Image Latent Ceiling Audit

## Trigger
`F071_CROSS_MODAL_ZBIO_RECOVERY_REDUCES_MISMATCH`

## Evidence
F071 showed that image observations can recover synthetic `z_bio` nearly perfectly using train-only oracle supervision, while RNA-only recovery is much weaker. The next step must remove the oracle target and test whether image-derived latents themselves carry the transition geometry.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F072 is diagnostic and cannot promote.

## New Diagnostic Branch
`F072_IMAGE_LATENT_CEILING_AUDIT`

## Implementation Tasks
- Reuse the F068-F071 split-contract benchmarks and seeds.
- Build train-only image PCA, RNA+image concat PCA, RNA-to-image PCA ridge, and image-to-RNA PCA ridge latents.
- Rerun train-inner versus real-heldout transition/delta/recall optimism.
- Report image latent recovery cosine for cross-modal ridge branches.
- Do not use synthetic `z_bio` targets in non-oracle branches.
- Do not train JEPA and do not promote.

## Decision Use
If non-oracle image latents reduce mismatch, reopen a small cross-modal JEPA representation repair branch centered on image-teacher/RNA-student alignment. If not, the useful image signal requires more structured objectives than PCA/ridge, and the next branch should implement a tiny image-teacher JEPA with strict floor checks.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F072 non-oracle image latent ceiling audit
