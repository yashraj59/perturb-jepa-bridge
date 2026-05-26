# Session Amendment 102: Image-Teacher RNA-Student JEPA Repair

## Trigger
`F072_IMAGE_LATENT_NONORACLE_REDUCES_MISMATCH`

## Evidence
F072 showed that non-oracle image PCA reduces calibration mismatch versus observed RNA PCA, but train-only linear RNA-to-image PCA prediction underfits and worsens delta optimism. The next smallest architecture step is a masked-RNA student/predictor trained against stop-gradient image-teacher targets.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F073 is Tier 1 diagnostic work and cannot promote.

## New Branch
`F073_IMAGE_TEACHER_RNA_STUDENT_JEPA`

## Implementation Tasks
- Fit image PCA on train rows only.
- Train a tiny masked-RNA student encoder plus predictor on train rows only.
- Use stop-gradient image PCA targets and JEPA-dominant cosine loss with a light MSE/variance anchor.
- Score the learned RNA-side predicted image latent under the same train-inner versus real-heldout transition/delta/recall optimism audit.
- Compare against observed RNA PCA and image PCA teacher ceiling.
- Do not use synthetic `z_bio`, `condition_key`, `biological_key`, exact heldout action one-hot, eval target means, or pooled train+test statistics.
- Do not promote.

## Decision Use
If F073 preserves image-teacher signal, reopen a small cross-modal BioGuard-WM-JEPA representation branch. If it underfits, run a failure localization audit before increasing model size.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F073 image-teacher RNA-student JEPA repair
