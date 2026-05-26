# Session Amendment 103: Image-Teacher Failure Localization

## Trigger
`F073_RNA_STUDENT_IMAGE_TEACHER_WEAK_OR_UNSTABLE`

## Evidence
F073 trained a tiny masked-RNA student against stop-gradient train-only image PCA targets. It achieved nontrivial target alignment but failed to preserve the image-teacher transition floor and recall. Before increasing capacity, the loop must localize whether the failure is image-target underfit or loss of perturbation structure.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F074 is diagnostic and cannot promote.

## New Diagnostic Branch
`F074_IMAGE_TEACHER_FAILURE_LOCALIZATION`

## Implementation Tasks
- Read F073 seed metrics and training trace.
- Compare candidate versus image teacher and observed RNA by policy and seed.
- Quantify image-target cosine, train prediction-target cosine, recall gap, delta cosine gap, transition gap, perturbation-probe gap, and batch-probe gap.
- Do not train, do not select a model, and do not promote.

## Decision Use
If target underfit dominates, consider a modest capacity/optimization repair. If perturbation structure is lost despite target alignment, add condition/program-preserving and multi-positive image-teacher objectives before any full JEPA wrapper.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F074 image-teacher failure localization
