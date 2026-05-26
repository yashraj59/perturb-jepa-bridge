# Session Amendment 084: Row-Wise Residual Abstention

## Trigger
`F054_DESCRIPTOR_MARGIN_RERUN_ZERO_FALLBACK`

## Evidence
F054 selected zero residual because every nonzero global scale caused train near-tie margin erosion. The residual direction still improved train transition and delta metrics, so the next mechanism should be local abstention rather than another scalar scale.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it. F055 is diagnostic validation only and cannot promote.

## New Diagnostic Branch
`F055_ROWWISE_ABSTENTION`

## Implementation Tasks
- Use fresh seeds 16, 17, and 18.
- Use `synth_program_aligned_genetic_lite` with non-exact program action descriptors only.
- Train the same low-compute real ProgramBootstrapJEPA path.
- Fit ridge floor and residual head on train rows only.
- Build train-only unsafe-margin labels from train retrieval/margin diagnostics.
- Use only inference-available source/action/floor/head-derived features for row-wise abstention.
- Score held-out rows only after the train-only rule is fixed.
- Do not promote.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F055 row-wise residual abstention
