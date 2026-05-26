# Session Amendment 120: scGeneScope Feature Storage And Backed-IO Preflight

## Trigger
`F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE`

## Evidence
F090 validated the scGeneScope Croissant field contract and replicate split mapping without touching payload files. The remaining bottleneck is whether the smallest paired feature files can be inspected safely under low-compute and storage/RAM constraints.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F091_SCGENESCOPE_FEATURE_PREFLIGHT`

## Implementation Tasks
- Read the saved F088 Hugging Face tree summary and F090 contract metrics.
- Pair RNA and image feature H5AD candidates by round.
- Estimate smallest paired feature footprint, disk buffer, backed-IO RAM requirement, and dependency readiness.
- Do not download H5AD/image payloads, do not train, and do not promote.

## Decision Use
If all gates pass, the next permissible step is an obs-only/backed H5AD open dry run with strict byte accounting. If any gate fails, require a smaller manifest-backed feature subset or continue metadata-only validation work.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F091 scGeneScope feature storage/backed-IO preflight
