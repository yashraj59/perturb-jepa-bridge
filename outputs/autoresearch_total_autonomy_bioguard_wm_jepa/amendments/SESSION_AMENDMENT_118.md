# Session Amendment 118: Official scGeneScope Supplement Metadata Harvest

## Trigger
`F088_SCGENESCOPE_REMOTE_FEATURES_FOUND_BUT_TOO_LARGE_FOR_LOW_COMPUTE`

## Evidence
F088 showed the Hugging Face file tree is accessible but exposes no light manifest and the smallest visible paired RNA/image feature footprint is about 13.6 GB. The NeurIPS proceedings page exposes a small official supplemental archive, so the next low-compute step is to harvest metadata from that archive without downloading dataset payloads.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F089_SCGENESCOPE_SUPPLEMENT_HARVEST`

## Implementation Tasks
- Download only the small official supplemental archive under a strict byte cap.
- Extract Croissant metadata, code README, split contract, and file inventory.
- Update the adapter contract if metadata contradicts assumptions such as dose availability.
- Do not download H5AD/image payloads, do not train, and do not promote.

## Decision Use
If Croissant metadata provides split/field contracts, use it to validate the adapter and design a capped feature-level dry run. If metadata is insufficient, search the official code repo for manifest builders next.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F089 official scGeneScope supplement metadata harvest
