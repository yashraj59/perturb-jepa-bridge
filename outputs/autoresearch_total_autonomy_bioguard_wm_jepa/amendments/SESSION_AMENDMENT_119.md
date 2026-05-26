# Session Amendment 119: scGeneScope Croissant Contract Validation

## Trigger
`F089_SCGENESCOPE_SUPPLEMENT_METADATA_RECOVERED_ADAPTER_UPDATED`

## Evidence
F089 recovered Croissant metadata and the official replicate-based split contract. Before any feature download, the adapter must prove it can map those fields into the project's no-leakage condition-paired manifest contract.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F090_SCGENESCOPE_CROISSANT_CONTRACT`

## Implementation Tasks
- Validate required Croissant fields: `cell_id`, `Treatment`, `Replicate`, `batch`, and `Group`.
- Validate replicate split mapping: `3=train`, `5=validation`, `4=test`, `1/2=alternate_test`.
- Build a dry-run paired RNA/image manifest with fixed dose and no payload access.
- Build condition-pair tables for train, validation, test, and alternate-test splits.
- Do not download H5AD/image payloads, do not train, and do not promote.

## Decision Use
If F090 passes, design a storage-gated feature-level preflight. If it fails, fix the adapter contract before any feature-level dry run.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F090 scGeneScope Croissant contract validation
