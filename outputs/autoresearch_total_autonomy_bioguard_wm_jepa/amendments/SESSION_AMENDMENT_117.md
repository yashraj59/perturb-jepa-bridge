# Session Amendment 117: Remote scGeneScope Metadata Discovery

## Trigger
`F087_SCGENESCOPE_ADAPTER_CONTRACT_READY_DATA_NOT_LOCAL`

## Evidence
F087 proved the local scGeneScope adapter contract but found no local dataset root. The next step is remote metadata discovery only, because scGeneScope is large and the loop must preserve low-compute and storage-safety constraints.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F088_SCGENESCOPE_REMOTE_DISCOVERY`

## Implementation Tasks
- Query Hugging Face dataset tree metadata for scGeneScope without downloading payload files.
- Identify small manifest, metadata, split, treatment, or sample files if exposed.
- Record visible RNA and imaging feature-file families and sizes.
- Determine whether a feature-level Tier 3 dry run is feasible under low-compute constraints.
- Do not train, do not download large payloads, and do not promote.

## Decision Use
If light metadata is found, build a manifest from it. If only large H5AD/image payloads are visible, search official supplementary/code metadata next or wait for a user-supplied manifest while continuing non-downloadable adapter work.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F088 remote scGeneScope metadata discovery
