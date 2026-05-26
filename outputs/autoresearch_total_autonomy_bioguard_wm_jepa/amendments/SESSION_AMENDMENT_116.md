# Session Amendment 116: scGeneScope Metadata Adapter Preflight

## Trigger
`F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED`

## Evidence
F086 passed corrected synthetic current-registry gates but could not launch Tier 3 because no paired scRNA plus imaging validator is local. Public source pages identify scGeneScope as the best future condition-paired scRNA-seq plus Cell Painting validator candidate, but the dataset is large, so ingestion must start with metadata and split contracts.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F087_SCGENESCOPE_ADAPTER_PREFLIGHT`

## Implementation Tasks
- Add a metadata-only scGeneScope adapter contract.
- Require modality, treatment, dose, round, batch, replicate, split, and URI columns.
- Reject `condition_key`, `biological_key`, exact target-key, and target-key shortcut manifest columns.
- Build condition-pair tables from metadata only.
- Audit whether a local manifest exists; do not download large image/count payloads.
- Do not train and do not promote.

## Decision Use
If a local manifest validates, run a low-compute Tier 3 dry run on profile/embedding-level features. If no manifest is local, keep building adapter documentation and avoid full dataset download until metadata access is solved.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F087 scGeneScope metadata adapter preflight
