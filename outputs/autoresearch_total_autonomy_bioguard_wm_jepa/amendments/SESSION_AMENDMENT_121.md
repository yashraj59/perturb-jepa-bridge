# Session Amendment 121: scGeneScope Obs-Only Backed Dry Run

## Trigger
`F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN`

## Evidence
F091 found the smallest paired scGeneScope feature footprint is within the registered low-compute cap and the current workspace has enough disk/RAM with `anndata` and `h5py` available. The next step is to prove the actual files can be opened safely in backed mode before any model code touches them.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 supersedes it.

## New Diagnostic Branch
`F092_SCGENESCOPE_OBS_ONLY_DRY_RUN`

## Implementation Tasks
- Download only the F091-approved smallest RNA and image feature H5AD files if not already local.
- Enforce the F091 byte cap before attempting download.
- Open each H5AD with `backed="r"` and inspect obs metadata only.
- Verify required Croissant-derived obs columns are present in both modalities.
- Do not train, fit encoders, fit calibrators, compute target means, whiten, score JEPA metrics, or promote.

## Decision Use
If F092 passes, run a shape/split/pairing audit on backed feature metadata next. If download/access fails because of license or auth gating, write a hard escalation report. If obs contract fails, return to adapter repair or smaller manifest construction.

## Hard Escalation Check
No hard escalation trigger present before launch.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F092 scGeneScope obs-only backed dry run
