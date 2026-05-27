# Session Amendment 127: PerturbMulti RNA Obs And Pairing Preflight

## Trigger
`F102_STRICT_SCRNA_IMAGING_CANDIDATE_FOUND_RNA_OBS_PREFLIGHT_PENDING`

## Evidence
PerturbMulti is the first public strict paired candidate found after stopping
the Rosetta loop. Its manifest includes CRISPR-screen RNA H5AD, protein-intensity
H5AD, perturbation metadata, spatial coordinates, and cell-ID-keyed image tar
archives. A backed obs/schema probe of the small protein H5AD passed.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless a later real
fresh Tier 3 pass explicitly supersedes it.

## New Branch
`F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT`

## Implementation Tasks
- Do not train or promote.
- Download only the CRISPR RNA H5AD to ignored storage if storage permits.
- Inspect RNA obs with backed or HDF5-level access; do not load `.X`.
- Confirm RNA obs columns for cell ID, perturbation, coordinates, and batch/split
  design.
- Confirm metadata-only overlap among RNA cell IDs, protein H5AD cell IDs, and
  image tar member IDs.
- Only after pairing passes, design a small sealed split and run the frozen
  F082/F096 ProgramBootstrapJEPA path on GPU.

## Hard Escalation Check
No hard escalation trigger present, but available RAM was low during F102.

## Continuation Rule
If RNA obs or image-key pairing fails, document validation blocked rather than
promoting from Rosetta or F096.
