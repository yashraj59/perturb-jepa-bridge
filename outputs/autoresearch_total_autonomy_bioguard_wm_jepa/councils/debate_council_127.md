# Debate Council 127

## Trigger
`F102_STRICT_SCRNA_IMAGING_CANDIDATE_FOUND_RNA_OBS_PREFLIGHT_PENDING`

## Evidence Summary
F102 found PerturbMulti as an actionable public candidate for strict paired
single-cell RNA plus imaging validation. The Hugging Face repository is public,
not gated, CC-BY-4.0, and includes RNA H5AD, protein-intensity H5AD, spatial
coordinates, perturbation metadata, and image tar archives keyed by cell ID.

Only the small protein-intensity H5AD was downloaded and opened in backed mode.
It has 99,294 observations, 18 imaging/protein/RNA-image channels, 508
perturbations, and required cell-ID/perturbation/spatial metadata columns.

## Independent Proposals
- Methodologist: proceed to RNA obs-only and image-key pairing preflight; do not
  train yet.
- Architect: no model redesign is justified; if pairing passes, reuse the frozen
  F082/F096 ProgramBootstrapJEPA path.
- Skeptic: the large RNA H5AD and image archives create storage/RAM risk; use
  HDF5/backed metadata checks and tar-member sampling only.
- Monitor: no promotion; raw data remains outside git.

## Decision
`COUNCIL_EXECUTE_PERTURBMULTI_RNA_OBS_PAIRING_PREFLIGHT`

## Exact Next Amendment
Execute `F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT`.
