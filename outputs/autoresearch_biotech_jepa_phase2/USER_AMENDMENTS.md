# User Amendments

## Genetic Perturbation Dose Correction

The user clarified that many single-cell perturbation datasets, including CRISPR-style genetic perturbation datasets, do not have meaningful chemical dose. Dose should not be treated as a universal axis.

Applied change:

- Use a genetic perturbation synthetic mode with fixed guide dose for CRISPR-like experiments.
- Treat chemical dose as relevant only for chemical/drug perturbation experiments.
- Do not require held-out dose evaluation for Norman-style genetic perturbation experiments.

## Norman-Specific Metadata Correction

The user clarified that if the processed Norman h5ad does not expose batch metadata, batch should be ignored for the Norman-specific experiment.

Applied change:

- Inspected `data/raw/gears_norman/norman/perturb_processed.h5ad`.
- Found obs columns: `condition`, `cell_type`, `dose_val`, `control`, `condition_name`.
- No batch/plate/lane/donor column is exposed in this processed file.
- `dose_val` has values `1` and `1+1`, consistent with guide count / single-vs-combinatorial perturbation notation, not chemical concentration.
- For Norman-specific experiments, ignore batch and chemical dose unless another metadata source exposes them.

## Current Status

The synthetic genetic-anchor audit now supports reopening architecture search for a genetic perturbation BioTech-JEPA test. The Norman file can be used as a biological perturbation reference, but not as a batch-disentanglement dataset unless batch metadata is recovered elsewhere.

## Implementation/Run Instruction

The user then instructed: run the next BioTech-JEPA step for both synthetic and Norman, and document the instruction history.

Applied change:

- Implemented a minimal real BioTech-JEPA path after the genetic-anchor reopening gate passed.
- Ran `synth_genetic_anchor_lite/test_heldout_perturbation` as the synthetic Tier 1 diagnostic.
- Ran Norman `perturb_processed.h5ad` as an RNA-only genetic perturbation diagnostic.
- For Norman, preserved the user correction: ignore batch and chemical dose for this specific experiment because the processed h5ad exposes no batch metadata and `dose_val` is guide-count notation.
- Kept the protected rank-3 train-split-only PLS raw-linear readout as the model of record.
