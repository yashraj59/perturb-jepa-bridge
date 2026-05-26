# Architectural Changes Log

No BioTech-JEPA architecture changes have been made at Phase 2 start.

Stage 1 is diagnostic-only. The only permitted code addition before reopening is the audit script and its focused test.

## Stage 1 Code Additions

- Added `scripts/run_biotech_batch_audit.py` for the diagnostic-only batch-disentanglement audit.
- Added `tests/test_biotech_batch_audit.py` for a smoke test that verifies required Stage 1 audit files are written.

No `BioTechJEPA` model, losses, trainer, or architecture-search scripts were implemented because the reopening gate failed.

## Genetic Amendment Code Changes

- Added `synth_genetic_anchor_lite` to `perturb_jepa/training/synthetic_biology_lite.py` for CRISPR-style fixed-dose genetic perturbation with held-out perturbations and cross-batch replicates.
- Added `synth_chemical_dose_anchor_lite` for future drug/concentration experiments where dose is meaningful.
- Updated `scripts/run_biotech_batch_audit.py` to report train cross-batch anchor fraction and eval cross-batch replicate fraction, and to allow a valid substitute teacher criterion for genetic held-out perturbation settings.
- Added `tests/test_synthetic_biology_lite.py::test_genetic_anchor_config_uses_fixed_dose_and_cross_batch_replicates`.

## BioTech-JEPA Tier 1 Code Additions

The user instructed that BioTech-JEPA should now be run for both synthetic and Norman.

Implemented files:

- `perturb_jepa/models/biotech_jepa.py`: `BioTechJEPAConfig` and `BioTechJEPA` with online/context encoders, EMA target encoders, detached teacher targets, target-query predictors, separate `z_bio`/`z_tech` branches, RNA program JEPA, image region JEPA, RNA<->image JEPA, and action-conditioned control-to-perturbed transition JEPA.
- `perturb_jepa/training/biotech_losses.py`: latent JEPA losses, `z_tech` batch CE when batch labels exist, bio/tech orthogonality, VICReg-style anti-collapse diagnostics, no raw reconstruction requirement.
- `perturb_jepa/training/biotech_trainer.py`: tiny trainer with EMA teacher updates.
- `perturb_jepa/evaluation/biotech_metrics.py`: identity checks, cross-modal retrieval, transition retrieval, batch/perturbation probes, source and target latent collapse diagnostics.
- `perturb_jepa/training/norman_biotech_batches.py`: Norman RNA-only condition-pair loader with train-selected gene subset, gene multi-hot action descriptors, constant guide dose, and batch id fixed to zero.
- `scripts/train_biotech_jepa.py`: train/evaluate entrypoint for synthetic and Norman.
- `scripts/evaluate_biotech_jepa.py`: standalone checkpoint evaluator.
- `tests/test_biotech_jepa_model.py`: model/loss smoke tests.
- `tests/test_norman_biotech_batches.py`: Norman loader contract test.

Important implementation constraints preserved:

- No `condition_key`, `biological_key`, or exact target-key one-hot is used as model input.
- PLS is not used in the representation path.
- Norman uses gene multi-hot action descriptors instead of exact condition-key one-hot lookup features.
- Norman batch and chemical dose are ignored for this specific processed h5ad per user instruction.
