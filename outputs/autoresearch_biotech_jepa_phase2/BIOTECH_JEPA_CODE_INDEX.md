# BioTech-JEPA Code Index

This file records where the current BioTech-JEPA implementation lives. It is not a code dump; the actual executable code is in the repository files listed below.

## Core Architecture

- `perturb_jepa/models/biotech_jepa.py`
  - `BioTechJEPAConfig`
  - `BioTechJEPA`
  - Online/context RNA and image encoders
  - EMA target RNA and image encoders
  - Separate `z_bio` and `z_tech` branches
  - Query predictors for RNA program, image region, cross-modal RNA<->image, and action-conditioned transition JEPA
  - Batch heads for `z_tech` and probe-only heads for `z_bio`

## Losses And Diagnostics

- `perturb_jepa/training/biotech_losses.py`
  - `BioTechJEPALossWeights`
  - `biotech_jepa_loss`
  - `biotech_collapse_diagnostics`
  - Bio/tech orthogonality and VICReg-style anti-collapse components

## Training

- `perturb_jepa/training/biotech_trainer.py`
  - `BioTechJEPATrainer`
  - EMA teacher update after optimizer step

- `scripts/train_biotech_jepa.py`
  - Synthetic paired RNA+image training/evaluation
  - Norman RNA-only training/evaluation
  - Writes `config.json`, `metrics_train.jsonl`, `metrics_eval.json`, `jepa_identity_report.md`, `model_card.md`, and optional `checkpoint.pt`

## Evaluation

- `perturb_jepa/evaluation/biotech_metrics.py`
  - Identity checks
  - Cross-modal retrieval
  - Transition-to-target retrieval
  - `z_bio`/`z_tech` batch and perturbation probes
  - Source and target latent collapse diagnostics

- `scripts/evaluate_biotech_jepa.py`
  - Standalone checkpoint evaluator

## Data Loaders

- `perturb_jepa/training/bioaction_batches.py`
  - Reused synthetic condition-pair batch contract.

- `perturb_jepa/training/norman_biotech_batches.py`
  - Norman control-to-perturbed RNA-only condition pairs
  - Train-split-selected gene subset
  - Gene multi-hot action descriptors
  - Batch id fixed to zero and dose fixed to one for Norman, per user instruction

- `perturb_jepa/training/synthetic_biology_lite.py`
  - `synth_genetic_anchor_lite`
  - `synth_chemical_dose_anchor_lite`

## Tests

- `tests/test_biotech_jepa_model.py`
- `tests/test_norman_biotech_batches.py`
- `tests/test_biotech_batch_audit.py`
- `tests/test_synthetic_biology_lite.py::test_genetic_anchor_config_uses_fixed_dose_and_cross_batch_replicates`

## Experiment Outputs

- `outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0`
- `outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ002_norman_rna_only_seed0`
