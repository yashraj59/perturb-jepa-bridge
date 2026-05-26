# Architectural Changes Log

## Setup Changes

- Added `--eval-split` support to synthetic baseline/evaluation scripts so held-out perturbation and held-out dose baselines can be run without changing split semantics.
- No BioAction-JEPA model files have been promoted.

## BioAction-JEPA Implementation

- Added `perturb_jepa/training/bioaction_batches.py` for leakage-safe condition-pair batches. Target rows are drawn from the requested split; held-out perturbation controls fall back to train controls when necessary.
- Added `perturb_jepa/models/bioaction_jepa.py` with RNA/image online encoders, EMA target encoders, stop-gradient target states, query-based predictors, RNA/image/joint targets, and action-conditioned control-to-perturbed transitions.
- Added `perturb_jepa/training/bioaction_losses.py` with latent JEPA losses, prototype distribution loss, VICReg, Barlow redundancy reduction, and later a batch-centroid invariance penalty.
- Added `perturb_jepa/training/bioaction_trainer.py` and CLI scripts `scripts/train_bioaction_jepa.py` / `scripts/evaluate_bioaction_jepa.py`.
- Added `perturb_jepa/evaluation/bioaction_metrics.py` with identity, retrieval, transition-null, collapse, and batch-probe diagnostics.

## Architecture Search Changes

| Experiment | Change | Reason | Result |
| --- | --- | --- | --- |
| EXP001-EXP003 | Minimal real BioAction-JEPA, no reconstruction | Establish whether encoder-first JEPA can learn cross-modal and action-transition signals | Held-out perturbation transition/retrieval signal exists, but exact split and dose show batch leakage and weak RNA->image retrieval |
| EXP004 | Added batch-centroid invariance loss, weight `0.5` | Reduce measured joint/image batch decodability | Insufficient; joint batch probe remained high and retrieval did not recover |
| EXP005 | Increased batch-centroid invariance loss, weight `5.0` | Test whether stronger penalty fixes leakage | Failed; standalone joint batch probe remained `0.9412` vs majority `0.5313` |

## Identity And Non-Promotion

- PLS is not used as the BioAction main representation path.
- `condition_key` and `biological_key` remain labels only.
- Exact target-key one-hot features were not added.
- Batch labels are used only for diagnostics and the Family F invariance loss, not as encoder or predictor inputs.
- No Tier 1/2 candidate was promoted; protected PLS remains model of record.
