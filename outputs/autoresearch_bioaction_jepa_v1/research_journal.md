# Research Journal

## Setup

**Protocol**: `prompt/codex_novel_bioaction_jepa_autoresearch_prompt.md`

**Active model of record**: protected rank-3 train-split-only PLS raw-linear readout. Family N and Family O remain audit references. No BioAction-JEPA candidate may supersede these without a Tier 3 pass.

**Initial action**: patched synthetic baseline scripts to support `--eval-split` before Step 0 held-out baselines.

## Step 0 Baselines

Step 0 completed before BioAction architecture search. Verification artifacts are under `outputs/autoresearch_bioaction_jepa_v1/step0_baselines/`.

- `synth_micro/test`: exact train-key coverage is `1.0`; matching baselines are strong and not sufficient for novelty claims.
- `synth_heldout_perturbation_lite/test_heldout_perturbation`: exact train-key coverage is `0.0`; matched perturbed mean gives program recovery `0.1168`, pseudobulk correlation `0.9613`, protected PLS RNA->image recall@1 `0.1852`.
- `synth_dose_extrapolation_lite/test_heldout_dose`: exact train-key coverage is `0.0`; matched perturbed mean gives program recovery `0.0892`, pseudobulk correlation `0.9946`, protected PLS RNA->image recall@1 `0.1806`.

The protected model of record remains the rank-3 train-split-only PLS raw-linear readout. Family N/O condition-mean and count-aware references remain audit baselines only.

## BioAction Implementation

Implemented the smallest real BioAction-JEPA path:

- condition-pair batch loader: `perturb_jepa/training/bioaction_batches.py`
- model/config/query predictors: `perturb_jepa/models/bioaction_jepa.py`
- latent JEPA objectives, VICReg/Barlow, distributional prototype loss, and batch-invariance probe loss: `perturb_jepa/training/bioaction_losses.py`
- trainer: `perturb_jepa/training/bioaction_trainer.py`
- evaluator with retrieval, transition, collapse, identity, and batch probe diagnostics: `perturb_jepa/evaluation/bioaction_metrics.py`
- CLIs: `scripts/train_bioaction_jepa.py`, `scripts/evaluate_bioaction_jepa.py`
- tests: `tests/test_bioaction_jepa_model.py`, `tests/test_bioaction_jepa_losses.py`, `tests/test_bioaction_condition_pairs.py`, `tests/test_bioaction_eval_split.py`

Identity checks pass: online/context encoders, EMA target encoders, stop-gradient latent targets, query predictors, latent losses with reconstruction weight zero, RNA<->image prediction, and action-conditioned control-to-perturbed transition prediction are active. PLS/raw-linear heads are not used in the BioAction main path.

## Tier 1 Runs

EXP001 on `synth_heldout_perturbation_lite/test_heldout_perturbation` is the best clean signal:

- exact match fraction `0.0`
- RNA->image recall@1 `0.0625` vs zero-step initial encoder `0.0000`
- image->RNA recall@1 `0.09375` vs initial `0.03125`
- transition-source cosine improvement `+0.00968`
- joint batch probe balanced accuracy `0.4787` vs majority `0.4063`

EXP002 on `synth_dose_extrapolation_lite/test_heldout_dose` learned transition but showed leakage:

- exact match fraction `0.0`
- transition-source cosine improvement `+0.02624`
- RNA->image recall@1 `0.03125` vs initial `0.0625`
- image batch probe balanced accuracy `0.6524`; joint batch probe `0.6316` vs majority `0.5000`

EXP003 on `synth_micro/test` failed the Tier 1 gate:

- exact match fraction `1.0`
- transition-source cosine improvement `+0.04168`
- RNA->image recall@1 `0.0000` vs initial `0.03125`
- joint batch probe balanced accuracy `0.9412` vs majority `0.5313`

Family F batch-invariance probes attempted to directly reduce the measured leakage:

- EXP004 with batch-invariance weight `0.5` reduced train-eval joint batch probe to `0.8157` but did not recover retrieval.
- EXP005 with batch-invariance weight `5.0` standalone eval still had joint batch probe `0.9412` vs majority `0.5313` and RNA->image recall@1 `0.0000`.

## Stop Decision

Stop condition fired: batch leakage dominates latent state across two candidate families, Family A and Family F. No Tier 2 or Tier 3 runs are allowed from this state. The protected rank-3 PLS model remains the model of record.

Final report written to `outputs/autoresearch_bioaction_jepa_v1/final_report.md`.
