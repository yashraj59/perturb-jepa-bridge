# BioAction-JEPA Final Report

## Stop Status

Autonomous loop stopped. Stop condition fired: batch leakage dominated the latent state across two candidate families, Family A minimal BioAction-JEPA and Family F batch-invariant latent JEPA.

No Tier 2 or Tier 3 runs were launched. No BioAction candidate supersedes the protected model of record.

## Model Of Record

The protected model of record remains the rank-3 train-split-only PLS raw-linear readout. Family N and Family O remain audit references only.

## What Was Built

BioAction-JEPA was implemented as a real JEPA path:

- online/context RNA and image encoders
- EMA target RNA and image encoders
- stop-gradient latent teacher targets
- target-query predictors
- RNA program JEPA and image region JEPA
- RNA->image and image->RNA latent prediction
- joint->RNA/image latent prediction
- control state + perturbation action -> perturbed teacher-state transition prediction
- VICReg/Barlow/prototype anti-collapse objectives
- identity, retrieval, transition-null, collapse, and batch-leakage diagnostics

Primary code files:

- `perturb_jepa/training/bioaction_batches.py`
- `perturb_jepa/models/bioaction_jepa.py`
- `perturb_jepa/training/bioaction_losses.py`
- `perturb_jepa/training/bioaction_trainer.py`
- `perturb_jepa/evaluation/bioaction_metrics.py`
- `scripts/train_bioaction_jepa.py`
- `scripts/evaluate_bioaction_jepa.py`

## Step 0

Step 0 baselines completed before architecture search. Summary is in `outputs/autoresearch_bioaction_jepa_v1/step0_baselines/SUMMARY.md`.

Key checks:

- `synth_micro/test` exact match fraction `1.0`; protected PLS RNA->image recall@1 `0.28125`.
- `synth_heldout_perturbation_lite/test_heldout_perturbation` exact match fraction `0.0`; protected PLS RNA->image recall@1 `0.1852`.
- `synth_dose_extrapolation_lite/test_heldout_dose` exact match fraction `0.0`; protected PLS RNA->image recall@1 `0.1806`.

## Tier 1 Results

EXP001, Family A, held-out perturbation:

- real-JEPA identity: pass
- exact match fraction: `0.0`
- RNA->image recall@1: `0.0625` vs zero-step initial `0.0000`
- image->RNA recall@1: `0.09375` vs zero-step initial `0.03125`
- transition-source cosine improvement: `+0.00968`
- joint batch probe balanced accuracy: `0.4787` vs majority `0.4063`

This is the best useful signal, but it is only Tier 1 and cannot promote.

EXP002, Family A, held-out dose:

- transition-source cosine improvement: `+0.02624`
- RNA->image recall@1: `0.03125` vs initial `0.0625`
- image batch probe balanced accuracy: `0.6524`
- joint batch probe balanced accuracy: `0.6316` vs majority `0.5000`

Transition learning exists, but batch leakage and retrieval weakness block Tier 2.

EXP003, Family A, exact `synth_micro/test`:

- exact match fraction: `1.0`
- transition-source cosine improvement: `+0.04168`
- RNA->image recall@1: `0.0000` vs initial `0.03125`
- joint batch probe balanced accuracy: `0.9412` vs majority `0.5313`

This fails Tier 1 due severe batch leakage and RNA->image retrieval regression.

EXP004/EXP005, Family F batch-invariance:

- batch-centroid invariance weights tested: `0.5` and `5.0`
- EXP005 standalone joint batch probe remained `0.9412` vs majority `0.5313`
- EXP005 RNA->image recall@1 remained `0.0000`

The mitigation did not fix the failure mode.

## Tests

Focused BioAction test subset passes:

```text
pytest tests/test_bioaction_jepa_model.py tests/test_bioaction_jepa_losses.py tests/test_bioaction_condition_pairs.py tests/test_bioaction_eval_split.py
16 passed, 5 warnings
```

Warnings are PyTorch transformer nested-tensor warnings.

## Why The Loop Stopped

The model is a real JEPA by identity checks, and it learns some action-conditioned transition signal. The blocker is that shared/joint latents remain highly predictive of technical batch on exact and dose settings. A direct batch-invariance penalty did not remove this. Continuing to Tier 2 would spend compute on a known invalid Tier 1 failure mode.

The next research step should not be another longer run of the same architecture. It should be a data/evaluation and architecture investigation focused on disentangling biological condition signal from technical batch before reopening Tier 1.
