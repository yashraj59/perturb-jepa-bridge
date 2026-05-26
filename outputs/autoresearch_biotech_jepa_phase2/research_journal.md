# Phase 2 Research Journal

## Setup

**Protocol**: `prompt/biotech_jepa_phase2_amendment_prompt.md`

**Research question**: Can a real cross-modal, action-conditioned JEPA learn a biological latent `z_bio` that predicts perturbation transitions while explicitly separating technical batch/acquisition state into `z_tech`?

**Active model of record**: protected rank-3 train-split-only PLS raw-linear readout. No BioAction candidate is promoted. Family N and Family O remain audit references.

**Phase 1 facts preserved**:

- BioAction-JEPA identity was implemented and passed.
- No Tier 2 or Tier 3 BioAction run was launched.
- Prior stop condition: batch leakage dominated latent state across Family A and Family F.
- Family F batch-centroid invariance weights `0.5` and `5.0` did not fix the failure mode.

## Stage 1 Plan

Run the diagnostic-only batch-disentanglement audit before changing model architecture. Architecture search may reopen only if `REOPENING_DECISION.md` satisfies every reopening criterion from the amendment.

## Stage 1 Audit Result

Command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/run_biotech_batch_audit.py --datasets synth_micro synth_heldout_perturbation_lite synth_dose_extrapolation_lite synth_batch_confound_lite --eval-splits test test_heldout_perturbation test_heldout_dose test --seeds 0 1 2 --device cpu --output-root outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit
```

Decision label: `PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN`.

Key findings:

- Biological signal exists: split-half RNA->image same-biological recall@1 reaches `0.5625` on `synth_micro/test`, and is nonzero on held-out splits.
- Batch signal is strong in raw inputs: raw image pooled pixels predict batch nearly perfectly across datasets; raw RNA pseudobulk also carries batch above majority.
- Protected PLS reduces some batch signal, but does not solve the Phase 2 teacher-target problem.
- Phase 1/zero-step BioAction latents contain strong batch signal, especially on held-out dose and batch-confounded splits.
- The blocking failure is anchoring, not lack of a candidate mechanism: held-out perturbation and held-out dose eval targets have `0.0` cross-batch train-anchor fraction and no valid substitute for constructing consensus/environment-swap biological teacher targets without leakage.

Stop decision:

At that point architecture search remained closed. `final_report.md` was completed and no BioTech-JEPA model code had been implemented under the original dose-inclusive framing.

## User Amendment: Genetic Perturbation Mode

The user clarified that dose is not a universal single-cell perturbation axis. For Norman/CRISPR-style perturbation, dose should be ignored or treated as fixed guide dosage, not chemical concentration.

The user also clarified that if the Norman processed h5ad does not expose batch metadata, batch should be ignored for the Norman-specific experiment.

Actions taken:

- Added `synth_genetic_anchor_lite`, a fixed-dose genetic perturbation synthetic mode with held-out genetic perturbations and cross-batch replicates.
- Added `synth_chemical_dose_anchor_lite` for future chemical/drug dose experiments where concentration is meaningful.
- Updated the Stage 1 audit reopening logic to allow a valid substitute when train biological keys have cross-batch anchors and held-out eval biological keys have cross-batch replicates, even if exact held-out perturbation keys are absent from train.
- Inspected Norman `perturb_processed.h5ad` and wrote `NORMAN_CONTEXT_AUDIT.md`.

## Genetic Anchor Audit Result

Command:

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/run_biotech_batch_audit.py --datasets synth_genetic_anchor_lite synth_batch_confound_lite --eval-splits test_heldout_perturbation test --seeds 0 1 2 --device cpu --output-root outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit_genetic_anchor
```

Decision label: `PHASE2_AUDIT_COMPLETE_REOPEN`.

Key values:

- Minimum exact held-out cross-batch train-anchor fraction: `0.0`
- Minimum train biological-key cross-batch anchor fraction: `1.0`
- Minimum held-out eval biological-key cross-batch replicate fraction: `1.0`
- Valid substitute for cross-batch teacher: `true`
- Max split-half RNA->image same-bio recall@1: `0.6389`
- Max raw/protected batch-probe excess over majority: `0.6667`

Interpretation:

The original audit failed because it treated dose/held-out exact biological keys as required for all perturbation styles. Under the corrected genetic perturbation framing, the synthetic benchmark has enough cross-batch structure to test a BioTech-JEPA factorization. Norman remains useful as a genetic perturbation reference but not as a batch-disentanglement validator unless batch metadata is recovered elsewhere.

## User Direction: Run BioTech-JEPA

The user instructed that the next step should be run for both synthetic and Norman and that the instruction history should be documented.

Implemented a minimal real BioTech-JEPA path:

- online/context RNA and image encoders;
- EMA target RNA and image encoders;
- stop-gradient teacher latents;
- query-based target predictors;
- separate `z_bio` and `z_tech` branches;
- RNA program JEPA;
- image region JEPA;
- RNA->image and image->RNA JEPA;
- action-conditioned control-to-perturbed transition JEPA;
- anti-collapse and bio/tech orthogonality diagnostics;
- Norman RNA-only loader using gene multi-hot action descriptors, fixed guide dose, and no batch feature.

Focused tests:

```text
pytest tests/test_biotech_jepa_model.py tests/test_norman_biotech_batches.py tests/test_synthetic_biology_lite.py::test_genetic_anchor_config_uses_fixed_dose_and_cross_batch_replicates tests/test_biotech_batch_audit.py
```

Result: `5 passed`.

Post-evaluator patch focused tests:

```text
pytest tests/test_biotech_jepa_model.py tests/test_norman_biotech_batches.py
```

Result: `3 passed`.

## BTJ001 Synthetic Genetic Anchor

Command:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/train_biotech_jepa.py --dataset synth_genetic_anchor_lite --eval-split test_heldout_perturbation --seed 0 --device cuda --steps 20 --eval-steps 4 --batch-size 2 --bag-size 3 --shared-dim 32 --bio-dim 24 --tech-dim 8 --predictor-dim 64 --num-condition-prototypes 4 --output-dir outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0 --save-checkpoint
```

Result summary:

- `encoder_path_used=1.0`
- `pls_raw_linear_main_path_used=0.0`
- `condition_key_feature_present=0.0`
- `separate_bio_and_tech_latents_present=1.0`
- RNA->image recall@1: `0.1875`
- Image->RNA recall@1: `0.0000`
- Transition-to-target recall@1: `0.4375`
- Transition source cosine improvement: `+0.0161`
- Joint `z_bio` batch-probe accuracy: `0.1875`
- Joint `z_tech` batch-probe accuracy: `0.4375`
- Batch allocation gap (`z_tech - z_bio`): `+0.2500`
- Joint `z_bio` effective rank: `7.5103`

Decision: `TIER1_DIAGNOSTIC_NO_PROMOTION`. The factorization mechanism is visible, but the transition gain is small and this is a single-seed, tiny low-compute run. Protected PLS remains model of record.

## BTJ002 Norman RNA-Only

Command:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python scripts/train_biotech_jepa.py --dataset norman --norman-h5ad data/raw/gears_norman/norman/perturb_processed.h5ad --eval-split test --seed 0 --split-seed 1 --device cuda --steps 10 --eval-steps 4 --batch-size 4 --shared-dim 32 --bio-dim 24 --tech-dim 8 --predictor-dim 64 --gene-count 256 --rna-only --output-dir outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ002_norman_rna_only_seed0 --save-checkpoint
```

Result summary:

- `encoder_path_used=1.0`
- `pls_raw_linear_main_path_used=0.0`
- `condition_key_feature_present=0.0`
- `separate_bio_and_tech_latents_present=1.0`
- RNA-only diagnostic: `1.0`
- Transition-to-target recall@1: `0.0625`
- Transition-to-target median rank: `8.5`
- Transition source cosine improvement: `+0.0313`
- Target `z_bio` effective rank: `7.4066`
- Target `z_bio` std mean: `0.0183`
- Target `z_tech` std mean: `0.1228`
- Batch probe is unavailable: Norman processed h5ad has only one synthetic batch label after user-directed batch ignore.

Interpretation:

Norman confirms the code path can run on the real processed h5ad without using real test rows for training. It is not a promotion candidate because it is RNA-only, has no exposed batch metadata, and cannot test cross-modal image prediction or batch disentanglement.
