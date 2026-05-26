# BioTech-JEPA Phase 2 Updated Report

## Stop Status

Initial Stage 1 audit completed and did not reopen architecture search under the original dose-inclusive framing.

Decision label: `PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN`.

Closure label: `SEARCH_CLOSED_NO_NEW_BASELINE`.

This decision is superseded for the genetic perturbation synthetic setting by a user amendment and rerun described below. No model has been promoted.

## Model Of Record

The protected model of record remains the rank-3 train-split-only PLS raw-linear readout. No BioAction or BioTech candidate is promoted. PLS remains restricted to protected baseline or short annealed bootstrap-teacher use only.

## What Was Run

The diagnostic-only audit was run on CPU with thread caps:

```text
python scripts/run_biotech_batch_audit.py --datasets synth_micro synth_heldout_perturbation_lite synth_dose_extrapolation_lite synth_batch_confound_lite --eval-splits test test_heldout_perturbation test_heldout_dose test --seeds 0 1 2 --device cpu --output-root outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit
```

Required Stage 1 artifacts were written under `outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit/`.

## Reopening Criteria

| Criterion | Result |
| --- | --- |
| measurable biological signal not fully explained by batch | pass |
| enough cross-batch biological anchors or valid substitute | fail |
| identified source of batch leakage | pass |
| targeted mechanism beyond increasing invariance weight | pass |
| updated gates with exact baselines | pass |

The failing criterion is decisive: held-out perturbation and held-out dose eval targets have `0.0` cross-batch train-anchor fraction, and the audit found no valid substitute for constructing cross-batch consensus or environment-swap biological teacher targets without leakage.

## Key Findings

- Max split-half RNA->image same-biological recall@1: `0.5625`. There is real biological/cross-modal signal in the synthetic benchmark.
- Max raw/protected batch-probe excess over majority: `0.6667`. Raw image and RNA inputs carry strong technical batch signal.
- Max Phase 1/zero-step representation batch-probe excess over majority: `0.6667`. Batch signal appears before and after Phase 1 training, especially on held-out dose.
- `synth_heldout_perturbation_lite/test_heldout_perturbation`: cross-batch train-anchor fraction `0.0`; no biological train anchor for all eval targets.
- `synth_dose_extrapolation_lite/test_heldout_dose`: cross-batch train-anchor fraction `0.0`; no biological train anchor for all eval targets.
- `synth_batch_confound_lite/test`: eval targets have only same-batch anchors, so it is useful as a confounding audit but not as a clean teacher-construction basis.

## Decision

Do not implement BioTech-JEPA yet. The required teacher-target mechanism cannot be tested cleanly on the current held-out splits. Reopening architecture search would risk training another batch-contaminated JEPA and violating the amendment's diagnostic-first gate.

Recommended next move is dataset/split redesign: create held-out perturbation and dose splits with cross-batch biological anchors, or define a validated substitute teacher that does not use test target means or exact biological-key lookup.

## User Amendment And Norman Check

The user clarified:

- Many single-cell genetic perturbation datasets do not have meaningful chemical dose.
- Only chemical perturbation/drug screens should require dose as a concentration axis.
- For the Norman-specific experiment, if the processed h5ad has no batch metadata, ignore batch.
- For Norman, ignore chemical dose as well.

Norman file inspected:

```text
data/raw/gears_norman/norman/perturb_processed.h5ad
```

Findings:

- Shape: `91,205` cells by `5,045` genes.
- `obs` columns: `condition`, `cell_type`, `dose_val`, `control`, `condition_name`.
- Conditions: `284`.
- Cell type: `A549` only.
- No batch/acquisition column is exposed.
- `dose_val` values are `1` and `1+1`; this is guide-count / perturbation-composition notation, not a chemical concentration series.

Norman-specific consequence:

```text
action = genetic perturbation or gene-pair perturbation
context = A549
chemical dose = ignored
batch = ignored unless metadata is recovered elsewhere
```

## Genetic Synthetic Rerun

Added synthetic modes:

- `synth_genetic_anchor_lite`: fixed guide dose, held-out genetic perturbations, cross-batch replicates.
- `synth_chemical_dose_anchor_lite`: chemical-dose style synthetic mode for future drug/concentration experiments.

Reran Stage 1 genetic audit:

```text
python scripts/run_biotech_batch_audit.py --datasets synth_genetic_anchor_lite synth_batch_confound_lite --eval-splits test_heldout_perturbation test --seeds 0 1 2 --device cpu --output-root outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit_genetic_anchor
```

Decision label: `PHASE2_AUDIT_COMPLETE_REOPEN`.

Key values:

- Minimum exact held-out cross-batch train-anchor fraction: `0.0000`
- Minimum train biological-key cross-batch anchor fraction: `1.0000`
- Minimum held-out eval biological-key cross-batch replicate fraction: `1.0000`
- Valid substitute for cross-batch teacher: `true`
- Max split-half RNA->image same-bio recall@1: `0.6389`
- Max raw/protected batch-probe excess over majority: `0.6667`
- Max Phase 1/zero-step representation batch-probe excess over majority: `0.6148`

Updated decision:

Architecture search is eligible to reopen for a synthetic genetic BioTech-JEPA Tier 1 test, using the valid substitute teacher structure. Norman can be used as a genetic perturbation reference with batch/dose ignored, but it cannot validate batch disentanglement from this processed h5ad alone.

## BioTech-JEPA Tier 1 Addendum

The user then instructed: run the next BioTech-JEPA step for both synthetic and Norman, and document the instruction history.

Implemented code:

- `perturb_jepa/models/biotech_jepa.py`
- `perturb_jepa/training/biotech_losses.py`
- `perturb_jepa/training/biotech_trainer.py`
- `perturb_jepa/evaluation/biotech_metrics.py`
- `perturb_jepa/training/norman_biotech_batches.py`
- `scripts/train_biotech_jepa.py`
- `scripts/evaluate_biotech_jepa.py`
- `tests/test_biotech_jepa_model.py`
- `tests/test_norman_biotech_batches.py`

Validation:

- Focused suite: `5 passed`
- Post-evaluator focused suite: `3 passed`
- Standalone synthetic checkpoint evaluator completed successfully.

### BTJ001 Synthetic Genetic Anchor

Run:

```text
outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0
```

Key metrics:

- RNA->image recall@1: `0.1875`
- Image->RNA recall@1: `0.0000`
- Transition-to-target recall@1: `0.4375`
- Transition source cosine improvement: `+0.0161`
- Joint `z_bio` batch-probe accuracy: `0.1875`
- Joint `z_tech` batch-probe accuracy: `0.4375`
- Batch allocation gap: `+0.2500`
- Joint `z_bio` effective rank: `7.5103`

Decision label: `TIER1_DIAGNOSTIC_NO_PROMOTION`.

### BTJ002 Norman RNA-Only

Run:

```text
outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ002_norman_rna_only_seed0
```

Key metrics:

- RNA-only diagnostic: `1.0`
- Transition-to-target recall@1: `0.0625`
- Transition-to-target median rank: `8.5`
- Transition source cosine improvement: `+0.0313`
- Target `z_bio` effective rank: `7.4066`
- Target `z_bio` std mean: `0.0183`
- Target `z_tech` std mean: `0.1228`
- Batch probe: unavailable because batch is intentionally ignored for this processed Norman h5ad.

Decision label: `NORMAN_RNA_ONLY_DIAGNOSTIC_NO_PROMOTION`.

## Current Decision

No BioTech-JEPA candidate is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

The synthetic run shows the intended factorization direction (`z_tech` carries more batch signal than `z_bio`) but the transition gain is still small. Norman ran successfully under the user-specified metadata correction, but it is RNA-only and cannot validate imaging or batch disentanglement.
