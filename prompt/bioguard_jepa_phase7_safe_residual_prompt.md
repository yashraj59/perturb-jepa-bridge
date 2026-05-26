# BioGuard-JEPA Phase 7 Amendment: Risk-Controlled Residual Selection Above a Protected Transition Floor

## Direct instruction

Read and apply this file verbatim as the active Phase 7 amendment.

Do **not** continue BSJ004 by training longer. Do **not** launch BSJ005, BSJ006, or Norman from the old Phase 6 plan. Phase 6 already showed that a train-fitted spectral residual can improve train residual fit while reducing held-out transition improvement and recall below the protected full-ridge transition floor.

Start Phase 7 in:

```text
outputs/autoresearch_bioguard_jepa_phase7/
```

The new candidate family is **BioGuard-JEPA**:

```text
BioGuard-JEPA =
  real cross-modal/action-conditioned JEPA backbone
  + z_bio / z_tech separation
  + protected train-only full-ridge transition floor
  + residual targets defined relative to that floor
  + action-grouped cross-fitting
  + train-only calibration / risk-control gate
  + zero-default floor fallback
```

The scientific question is:

> Can a real action-conditioned biological JEPA safely learn residual transition structure above the full train-only action-ridge floor, using only train-internal validation/calibration to prevent residual overfit?

A Phase 7 candidate is not allowed to be called a success merely because it improves train fit, train delta cosine, or training residual MSE. It must preserve or improve the protected full-ridge floor on held-out perturbations without forbidden leakage.

---

## Current state that must be preserved

### Model of record

The protected rank-3 train-split-only PLS raw-linear readout remains the model of record unless a future Tier 3 pass explicitly supersedes it.

PLS is audit-only. It may be used as a protected baseline or short annealed bootstrap teacher, but it must not become the BioGuard-JEPA main representation path.

### Phase 6 facts

Phase 6 closed with:

```text
BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR
```

The exact Phase 6 floor and residual metrics are:

| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | floor gap | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| BSJ000 full-ridge audit | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | 0.0000 | PHASE6_REOPEN_BIOSPECTRAL_APPROVED |
| BSJ001 neural low-rank | 0.0046 | 0.3877 | 0.4074 | 6.7681 | 0.7585 | -0.0011 | BSJ001_KEEP_NEURAL_LOWRANK_MATCHES_ANALYTIC |
| BSJ002 floor wrapper | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | -0.0000 | BSJ002_KEEP_FLOOR_WRAPPER_MATCHES_FULL_RIDGE |
| BSJ003 rank ladder | 0.0057 | 0.3980 | 0.4815 | 10.2835 | 0.7744 | -0.0000 | BSJ003_KEEP_RANK_LADDER_PRESERVES_FLOOR |
| BSJ004 spectral residual | 0.0050 | 0.4059 | 0.4074 | 10.4574 | 0.8000 | -0.0007 | BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR |

Interpretation:

1. The full train-only action-ridge transition floor is reproducible.
2. The floor wrapper preserves the floor exactly when the residual is zero.
3. Rank-adaptive machinery can preserve the floor.
4. The failure moved to the residual branch.
5. The spectral residual increased delta cosine and delta rank, but harmed held-out transition improvement and recall@1.
6. Therefore Phase 7 must redesign residual **selection and deployment**, not merely residual capacity.

### Phase 4 / Phase 5 facts that still matter

The frozen-latent action-ridge floor has weak but real held-out signal:

```text
eval action_ridge_delta transition improvement = +0.0057
eval action_ridge_delta delta cosine = 0.3980
eval action_ridge_delta recall@1 = 0.4815
eval action_ridge_delta delta rank = 10.2835
```

The null source-as-target baseline has:

```text
transition improvement = 0.0000
delta cosine = 0.0000
delta prediction rank = 0.0000
```

A full neural ridge head reproduced the floor exactly, but a fixed low-rank control-affine operator fell below the floor. Therefore Phase 7 must preserve the full-ridge operator as a non-negotiable fallback.

---

## Literature-inspired design principles

Record these papers and ideas in `papers_consulted.md`. Do not overclaim that the implementation proves the theory of these papers; use them as design inspiration.

### JEPA / action-conditioned world modeling

Use the V-JEPA / V-JEPA 2 principle: prediction should happen in latent representation space, and actions can condition latent future-state prediction. In this project, the biological analogue is:

```text
control z_bio + perturbation action -> perturbed teacher z_bio
```

### Safe improvement over a baseline

Borrow the safety mindset from safe policy improvement: a learned update should not be deployed if the evidence says it can underperform the protected baseline. In Phase 7, the baseline is not a policy; it is the full train-only action-ridge transition floor.

### Conformal / risk-controlled selection

Borrow the split-calibration idea from conformal risk control: train a flexible candidate on one part of the training data, evaluate risk on a held-out calibration part, and select a threshold/gate only when calibration evidence supports it. In this project, use train-only calibration to decide whether a residual is allowed to activate above the ridge floor.

### Cross-fitting / sample splitting

Borrow from double/debiased machine learning: train nuisance/flexible components on one fold and evaluate them on a different fold to reduce overfit bias. In this project, residual heads are nuisance-like components that must be cross-fitted before they are trusted.

### Single-cell perturbation modeling

Borrow the perturbation-response structure from CellOT, GEARS, and CPA:

- perturbations are biological actions;
- held-out perturbation generalization is the important benchmark;
- population-level transitions matter, not only one embedding cosine;
- genetic action descriptors should be compositional and should not collapse into exact condition-key one-hot lookup.

---

## Non-negotiable identity and leakage rules

### Real JEPA identity

A full BioGuard-JEPA candidate must retain:

- online/context encoders;
- EMA target encoders;
- stop-gradient teacher targets;
- latent-space target prediction;
- RNA program JEPA when RNA is present;
- image region JEPA when imaging is present;
- RNA -> image and image -> RNA latent prediction when both modalities are present;
- action-conditioned control -> perturbed teacher-state prediction;
- separate biological and technical branches when batch/acquisition metadata exists;
- encoder readouts as the main path.

Operator-only probes are permitted in Phase 7 but cannot promote the model. They can only decide whether a full BioGuard-JEPA wrapper is worth running.

### Forbidden shortcuts

Do not use any of the following as model inputs, residual targets, calibration features, validation labels for fitting, or implicit lookup keys:

```text
condition_key
biological_key
exact target-key one-hot features
eval/test target means
pooled train+test target statistics
batch id as a biological transition shortcut
raw-linear PLS as the main representation path
held-out perturbation target rows for fitting or calibration
```

`condition_key` may appear only in metadata, grouping reports, leakage reports, and post-hoc evaluation tables.

### Protected floor rule

Every residual candidate must satisfy:

```text
prediction = source_z_bio + ridge_floor_delta + residual_gate * residual_scale * residual_delta
```

where:

```text
residual_gate = 0
```

or:

```text
residual_scale = 0
```

must reproduce the full-ridge floor exactly to numerical tolerance.

No candidate may replace the floor with a residual. A residual is only allowed to add evidence-backed correction on top of the floor.

---

## Phase 7 output files

Create and maintain:

```text
outputs/autoresearch_bioguard_jepa_phase7/results.tsv
outputs/autoresearch_bioguard_jepa_phase7/research_journal.md
outputs/autoresearch_bioguard_jepa_phase7/architectural_changes_log.md
outputs/autoresearch_bioguard_jepa_phase7/family_allocation.md
outputs/autoresearch_bioguard_jepa_phase7/BASELINE_REGISTRY.md
outputs/autoresearch_bioguard_jepa_phase7/papers_consulted.md
outputs/autoresearch_bioguard_jepa_phase7/external_resources.md
outputs/autoresearch_bioguard_jepa_phase7/identity_violations_considered.md
outputs/autoresearch_bioguard_jepa_phase7/residual_selection_report.md
outputs/autoresearch_bioguard_jepa_phase7/residual_calibration_report.md
outputs/autoresearch_bioguard_jepa_phase7/leakage_audit.md
outputs/autoresearch_bioguard_jepa_phase7/REOPENING_DECISION.md
outputs/autoresearch_bioguard_jepa_phase7/final_report.md
```

`results.tsv` must include at least:

```text
experiment_id
dataset
eval_split
seed
mode
status
transition_improvement
delta_cosine
recall_at_1
delta_rank
magnitude_ratio
floor_transition_improvement
floor_delta_cosine
floor_recall_at_1
floor_delta_rank
floor_gap_transition
floor_gap_recall
floor_gap_delta_cosine
cv_lcb_transition_gap
cv_lcb_recall_gap
residual_gate_mean
residual_gate_nonzero_fraction
residual_scale
train_residual_fit_metric
calibration_residual_fit_metric
action_negative_gap
leakage_status
decision_label
```

---

## Stage A: reproduce the floor and the Phase 6 residual failure

Before implementing new architecture, reproduce the key Phase 6 facts from cached train-only teacher latents.

Write:

```text
outputs/autoresearch_bioguard_jepa_phase7/stage_a_reproduction/PHASE7_REPRODUCTION.md
```

Required checks:

1. Full action-ridge floor matches BSJ000 / BSJ002:

```text
transition improvement ~= 0.0057
delta cosine ~= 0.3980
recall@1 ~= 0.4815
delta rank ~= 10.2835
```

2. Zero residual wrapper matches the full floor exactly.
3. The old BSJ004 residual-style training improves train residual fit but falls below held-out floor when evaluated without calibration.
4. No eval target rows are used in fitting, whitening, residual selection, or calibration.

If Stage A cannot reproduce these facts, stop and write:

```text
PHASE7_ABORT_REPRODUCTION_MISMATCH
```

No architecture changes are permitted after this failure.

---

## Stage B: train-only action-grouped cross-fitting protocol

Implement a train-only splitting utility for residual selection.

### Required module

Create:

```text
perturb_jepa/training/bioguard_splits.py
```

with:

```python
class ActionGroupedResidualSplitConfig: ...
class ActionGroupedResidualSplitter: ...
```

The splitter must support:

- grouped folds by perturbation/action descriptor;
- optional stratification by cell line / batch when available;
- no eval/test rows in any fold;
- deterministic folds from seed;
- leave-action-out folds when enough actions exist;
- fallback grouped K-fold when leave-action-out is too small;
- explicit report of rows/actions in fit/calibration/validation partitions.

### Internal split semantics

Use only training rows to construct:

```text
fit_folds: used to fit residual candidates
calibration_folds: used to choose residual scale/gate/threshold
selection_summary: used to decide whether to allow residual deployment
```

Do not use the held-out perturbation eval split for residual selection. The eval split is scoring-only.

### Required tests

Create:

```text
tests/test_bioguard_splits.py
```

Tests must verify:

- no eval rows enter fit or calibration;
- action-group leakage is prevented when leave-action-out is requested;
- fold construction is deterministic;
- each fold writes explicit action coverage;
- tiny-data fallback is documented and does not silently random-row split actions when action-grouped splitting was requested.

---

## Stage C: residual target construction relative to the floor

Implement residual target caching.

### Required module

Create:

```text
perturb_jepa/operators/bioguard_residuals.py
```

with:

```python
class RidgeFloorHead: ...
class ResidualTargetCache: ...
class ResidualWhitening: ...
```

Definitions:

```text
teacher_delta = target_teacher_z_bio - source_teacher_z_bio
floor_delta   = train-only full action-ridge prediction
residual_star = teacher_delta - floor_delta
```

All whitening / normalization statistics must be fit on training-fit folds only, then applied to calibration/eval scoring.

### Required reports

Write:

```text
residual_target_stats.md
```

with:

- teacher delta rank;
- floor delta rank;
- residual target rank;
- residual target magnitude;
- residual target action-probe accuracy;
- residual target batch-probe accuracy where batch exists;
- floor residual error by action;
- residual near-zero fraction;
- residual train/calibration distribution shift.

### Required stop condition

If residual targets have near-zero rank, near-zero magnitude, or are dominated by batch/acquisition leakage, stop with:

```text
PHASE7_CLOSE_NO_VALID_RESIDUAL_TARGET
```

---

## Stage D: residual candidate library

Implement a small residual library, but do not deploy any candidate until Stage E selection passes.

### Required candidates

1. **ZeroResidual**

Always returns zero. Must preserve the full floor exactly.

2. **SpectralResidualHead**

A corrected version of BSJ004 that is trained only on `residual_star`, not the full teacher delta.

3. **KernelResidualHead**

A nonparametric residual smoother over:

```text
action descriptor
source z_bio
cell-line/context descriptor when present
```

The kernel may be RBF, cosine, or kNN, but it must use train-only support and must include a calibration-selected shrinkage parameter.

4. **ProgramActionResidualHead**

A gene/program action descriptor residual inspired by genetic perturbation structure. It may use gene multi-hot action descriptors, known gene-program assignments, or train-derived program clusters. It must not use exact condition-key one-hots.

5. **PrototypeTransportResidualHead**

A population-level residual based on condition prototypes. It should estimate a correction between floor-predicted perturbed prototypes and teacher perturbed prototypes using train-only matched/control-to-perturbed pairs. This is allowed only if population prototypes are available without eval target leakage.

6. **ResidualEnsembleHead**

A calibrated weighted ensemble of the above residuals. Weights are selected only by train-internal cross-fitting/calibration.

### Required implementation rule

All residual heads must have a parameter or switch such that:

```text
residual contribution = 0
```

at initialization or when disabled, yielding exact floor preservation.

---

## Stage E: cross-fitted residual selection and risk-control gate

This is the core of Phase 7.

### Required module

Create:

```text
perturb_jepa/operators/bioguard_selection.py
```

with:

```python
class ResidualCandidateResult: ...
class CrossFittedResidualSelector: ...
class ResidualRiskControlGate: ...
class FloorPreservationContract: ...
```

### Selection metrics

For each candidate and fold, compute candidate minus floor:

```text
transition_improvement_gap
recall_at_1_gap
delta_cosine_gap
delta_rank_gap
magnitude_ratio_gap
```

Also compute:

```text
action_negative_gap
permutation_action_gap
residual_train_to_calibration_gap
residual_gate_nonzero_fraction
```

### Lower confidence bound rule

Compute a conservative lower confidence bound across folds for at least:

```text
transition_improvement_gap
recall_at_1_gap
```

Use bootstrap, t-interval, empirical quantile, or a clearly documented conservative small-sample rule.

A candidate may be deployed only if:

```text
LCB(transition_improvement_gap) > 0
LCB(recall_at_1_gap) >= 0
mean(delta_cosine_gap) >= 0
residual_train_to_calibration_gap is not pathological
action_negative_gap <= 0 or clearly below candidate true-action gap
magnitude_ratio remains in allowed band
```

Do not choose a residual by eval/test performance.

### Gate semantics

The final residual gate must default to floor fallback.

For each example:

```text
if calibrated confidence is high:
    use residual with calibrated scale alpha
else:
    residual_gate = 0
```

The gate may be scalar, per-action, per-action-family, or example-level. But it must be calibrated using train-only folds and must report:

```text
residual_gate_mean
residual_gate_nonzero_fraction
residual_gate_by_action
residual_scale_alpha
floor_fallback_fraction
```

### Required stop condition

If no residual candidate passes train-internal cross-fitted selection, stop with:

```text
PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA
```

This is a scientifically valid closure, not a failure.

---

## Stage F: BioGuard transition head

Only if Stage E passes, implement:

```text
perturb_jepa/operators/bioguard_transition.py
```

with:

```python
class BioGuardTransitionHead: ...
```

The head must combine:

```text
floor_delta = RidgeFloorHead(source_z_bio, action)
residual_delta = SelectedResidualHead(source_z_bio, action, context)
gate, scale = ResidualRiskControlGate(...)
predicted_delta = floor_delta + gate * scale * residual_delta
predicted_target_z_bio = source_z_bio + predicted_delta
```

### Contracts

Write tests in:

```text
tests/test_bioguard_transition.py
```

Required tests:

- zero residual equals floor exactly;
- zero gate equals floor exactly;
- alpha zero equals floor exactly;
- floor parameters are frozen unless explicitly configured otherwise;
- residual cannot receive eval target rows;
- residual contribution ratio is logged;
- gate nonzero fraction is logged;
- action-negative residual cannot pass selection in a synthetic sanity test.

---

## Stage G: full BioGuard-JEPA wrapper

Only if the frozen-latent BioGuard transition head beats the full floor without violating Stage E, implement a full JEPA wrapper.

Create:

```text
perturb_jepa/models/bioguard_jepa.py
perturb_jepa/training/bioguard_losses.py
perturb_jepa/training/bioguard_trainer.py
perturb_jepa/evaluation/bioguard_metrics.py
scripts/train_bioguard_jepa.py
scripts/evaluate_bioguard_jepa.py
```

The full model must retain:

- BioTech/BioMechanistic JEPA identity;
- `z_bio` / `z_tech` separation where batch exists;
- cross-modal RNA/image latent prediction;
- action-conditioned transition JEPA;
- BioGuard floor + selected residual transition head;
- anti-collapse diagnostics;
- batch leakage diagnostics;
- residual gate diagnostics.

The loss must not let the residual branch dominate. Add explicit logging:

```text
floor_delta_norm
residual_delta_norm
gated_residual_delta_norm
residual_to_floor_ratio
residual_cap_hit_fraction
transition_loss_floor
transition_loss_residual
weighted_residual_loss_to_main_ratio
```

---

## Experiment plan

### BSG000: Phase 7 reproduction

Purpose: reproduce floor and old residual failure.

Decision labels:

```text
BSG000_PASS_REPRODUCED_PHASE6
BSG000_ABORT_REPRODUCTION_MISMATCH
```

### BSG001: residual target and split audit

Purpose: validate train-only action-grouped folds and residual targets.

Decision labels:

```text
BSG001_PASS_RESIDUAL_TARGET_VALID
BSG001_CLOSE_NO_VALID_RESIDUAL_TARGET
BSG001_ABORT_SPLIT_LEAKAGE
```

### BSG002: spectral residual with cross-fitted calibration

Purpose: retest the Phase 6 spectral idea under the new residual target and selection contract.

Run only after BSG001 passes.

Decision labels:

```text
BSG002_KEEP_SPECTRAL_RESIDUAL_PASSES_CV_GATE
BSG002_DISCARD_SPECTRAL_RESIDUAL_FAILS_CV_GATE
BSG002_DISCARD_SPECTRAL_RESIDUAL_BELOW_FLOOR
```

### BSG003: kernel residual with cross-fitted calibration

Purpose: test whether residuals are better captured by local train support than spectral global structure.

Decision labels:

```text
BSG003_KEEP_KERNEL_RESIDUAL_PASSES_CV_GATE
BSG003_DISCARD_KERNEL_RESIDUAL_FAILS_CV_GATE
BSG003_DISCARD_KERNEL_RESIDUAL_BELOW_FLOOR
```

### BSG004: program/action residual

Purpose: test whether genetic action structure improves held-out perturbation residuals.

Decision labels:

```text
BSG004_KEEP_PROGRAM_RESIDUAL_PASSES_CV_GATE
BSG004_DISCARD_PROGRAM_RESIDUAL_FAILS_CV_GATE
BSG004_DISCARD_PROGRAM_RESIDUAL_BELOW_FLOOR
```

### BSG005: calibrated residual ensemble

Purpose: combine only residuals that passed BSG002-BSG004 cross-fitted gates.

If no residual passed, do not run BSG005.

Decision labels:

```text
BSG005_KEEP_ENSEMBLE_ABOVE_FLOOR
BSG005_DISCARD_ENSEMBLE_BELOW_FLOOR
BSG005_CLOSE_NO_RESIDUAL_TO_ENSEMBLE
```

### BSG006: frozen-backbone BioGuard-JEPA wrapper

Purpose: attach the selected BioGuard transition head to the real JEPA backbone while keeping encoders frozen.

Run only if BSG005 is above floor.

Decision labels:

```text
BSG006_KEEP_FROZEN_BACKBONE_ABOVE_FLOOR
BSG006_DISCARD_FROZEN_BACKBONE_BELOW_FLOOR
BSG006_ABORT_JEPA_IDENTITY_VIOLATION
```

### BSG007: end-to-end BioGuard-JEPA

Purpose: train the full model end-to-end with the residual guarded by calibration. The residual must stay optional and cannot erase the floor.

Run only if BSG006 passes.

Decision labels:

```text
BSG007_TIER1_KEEP_READY_FOR_MULTI_SEED
BSG007_DISCARD_BELOW_FLOOR
BSG007_DISCARD_BATCH_LEAKAGE
BSG007_ABORT_JEPA_IDENTITY_VIOLATION
```

### BSG008: Norman RNA-only diagnostic

Run only after synthetic BSG007 passes.

Norman remains RNA-only and cannot validate imaging or batch disentanglement unless real batch/acquisition metadata is recovered.

Decision labels:

```text
BSG008_NORMAN_DIAGNOSTIC_SIGNAL
BSG008_NORMAN_DIAGNOSTIC_NO_SIGNAL
BSG008_NORMAN_ABORT_CONTEXT_MISMATCH
```

---

## Gates and thresholds

### Floor values to preserve

Use these as the scoring floor on `synth_genetic_anchor_lite/test_heldout_perturbation` unless Stage A reproduction gives a documented tiny numerical tolerance:

```text
floor_transition_improvement = 0.0057
floor_delta_cosine = 0.3980
floor_recall_at_1 = 0.4815
floor_delta_rank = 10.2835
floor_magnitude_ratio = 0.7744
```

### Hard discard gates

Discard and stop the relevant sequence if any candidate has:

```text
transition_improvement < floor_transition_improvement - 0.0001
recall_at_1 < floor_recall_at_1 - 0.0001
delta_cosine < floor_delta_cosine - 0.02
floor_gap_transition < -0.0001
floor_gap_recall < -0.0001
residual_to_floor_ratio > configured cap
magnitude_ratio outside [0.4, 1.4]
condition_key_feature_present = 1
teacher_stop_gradient_verified = 0
encoder_path_used = 0 for full JEPA candidates
pls_raw_linear_main_path_used = 1 for full JEPA candidates
```

### Keep gates

A Tier 1 operator probe can be kept only if:

```text
transition_improvement >= floor_transition_improvement + 0.0005
recall_at_1 >= floor_recall_at_1
delta_cosine >= floor_delta_cosine
cv_lcb_transition_gap > 0
cv_lcb_recall_gap >= 0
action_negative_gap <= 0 or action_negative_gap << true_action_gap
leakage_status = PASS
```

A full BioGuard-JEPA candidate can be kept only if all real-JEPA identity checks pass and the BioGuard transition metrics satisfy the same floor-preserving gates.

No Phase 7 Tier 1 result may promote the model of record. A clean Phase 7 pass only permits a future Tier 2/Tier 3 amendment.

---

## Required leakage audit

For every experiment BSG000-BSG008, write:

```text
experiments/<experiment_id>/leakage_report.md
```

The report must explicitly answer:

1. Were eval/test target rows used for fitting? `yes/no`
2. Were eval/test target rows used for whitening/statistics? `yes/no`
3. Were eval/test target rows used for residual calibration or selection? `yes/no`
4. Was `condition_key` used as a model feature? `yes/no`
5. Was `biological_key` used as a model feature? `yes/no`
6. Were exact target-key one-hot features used? `yes/no`
7. Was batch id used as a biological transition shortcut? `yes/no`
8. Were raw-linear PLS features used as the main representation path? `yes/no`
9. Were any candidate choices based on eval/test performance? `yes/no`

Any `yes` to 1-9 is a hard failure except where the question is purely post-hoc reporting and the report explains why it is not a model input or selection signal.

---

## Required diagnostics

Every residual experiment must log:

```text
floor_delta_norm
raw_residual_delta_norm
gated_residual_delta_norm
residual_to_floor_ratio
residual_gate_mean
residual_gate_nonzero_fraction
residual_gate_by_action
residual_scale_alpha
floor_fallback_fraction
residual_cap_hit_fraction
train_residual_mse
calibration_residual_mse
train_to_calibration_residual_gap
transition_improvement_gap_by_fold
recall_gap_by_fold
delta_cosine_gap_by_fold
action_negative_gap
permutation_action_gap
```

Full JEPA candidates must additionally log:

```text
encoder_path_used
pls_raw_linear_main_path_used
condition_key_feature_present
teacher_stop_gradient_verified
separate_bio_and_tech_latents_present
rna_to_image_recall_at_1
image_to_rna_recall_at_1
z_bio_effective_rank
z_tech_effective_rank
z_bio_batch_probe_accuracy
z_tech_batch_probe_accuracy
batch_allocation_gap
```

---

## Stop conditions

Stop immediately and write `final_report.md` if any of the following occurs:

1. Stage A cannot reproduce the Phase 6 floor.
2. Train-only residual splitting leaks eval/test rows.
3. Residual targets are invalid or batch-dominated.
4. No residual passes train-internal cross-fitted selection.
5. A selected residual falls below the full-ridge floor on held-out perturbation scoring.
6. A full JEPA wrapper violates JEPA identity.
7. Batch leakage dominates `z_bio` on a dataset where batch disentanglement is testable.
8. Residual gates are almost always on and residual-to-floor ratio is cap-bound.
9. Action-negative or permuted-action residuals pass the same selection gate as true actions.
10. The code cannot prove that eval/test targets were excluded from fitting, whitening, calibration, and selection.

When a stop condition fires, do not continue to the next experiment. Write `final_report.md`, update `results.tsv`, and stop the loop.

---

## Final report requirements

The Phase 7 final report must include:

- decision label;
- model of record status;
- which BSG experiments ran and why later ones did or did not run;
- exact floor values used;
- candidate vs floor table;
- train-internal cross-fit selection table;
- calibration/gating table;
- leakage audit summary;
- JEPA identity status;
- whether any full JEPA candidate was actually trained;
- Norman status, if run;
- recommendation: close, amend, or prepare Tier 2.

Use one of these final decision labels:

```text
PHASE7_ABORT_REPRODUCTION_MISMATCH
PHASE7_CLOSE_NO_VALID_RESIDUAL_TARGET
PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA
PHASE7_DISCARD_SELECTED_RESIDUAL_BELOW_FLOOR
PHASE7_KEEP_BIOGUARD_OPERATOR_READY_FOR_JEPA_WRAPPER
PHASE7_KEEP_BIOGUARD_JEPA_TIER1_READY_FOR_TIER2
PHASE7_CLOSE_IDENTITY_OR_LEAKAGE_FAILURE
```

---

## Do-not-run list

Do not run:

- BSJ005, BSJ006, or BSJ007 from the old Phase 6 plan;
- any residual trained and selected on the same rows without cross-fitting;
- any residual selected by held-out eval performance;
- any full end-to-end JEPA before the frozen-latent BioGuard transition head beats the floor;
- any Norman run before synthetic BioGuard passes;
- any chemical-dose interpretation for Norman unless real chemical dose metadata is recovered;
- any batch-disentanglement claim for Norman unless real batch/acquisition metadata is recovered.

---

## Minimal launch command targets

Add scripts so the following conceptual commands work after implementation:

```bash
python scripts/run_bioguard_phase7_reproduction.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --output-dir outputs/autoresearch_bioguard_jepa_phase7/stage_a_reproduction
```

```bash
python scripts/run_bioguard_residual_selection.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --folds 4 \
  --candidate all \
  --output-dir outputs/autoresearch_bioguard_jepa_phase7/experiments/BSG001_residual_selection_seed0
```

```bash
python scripts/train_bioguard_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --mode frozen_backbone \
  --output-dir outputs/autoresearch_bioguard_jepa_phase7/experiments/BSG006_frozen_backbone_seed0
```

Exact CLI flags may differ, but the implemented scripts must expose the same semantic controls.

---

## One-sentence thesis

BioGuard-JEPA is not another larger residual head. It is a floor-preserving biological action JEPA that only activates residual transition structure when train-only cross-fitted calibration shows the residual is likely to safely improve the protected ridge transition floor.
