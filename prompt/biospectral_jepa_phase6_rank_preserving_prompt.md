# Phase 6 Codex Amendment: BioSpectral-JEPA, a rank-preserving real JEPA transition operator

## One-line instruction

Read this amendment completely and apply it verbatim. Do **not** continue BOJ002 by training longer. Do **not** launch BOJ003/BOJ004 from the old Phase 5 plan. Phase 5 showed that the analytical/neural full-ridge transition floor is reproducible, but the structured low-rank control-affine operator loses held-out rank, recall, and transition improvement. Phase 6 must first determine whether that is an implementation/factorization bug or a real rank bottleneck, then implement a floor-preserving, rank-adaptive **BioSpectral-JEPA** only if the diagnostics justify it.

---

## Active model of record

The protected model of record remains:

```text
rank-3 train-split-only PLS raw-linear readout
```

Rules:

1. The protected PLS raw-linear readout remains the model of record unless a future Tier 3 pass explicitly supersedes it.
2. PLS is audit-only. It must not become the BioSpectral-JEPA representation path.
3. BOJ001 is retained only as a frozen operator-floor audit reference.
4. BOJ002 is discarded as a below-floor structured-operator failure.
5. No Phase 6 Tier 1 or Tier 2 result can promote the model. Tier 1/Tier 2 can only justify a future Tier 3 protocol.

---

## Latest evidence that motivates Phase 6

### Phase 5 facts to preserve

Phase 5 ended with:

```text
PHASE5_OPERATOR_FLOOR_REPRODUCED_BUT_NEURAL_FAILURE_CLOSE
```

Key outcomes:

| item | result |
|---|---|
| Stage A operator floor reproduction | passed |
| Stage B sign/gradient/loss contracts | passed, `5 passed` |
| BOJ001 frozen neural ridge-equivalence | passed; matched action-ridge floor |
| BOJ002 frozen low-rank control-affine operator | failed below ridge floor |
| BOJ003 frozen-backbone integration | not run |
| BOJ004 end-to-end JEPA | not run |
| BOJ005 Norman RNA-only diagnostic | not run |

Critical metrics:

| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | decision |
|---|---:|---:|---:|---:|---:|---|
| BOJ001 full neural ridge | `0.0057` | `0.3980` | `0.4815` | `10.2835` | `0.7744` | `TIER1_KEEP_OPERATOR_MATCHES_FLOOR` |
| BOJ002 low-rank control-affine | `0.0025` | `0.3603` | `0.3704` | `7.5812` | `0.7579` | `TIER1_DISCARD_OPERATOR_BELOW_FLOOR` |

Interpretation:

```text
The neural form can reproduce the full train-only action-ridge floor.
The fixed low-rank control-affine form cannot preserve that floor.
The next bottleneck is not JEPA identity, sign convention, or basic gradient flow.
The next bottleneck is structured operator capacity / rank preservation / factorization.
```

### Phase 4 latent/delta facts to preserve

Use the same cached teacher latents from the Phase 4 delta audit unless the cache is missing or corrupted. The audit used:

```text
Dataset: synth_genetic_anchor_lite
Eval split: test_heldout_perturbation
Latent cache model: BTJ001 BioTech-JEPA checkpoint
```

Forbidden shortcuts already checked in the Phase 4 audit:

```text
no condition-key input
no biological-key one-hot input
no test target means
no pooled train+test targets
```

Teacher-delta statistics show real target structure:

| split | delta mean norm | delta std mean | delta effective rank | perturbation NN recall@1 |
|---|---:|---:|---:|---:|
| train | `0.4310` | `0.0846` | `13.5627` | `0.3056` |
| eval | `0.4252` | `0.0832` | `11.7819` | `0.5926` |

Simple transition baselines:

| split | baseline | transition improvement | recall@1 | delta cosine | magnitude ratio | predicted delta rank |
|---|---|---:|---:|---:|---:|---:|
| eval | action_ridge_delta | `0.0057` | `0.4815` | `0.3980` | `0.7744` | `10.2835` |
| eval | action_low_rank_ridge | `0.0046` | `0.4074` | `0.3877` | `0.7585` | `6.7681` |
| eval | source_as_target_null | `0.0000` | `0.2963` | `0.0000` | `0.0000` | `0.0000` |
| train | action_ridge_delta | `0.0769` | `0.7778` | `0.8483` | `0.8535` | `11.6312` |
| train | action_low_rank_ridge | `0.0740` | `0.6111` | `0.8230` | `0.8318` | `7.0452` |

Important observation:

```text
The analytical eval low-rank ridge floor is weaker than full ridge but still better than BOJ002.
Therefore BOJ002 may have both:
  (1) a factorization/training implementation gap relative to analytical low-rank ridge, and
  (2) a true rank bottleneck relative to full ridge.
```

Phase 6 must separate those two possibilities before architecture search.

---

## Literature anchors to record in `papers_consulted.md`

Before code changes, write or update:

```text
outputs/autoresearch_biospectral_jepa_phase6/papers_consulted.md
```

Include these anchors. Do not implement any paper wholesale. Extract only the minimal mechanism that maps to the current failure.

### JEPA / action-conditioned world modeling

1. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**
   - URL: https://arxiv.org/abs/2506.09985
   - Technique to extract: action-conditioned latent prediction, not pixel reconstruction.
   - Mapping: perturbation action + control biological latent predicts perturbed biological teacher latent.
   - Identity constraint: keep online/context encoders, EMA target encoders, stop-gradient targets, latent target prediction.

### Control/operator learning

2. **Dynamic Mode Decomposition with Control**
   - URL: https://epubs.siam.org/doi/10.1137/15M1013857
   - Technique to extract: separate state dynamics and control/action effects; audit low-order approximations before trusting them.
   - Mapping: distinguish full-rank action ridge, reduced-rank action operator, and action-conditioned latent transition.

3. **Koopman / EDMD-style operator learning**
   - Example URL: https://faculty.washington.edu/kutz/page1/page13/
   - Technique to extract: use linear operators in lifted/latent spaces, but validate finite-rank approximations.
   - Mapping: `z_bio` is a learned observable space; perturbation actions induce operators or delta fields.

### Rank-adaptive parameterization from other ML fields

4. **LoRA / adaptive low-rank and mixture-of-rank experts**
   - Example URL: https://arxiv.org/abs/2505.22694
   - Technique to extract: fixed rank can be too restrictive; use adaptive rank selection or mixtures of rank experts.
   - Mapping: replace fixed-rank control-affine bottleneck with rank ladder experts plus full-ridge floor preservation.

### Single-cell perturbation modeling

5. **CellOT: Learning single-cell perturbation responses using neural optimal transport**
   - URL: https://www.nature.com/articles/s41592-023-01969-x
   - Technique to extract: perturbations are distributional maps, not just centroid deltas.
   - Mapping: keep population/prototype metrics and do not overclaim from centroid-only operator metrics.

6. **CPA: Compositional perturbation autoencoder**
   - URL: https://www.embopress.org/doi/full/10.15252/msb.202211517
   - Technique to extract: perturbation, context, dose/composition factors can be compositional.
   - Mapping: genetic actions may be single genes or gene pairs; Norman dose remains fixed guide presence, not chemical concentration.

7. **GEARS-style genetic perturbation priors**
   - URL: https://www.nature.com/articles/s41587-023-01905-6
   - Technique to extract: gene/gene-pair action descriptors and gene-network priors can help held-out perturbation generalization.
   - Mapping: optional later action features, not a substitute for first reproducing the operator floor.

### Statistical / uncertainty-inspired operators

8. **GPerturb or Gaussian-process perturbation modeling**
   - URL: https://www.nature.com/articles/s41467-025-61165-7
   - Technique to extract: uncertainty-aware perturbation response and kernelized effects.
   - Mapping: optional kernel residual branch only after full-ridge and spectral-rank contracts pass.

---

## Phase 6 research thesis

The Phase 5 failure is not “JEPA failed.” Phase 5 showed that a neural linear head can reproduce the action-ridge transition floor exactly. The failure is that a **fixed low-rank control-affine constraint destroys too much held-out delta geometry**.

The new thesis:

```text
A real JEPA perturbation world model needs a rank-preserving transition operator.
The operator should retain the full train-only ridge floor, then add controlled spectral or kernel residuals only when they improve held-out perturbation transition without leakage, rank collapse, or cross-modal regression.
```

The proposed candidate is:

```text
BioSpectral-JEPA =
  real cross-modal BioTech/BioOperator JEPA backbone
  + protected z_bio / z_tech separation
  + train-only full-ridge transition floor
  + rank-adaptive spectral transition residual
  + floor-preservation contracts
  + spectral/rank diagnostics
  + optional kernel residual if spectral residual has evidence
```

Do **not** reduce this to “low-rank operator but bigger.” The novelty is a **floor-preserving, rank-adaptive transition operator** inside a real action-conditioned JEPA.

---

## Real-JEPA identity requirements

A model may be called BioSpectral-JEPA only if it satisfies all of these:

1. Uses online/context RNA encoder.
2. Uses online/context image encoder when image data is present.
3. Uses EMA target RNA encoder.
4. Uses EMA target image encoder when image data is present.
5. Uses stop-gradient target latents.
6. Predicts teacher latent targets, not primarily raw reconstruction.
7. Uses target-query prediction for RNA program / image region / cross-modal targets.
8. Uses RNA→image and image→RNA latent prediction when both modalities exist.
9. Uses control `z_bio` + perturbation action → perturbed teacher `z_bio` transition prediction.
10. Keeps `z_bio` and `z_tech` separated when batch/acquisition information exists.
11. Does not use raw-linear PLS as the main representation path.
12. Does not use `condition_key`, `biological_key`, exact target-key one-hot features, test target means, or pooled train+test target statistics.

Operator-only probes such as BSJ000–BSJ004 are **not** full JEPA promotions. They are allowed as necessary contracts before building the full BioSpectral-JEPA wrapper.

---

## Output directory

Create:

```text
outputs/autoresearch_biospectral_jepa_phase6/
```

Required files:

```text
outputs/autoresearch_biospectral_jepa_phase6/results.tsv
outputs/autoresearch_biospectral_jepa_phase6/research_journal.md
outputs/autoresearch_biospectral_jepa_phase6/architectural_changes_log.md
outputs/autoresearch_biospectral_jepa_phase6/family_allocation.md
outputs/autoresearch_biospectral_jepa_phase6/BASELINE_REGISTRY.md
outputs/autoresearch_biospectral_jepa_phase6/papers_consulted.md
outputs/autoresearch_biospectral_jepa_phase6/external_resources.md
outputs/autoresearch_biospectral_jepa_phase6/identity_violations_considered.md
outputs/autoresearch_biospectral_jepa_phase6/rank_bottleneck_audit/RANK_BOTTLENECK_AUDIT.md
outputs/autoresearch_biospectral_jepa_phase6/final_report.md
```

Append one row per experiment to `results.tsv` with at least:

```text
experiment_id	stage	mode	seed	dataset	eval_split	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio	floor_gap	identity_pass	decision	artifact_dir
```

---

## Files to read before coding

Read these completely before changing code:

```text
outputs/autoresearch_biooperator_jepa_phase5/final_report.md
outputs/autoresearch_biooperator_jepa_phase5/research_journal.md
outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md
outputs/autoresearch_biomech_jepa_phase3/final_report.md
outputs/autoresearch_biotech_jepa_phase2/final_report.md
outputs/autoresearch_biotech_jepa_phase2/NORMAN_CONTEXT_AUDIT.md
perturb_jepa/models/biotech_jepa.py
perturb_jepa/training/biotech_losses.py
perturb_jepa/evaluation/biotech_metrics.py
perturb_jepa/models/biooperator_jepa.py
perturb_jepa/training/biooperator_losses.py
perturb_jepa/evaluation/biooperator_metrics.py
scripts/train_biooperator_jepa.py
scripts/evaluate_biooperator_jepa.py
tests/test_biooperator_contracts.py
tests/test_biooperator_losses.py
```

If a listed file is absent, document it in `final_report.md` and infer the nearest existing file from the repository. Do not silently skip missing evidence.

---

## Stage A: exact baseline registry and reproducibility check

### A1. Register fixed floors

Write `BASELINE_REGISTRY.md` with these fixed Phase 5 / Phase 4 values:

```text
full_action_ridge_eval_transition_improvement = 0.0057
full_action_ridge_eval_delta_cosine = 0.3980
full_action_ridge_eval_recall@1 = 0.4815
full_action_ridge_eval_delta_rank = 10.2835
full_action_ridge_eval_magnitude_ratio = 0.7744

low_rank_action_ridge_eval_transition_improvement = 0.0046
low_rank_action_ridge_eval_delta_cosine = 0.3877
low_rank_action_ridge_eval_recall@1 = 0.4074
low_rank_action_ridge_eval_delta_rank = 6.7681
low_rank_action_ridge_eval_magnitude_ratio = 0.7585

source_as_target_eval_transition_improvement = 0.0000
source_as_target_eval_recall@1 = 0.2963
```

### A2. Reproduce floors from cached latents

Run or write a script such as:

```bash
python scripts/run_biospectral_rank_audit.py \
  --latent-cache outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/latent_cache.pt \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --output-dir outputs/autoresearch_biospectral_jepa_phase6/rank_bottleneck_audit \
  --device cuda
```

The script must reproduce:

```text
action_ridge_delta eval improvement ≈ 0.0057
action_low_rank_ridge eval improvement ≈ 0.0046
```

Allowed numerical tolerance:

```text
absolute metric drift <= 1e-4 for reproduced deterministic values
```

If the floors cannot be reproduced, stop and write:

```text
PHASE6_ABORT_FLOOR_REPRODUCTION_FAILED
```

Do not implement BioSpectral-JEPA.

---

## Stage B: rank bottleneck audit before architecture changes

Write all results to:

```text
outputs/autoresearch_biospectral_jepa_phase6/rank_bottleneck_audit/RANK_BOTTLENECK_AUDIT.md
```

### B1. Full-ridge singular/eigen spectrum

Compute on train and eval:

1. Teacher delta covariance eigenvalues.
2. Full-ridge predicted delta covariance eigenvalues.
3. Low-rank ridge predicted delta covariance eigenvalues.
4. BOJ002 predicted delta covariance eigenvalues if checkpoint/output is available.
5. Effective rank.
6. Spectral entropy.
7. Cumulative variance explained by top `1, 2, 4, 6, 8, 10, 12, 16, full` components.
8. Principal angles between teacher-delta subspace and predicted-delta subspace.
9. Per-perturbation delta direction agreement.
10. Per-action residual norm after full ridge and after low-rank ridge.

### B2. Reduced-rank sweep

Fit train-only reduced-rank ridge models at:

```text
ranks = [1, 2, 4, 6, 8, 10, 12, 16, 24, full]
```

For each rank, report:

```text
train transition improvement
train delta cosine
train recall@1
train delta rank
train magnitude ratio

eval transition improvement
eval delta cosine
eval recall@1
eval median rank
eval delta rank
eval magnitude ratio
floor gap vs full ridge
```

Decision from B2:

```text
If no rank below full reaches full-ridge eval recall@1 and rank within tolerance,
then fixed low-rank is a real bottleneck.

If a moderate rank reaches full-ridge eval metrics analytically,
then BOJ002 failure is likely an implementation/factorization/training bug.
```

### B3. Neural low-rank equivalence contract

Before any new architecture, implement or run:

```text
NeuralReducedRankRidgeHead
```

It must exactly reproduce the analytical low-rank ridge at the same rank.

Gate:

```text
abs(neural_low_rank_eval_improvement - analytical_low_rank_eval_improvement) <= 1e-4
abs(neural_low_rank_eval_delta_cosine - analytical_low_rank_eval_delta_cosine) <= 1e-4
abs(neural_low_rank_eval_recall@1 - analytical_low_rank_eval_recall@1) <= 1e-4
abs(neural_low_rank_eval_delta_rank - analytical_low_rank_eval_delta_rank) <= 1e-3
```

If this fails, the next work is implementation debugging, not BioSpectral-JEPA. Stop with:

```text
PHASE6_STOP_NEURAL_LOWRANK_EQUIVALENCE_FAILED
```

### B4. Control-affine decomposition audit

If BOJ002 code/checkpoint is available, compare it against analytical low-rank ridge:

1. Same action features?
2. Same intercept handling?
3. Same whitening/unwhitening?
4. Same ridge regularization?
5. Same train rows only?
6. Same output normalization?
7. Same rank orientation?
8. Same bias term?
9. Same feature standardization?
10. Same action descriptor dimensions?

Write exact discrepancy table.

### B5. Reopening decision

At the end of Stage B, write:

```text
outputs/autoresearch_biospectral_jepa_phase6/rank_bottleneck_audit/REOPENING_DECISION.md
```

Allowed labels:

```text
PHASE6_REOPEN_BIOSPECTRAL_APPROVED
PHASE6_STOP_FLOOR_REPRODUCTION_FAILED
PHASE6_STOP_NEURAL_LOWRANK_EQUIVALENCE_FAILED
PHASE6_STOP_CONTROL_AFFINE_BUG_ONLY
PHASE6_STOP_NO_RANK_HEADROOM
```

Reopen only if:

1. Full-ridge floor is reproduced.
2. Low-rank ridge floor is reproduced.
3. Neural low-rank equivalence passes, or the precise bug is fixed and retested.
4. The audit shows either:
   - fixed low-rank bottleneck is real, or
   - rank-adaptive/floor-preserving residual has plausible headroom.
5. No leakage check fails.
6. The floor-preserving wrapper can be tested without using eval/test target means in training.

---

## Stage C: implement BioSpectral-JEPA only after reopening is approved

Only proceed if `REOPENING_DECISION.md` says:

```text
PHASE6_REOPEN_BIOSPECTRAL_APPROVED
```

### C1. New files

Create the smallest maintainable implementation:

```text
perturb_jepa/models/biospectral_jepa.py
perturb_jepa/training/biospectral_losses.py
perturb_jepa/evaluation/biospectral_metrics.py
perturb_jepa/training/biospectral_operator.py
scripts/run_biospectral_rank_audit.py
scripts/train_biospectral_jepa.py
scripts/evaluate_biospectral_jepa.py
tests/test_biospectral_operator_contracts.py
tests/test_biospectral_jepa_identity.py
tests/test_biospectral_no_leakage.py
```

### C2. Required classes

Implement these classes or equivalent names.

#### `BioSpectralJEPAConfig`

Fields:

```python
@dataclass(frozen=True)
class BioSpectralJEPAConfig:
    shared_dim: int = 32
    bio_dim: int = 24
    tech_dim: int = 8
    predictor_dim: int = 64
    action_dim: int = 32
    rank_ladder: tuple[int, ...] = (4, 8, 12, 24)
    use_full_ridge_floor: bool = True
    freeze_ridge_floor: bool = True
    residual_init_scale: float = 0.0
    residual_norm_cap: float = 0.25
    spectral_entropy_weight: float = 0.05
    covariance_shape_weight: float = 0.05
    floor_distillation_weight: float = 1.0
    delta_direction_weight: float = 1.0
    endpoint_jepa_weight: float = 1.0
    source_improvement_hinge_weight: float = 0.5
    anti_collapse_weight: float = 0.1
    allow_kernel_residual: bool = False
    kernel_num_features: int = 128
    kernel_residual_init_scale: float = 0.0
```

#### `RidgeFloorHead`

Purpose:

```text
Train-only analytical full-ridge action -> delta map.
Must reproduce BOJ001 floor exactly when wrapped in neural/module form.
```

Rules:

1. Fitted only on train source/target teacher latents.
2. No eval/test target latents may be used for fitting.
3. May be frozen or lightly trainable only after BSJ004 passes.
4. Default: frozen.
5. Must log floor metrics every evaluation.

#### `DeltaSpectralBasis`

Purpose:

```text
Train-only spectral basis for teacher deltas and ridge residuals.
```

Stores:

```text
teacher_delta_mean
teacher_delta_eigenvectors
teacher_delta_eigenvalues
ridge_pred_delta_mean
ridge_residual_basis
rank cumulative variance table
```

Rules:

1. Fit on train only.
2. Eval/test may be projected for diagnostics but not used to fit basis.
3. Must expose `project`, `reconstruct`, and `effective_rank` helpers.

#### `NeuralReducedRankRidgeHead`

Purpose:

```text
Exact neural equivalent of analytical reduced-rank ridge.
```

This exists primarily as a contract test. If it cannot reproduce analytical low-rank ridge, do not proceed.

#### `RankLadderTransitionHead`

Purpose:

```text
Adaptive mixture of spectral/rank experts.
```

Inputs:

```text
z_control_bio
action_features
optional perturbation/gene descriptor
```

Outputs:

```text
delta_ladder
router_weights
expert_deltas_by_rank
rank_usage
spectral_entropy
```

Mechanism:

1. Experts correspond to rank ladder values, for example `[4, 8, 12, 24, full]`.
2. A full-ridge/floor expert must be available unless explicitly disabled in an ablation.
3. Router is action-conditioned and optionally control-state-conditioned.
4. Router entropy is logged.
5. Router must not collapse to a single low-rank expert unless it improves metrics and preserves rank.

#### `SpectralResidualHead`

Purpose:

```text
Predict a small residual on top of the ridge floor in spectral coordinates.
```

Rules:

1. Residual branch initialized exactly zero.
2. Output starts as the ridge floor exactly.
3. Residual contribution ratio logged.
4. Residual norm capped relative to ridge floor norm.
5. Residual can be constrained to the ridge-residual basis or teacher-delta complement basis.
6. Residual must never be allowed to erase the ridge floor silently.

Diagnostics:

```text
residual_to_floor_norm_ratio
residual_cap_hit_fraction
residual_cosine_with_floor
residual_cosine_with_teacher_residual
predicted_delta_effective_rank
predicted_delta_spectral_entropy
```

#### Optional `KernelResidualHead`

Only implement this if spectral residual has a controlled non-destructive signal or if Stage B shows kernel headroom.

Purpose:

```text
Train-only kernel/random-feature residual over [z_control_bio, action_features].
```

Rules:

1. Initialized zero.
2. Must preserve ridge floor at initialization.
3. Must have norm cap and contribution logs.
4. Must not use eval/test target means.
5. Must not be run if spectral residual already fails floor preservation.

#### `FloorPreservingTransitionHead`

Final transition form:

```python
delta_floor = ridge_floor(action_features)
delta_rank_ladder = rank_ladder(z_control_bio, action_features)
delta_residual = spectral_residual(z_control_bio, action_features, delta_floor)

delta = delta_floor + residual_gate * delta_residual
z_pred_target = normalize_or_project(z_control_bio + delta)
```

Initialization contract:

```text
At step 0, z_pred_target must equal z_control_bio + delta_floor within numerical tolerance.
```

#### `BioSpectralJEPA`

Wraps the transition head in a real JEPA architecture:

1. Reuse or extend BioTech-JEPA online/context encoders.
2. Reuse EMA target encoders.
3. Preserve `z_bio` / `z_tech` split.
4. Use `z_bio` for transition and retrieval.
5. Use `z_tech` for technical branch only.
6. Use cross-modal RNA↔image JEPA when images exist.
7. Use RNA-only mode for Norman diagnostics.
8. Keep raw reconstruction auxiliary or disabled; reconstruction must not dominate.

---

## Stage D: losses and diagnostics

### D1. Loss terms

Implement in `perturb_jepa/training/biospectral_losses.py`:

```text
endpoint_latent_jepa_loss
whitened_delta_mse_loss
delta_direction_cosine_loss
source_improvement_hinge_loss
floor_distillation_loss
spectral_entropy_floor_loss
covariance_shape_loss
residual_norm_cap_loss
residual_orthogonality_loss
bio_tech_orthogonality_loss
VICReg-style anti-collapse loss
```

Recommended first objective for frozen-operator experiments:

```text
L =
  1.0 * endpoint_latent_jepa_loss
+ 1.0 * delta_direction_cosine_loss
+ 1.0 * floor_distillation_loss
+ 0.5 * source_improvement_hinge_loss
+ 0.05 * spectral_entropy_floor_loss
+ 0.05 * covariance_shape_loss
+ 0.1 * residual_norm_cap_loss
```

For full JEPA experiments:

```text
L =
  1.0 * endpoint_latent_jepa_loss
+ 1.0 * transition_delta_direction_loss
+ 1.0 * RNA_program_JEPA
+ 1.0 * image_region_JEPA
+ 2.0 * RNA_to_image_JEPA
+ 2.0 * image_to_RNA_JEPA
+ 0.1 * anti_collapse
+ 0.05 * bio_tech_orthogonality
+ 0.05 * auxiliary_count_or_reconstruction_loss
```

### D2. Required metric functions

Implement / expose:

```text
transition_source_cosine_improvement
absolute_target_cosine
delta_cosine
transition_to_target_recall@1
transition_to_target_median_rank
delta_magnitude_ratio
delta_prediction_effective_rank
delta_prediction_spectral_entropy
teacher_delta_effective_rank
principal_angle_to_teacher_delta_subspace
floor_gap_transition_improvement
floor_gap_delta_cosine
floor_gap_recall@1
floor_gap_delta_rank
residual_to_floor_norm_ratio
residual_cap_hit_fraction
rank_ladder_router_entropy
rank_ladder_usage_by_action
batch_probe_z_bio
batch_probe_z_tech
perturbation_probe_z_bio
cross_modal_RNA_to_image_recall@1
cross_modal_image_to_RNA_recall@1
```

### D3. Required logs per training step/eval

Log:

```text
loss/endpoint_latent_jepa
loss/delta_direction
loss/floor_distillation
loss/source_improvement_hinge
loss/spectral_entropy
loss/covariance_shape
loss/residual_norm_cap
loss/anti_collapse

operator/floor_delta_norm
operator/residual_delta_norm
operator/residual_to_floor_norm_ratio
operator/residual_cap_hit_fraction
operator/router_entropy
operator/effective_rank
operator/spectral_entropy
operator/floor_gap_improvement
operator/floor_gap_recall
operator/floor_gap_rank

identity/encoder_path_used
identity/pls_raw_linear_main_path_used
identity/condition_key_feature_present
identity/teacher_stop_gradient_verified
identity/separate_bio_and_tech_latents_present
identity/heldout_action_descriptor_valid
```

---

## Stage E: tests before experiments

Run these before any training experiment:

```bash
pytest \
  tests/test_biospectral_operator_contracts.py \
  tests/test_biospectral_jepa_identity.py \
  tests/test_biospectral_no_leakage.py \
  tests/test_biooperator_contracts.py \
  tests/test_biooperator_losses.py
```

Minimum tests to implement:

1. Full ridge floor reproduction.
2. Low-rank ridge floor reproduction.
3. Neural low-rank exact equivalence.
4. Zero residual preserves full ridge floor.
5. Residual branch initialized zero.
6. Residual cap triggers and logs cap-hit fraction.
7. Rank ladder includes a full-ridge/floor expert.
8. Router outputs valid probabilities.
9. No `condition_key` or `biological_key` can enter model inputs.
10. No test/eval target means used in fitting floor/basis/residual.
11. Teacher targets are stop-gradient.
12. Full BioSpectral-JEPA identity report passes.

If tests fail, stop with:

```text
PHASE6_STOP_TEST_FAILURE
```

---

## Experiment sequence

Run experiments in this exact order. Do not skip gates.

### BSJ000: rank bottleneck audit only

Purpose:

```text
Separate implementation gap from true rank bottleneck.
```

No architecture promotion possible.

Required output:

```text
rank_bottleneck_audit/RANK_BOTTLENECK_AUDIT.md
rank_bottleneck_audit/REOPENING_DECISION.md
```

Decision labels:

```text
PHASE6_REOPEN_BIOSPECTRAL_APPROVED
PHASE6_STOP_FLOOR_REPRODUCTION_FAILED
PHASE6_STOP_NEURAL_LOWRANK_EQUIVALENCE_FAILED
PHASE6_STOP_CONTROL_AFFINE_BUG_ONLY
PHASE6_STOP_NO_RANK_HEADROOM
```

### BSJ001: neural low-rank equivalence

Mode:

```text
frozen_neural_low_rank_equivalence
```

Goal:

```text
Match analytical action_low_rank_ridge exactly.
```

Keep gate:

```text
transition improvement within 1e-4 of 0.0046
delta cosine within 1e-4 of 0.3877
recall@1 within 1e-4 of 0.4074
delta rank within 1e-3 of 6.7681
```

If fail, stop. This is a code/factorization bug.

### BSJ002: full-ridge floor wrapper

Mode:

```text
frozen_full_ridge_floor_wrapper_zero_residual
```

Goal:

```text
Wrap the full ridge floor in the BioSpectral transition module without changing output.
```

Keep gate:

```text
transition improvement within 1e-4 of 0.0057
delta cosine within 1e-4 of 0.3980
recall@1 within 1e-4 of 0.4815
delta rank within 1e-3 of 10.2835
residual_to_floor_norm_ratio == 0.0 or <= 1e-6
```

If fail, stop. Floor preservation is broken.

### BSJ003: rank-ladder router, no trainable residual

Mode:

```text
frozen_rank_ladder_router
```

Goal:

```text
Show that adaptive rank selection can preserve or match the full floor when full-ridge expert is available.
```

Keep gate:

```text
transition improvement >= 0.0057 - 1e-4
delta cosine >= 0.3980 - 1e-4
recall@1 >= 0.4815 - 1e-4
delta rank >= 10.0
router entropy logged
rank usage by action logged
```

If it fails, stop. The router is degrading the floor.

### BSJ004: floor + spectral residual, frozen backbone

Mode:

```text
frozen_floor_plus_spectral_residual
```

Goal:

```text
Improve beyond full ridge without erasing the floor.
```

Keep gate:

```text
transition improvement >= 0.0075
delta cosine >= 0.3980
recall@1 >= 0.4815
delta rank >= 10.0
magnitude ratio between 0.50 and 1.50
residual_to_floor_norm_ratio <= 0.25
residual_cap_hit_fraction <= 0.25
floor_gap_transition_improvement >= +0.0015
no leakage flags
```

If improvement is positive but below `0.0075`, label:

```text
TIER1_NEARMISS_OPERATOR_ABOVE_FLOOR_WEAK_SIGNAL
```

Do not proceed to end-to-end JEPA unless the keep gate passes.

If below floor, label:

```text
TIER1_DISCARD_RESIDUAL_BELOW_FLOOR
```

Stop.

### BSJ005: optional kernel residual, frozen backbone

Run only if one of the following is true:

1. BSJ004 passes but residual analysis suggests nonlinear residual headroom.
2. BSJ004 is a near miss above floor and `RANK_BOTTLENECK_AUDIT.md` shows action/control neighborhoods explain residuals.

Mode:

```text
frozen_floor_plus_kernel_residual
```

Keep gate is the same as BSJ004, but require:

```text
kernel_residual_to_floor_norm_ratio <= 0.20
no train/eval target leakage
```

If BSJ005 beats BSJ004, retain BSJ005. Otherwise retain BSJ004 as the operator candidate.

### BSJ006: full BioSpectral-JEPA end-to-end

Run only if BSJ004 or BSJ005 passes.

Mode:

```text
end_to_end_biospectral_jepa
```

Goal:

```text
Train a real cross-modal, action-conditioned JEPA while preserving the operator floor.
```

Identity gates:

```text
encoder_path_used == 1.0
pls_raw_linear_main_path_used == 0.0
condition_key_feature_present == 0.0
teacher_stop_gradient_verified == 1.0
separate_bio_and_tech_latents_present == 1.0
heldout_action_descriptor_valid == 1.0
```

Transition gates:

```text
transition improvement >= operator_candidate_improvement - 0.001
delta cosine >= operator_candidate_delta_cosine - 0.02
recall@1 >= operator_candidate_recall@1 - 0.05
delta rank >= operator_candidate_delta_rank - 1.0
magnitude ratio between 0.50 and 1.50
```

Cross-modal gates on `synth_genetic_anchor_lite`:

```text
RNA->image recall@1 >= 0.1875 or improves over BSJ006 zero-step encoder baseline
image->RNA recall@1 > 0.0000
no collapse in target z_bio effective rank
```

Batch/technical gates:

```text
z_bio should not become more batch-predictive than z_tech when batch labels exist
batch allocation gap should be non-negative where meaningful
```

If encoders drift and destroy the operator floor, label:

```text
TIER1_DISCARD_END_TO_END_DESTROYS_OPERATOR_FLOOR
```

### BSJ007: Norman RNA-only diagnostic

Run only if BSJ006 passes synthetic gates.

Norman constraints:

```text
RNA-only diagnostic
A549-only context
batch ignored unless real batch/acquisition metadata is recovered
dose fixed as guide presence, not chemical concentration
action is gene or gene-pair descriptor
no exact condition-key one-hot
```

No Norman-only result can promote the model because Norman lacks imaging and exposed batch/acquisition metadata in the processed h5ad.

Keep gate:

```text
transition source cosine improvement > 0
transition-to-target recall@1 improves over source-as-target null
target z_bio effective rank does not collapse
no condition-key leakage
```

---

## Tier 2 / Tier 3 policy

Phase 6 Tier 1 can only create a candidate. It cannot promote.

If BSJ006 passes Tier 1 cleanly, write a separate `TIER2_PLAN.md` with:

```text
seeds = [0, 1, 2]
dataset = synth_genetic_anchor_lite
eval_split = test_heldout_perturbation
candidate = best of BSJ004/BSJ005 operator + BSJ006 full JEPA wrapper
fixed gates copied from this prompt
```

Tier 2 pass requires:

```text
mean transition improvement above full ridge floor by >= 0.0015
no seed has negative transition improvement
mean delta cosine >= 0.3980
mean recall@1 >= 0.4815
mean delta rank >= 10.0
cross-modal retrieval non-collapsed
identity gates pass for every seed
no leakage flags
```

Tier 3 can only be proposed after Tier 2 passes. Tier 3 must include a no-regression / generalization split, for example:

```text
synth_batch_confound_lite or new genetic-anchor variant
Norman RNA-only diagnostic as external biological reference
future real paired imaging + scRNA dataset if available
```

Do not promote from synthetic single-seed results.

---

## Leakage rules

Hard forbidden:

```text
condition_key as model input
biological_key as model input
exact target-key one-hot features
test/eval target means in training or floor fitting
pooled train+test target statistics
batch id as shortcut for biological transition
raw-linear PLS as main representation path
```

Allowed:

```text
condition_key as label for grouping/evaluation only
batch id for diagnostics only, or z_tech branch when explicitly testing technical separation
train-only ridge floor fitted from train latents
train-only spectral basis fitted from train deltas
Norman action descriptor as gene multi-hot / gene-pair multi-hot
```

Every script must write a leakage report:

```text
leakage_report.md
```

with:

```text
train rows used for fitting
validation/eval rows used for scoring only
feature names used by action builder
confirmation that condition_key/biological_key are absent from tensors
confirmation that test/eval target means are not used
```

---

## Decision labels

Use exact labels only:

```text
PHASE6_ABORT_FLOOR_REPRODUCTION_FAILED
PHASE6_STOP_TEST_FAILURE
PHASE6_STOP_NEURAL_LOWRANK_EQUIVALENCE_FAILED
PHASE6_STOP_CONTROL_AFFINE_BUG_ONLY
PHASE6_STOP_NO_RANK_HEADROOM
PHASE6_REOPEN_BIOSPECTRAL_APPROVED
BSJ001_KEEP_NEURAL_LOWRANK_MATCHES_ANALYTIC
BSJ001_DISCARD_NEURAL_LOWRANK_MISMATCH
BSJ002_KEEP_FLOOR_WRAPPER_MATCHES_FULL_RIDGE
BSJ002_DISCARD_FLOOR_WRAPPER_DRIFT
BSJ003_KEEP_RANK_LADDER_PRESERVES_FLOOR
BSJ003_DISCARD_ROUTER_DEGRADES_FLOOR
BSJ004_KEEP_SPECTRAL_RESIDUAL_ABOVE_FLOOR
BSJ004_NEARMISS_OPERATOR_ABOVE_FLOOR_WEAK_SIGNAL
BSJ004_DISCARD_RESIDUAL_BELOW_FLOOR
BSJ005_KEEP_KERNEL_RESIDUAL_ABOVE_FLOOR
BSJ005_DISCARD_KERNEL_RESIDUAL_BELOW_FLOOR
BSJ006_KEEP_REAL_JEPA_OPERATOR_PRESERVED
BSJ006_DISCARD_END_TO_END_DESTROYS_OPERATOR_FLOOR
BSJ007_NORMAN_RNA_ONLY_DIAGNOSTIC_KEEP
BSJ007_NORMAN_RNA_ONLY_DIAGNOSTIC_DISCARD
SEARCH_CLOSED_NO_NEW_BASELINE
```

---

## Stop conditions

Stop immediately and write `final_report.md` if any of these occur:

1. Full ridge floor cannot be reproduced.
2. Low-rank ridge floor cannot be reproduced.
3. Neural low-rank equivalence fails.
4. Floor-preserving wrapper changes the full-ridge output at initialization.
5. Rank ladder degrades the full-ridge floor despite full-ridge expert being available.
6. Spectral/kernel residual falls below the full-ridge floor.
7. Residual cap-hit fraction exceeds `0.25` and metrics do not improve.
8. Predicted delta effective rank collapses below `8.0` in a candidate meant to beat full ridge.
9. Any leakage report flags forbidden inputs/statistics.
10. Full JEPA identity checks fail.
11. End-to-end encoder training destroys the operator floor.
12. Two consecutive candidates fall below floor after Stage B.

When a stop condition fires:

1. Finish the current eval.
2. Write the journal entry.
3. Update `results.tsv`.
4. Update `family_allocation.md`.
5. Write `final_report.md`.
6. Stop the loop.

Do not launch another experiment after a stop condition.

---

## Final report template

At closure, write:

```markdown
# BioSpectral-JEPA Phase 6 Final Report

## Decision label
<exact label>

## Model of record
Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless Tier 3 pass supersedes it.

## What was tested
- Stage A floor reproduction: <pass/fail>
- Stage B rank bottleneck audit: <summary>
- BSJ001 neural low-rank equivalence: <summary>
- BSJ002 floor wrapper: <summary>
- BSJ003 rank ladder: <summary>
- BSJ004 spectral residual: <summary>
- BSJ005 kernel residual: <summary or not run>
- BSJ006 full BioSpectral-JEPA: <summary or not run>
- BSJ007 Norman RNA-only: <summary or not run>

## Key metrics
<table>

## Main interpretation
Was BOJ002 primarily a bug, a rank bottleneck, or both?

## JEPA identity status
Did any full candidate satisfy real-JEPA identity?

## Leakage status
Any forbidden feature/statistic usage?

## Recommendation
Continue / close / run Tier 2 plan / redesign data.
```

---

## The main idea to preserve

The current project already has real JEPA machinery. The next scientific contribution should not be another arbitrary neural head. It should be:

```text
A rank-preserving action-conditioned JEPA transition operator that keeps a train-only analytical floor, learns only controlled spectral residuals above that floor, and preserves cross-modal biological latent prediction without leaking condition keys or test targets.
```

That is the Phase 6 novelty.
