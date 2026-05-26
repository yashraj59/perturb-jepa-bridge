# Phase 5 Codex Amendment: BioOperator-JEPA Contract-First Transition Learning

## Direct instruction

Read and apply this amendment verbatim.

Do **not** continue BFJ001 by training longer. Do **not** launch BFJ002-BFJ006. Phase 4 ended correctly: the BioFlow-JEPA vector-field candidate failed Tier 1 with negative transition signal.

Start Phase 5 in:

```text
outputs/autoresearch_biooperator_jepa_phase5/
```

The active protected model of record remains:

```text
rank-3 train-split-only PLS raw-linear readout
```

No BioAction-JEPA, BioTech-JEPA, BioMechanistic-JEPA, or BioFlow-JEPA candidate is promoted. PLS is audit-only and must not become the main BioOperator-JEPA representation path.

## Why this amendment exists

Phase 4 produced two facts that must be reconciled before any further architecture search:

1. The delta operator audit showed measurable teacher deltas and a positive simple frozen-latent floor.
2. BFJ001 failed below that floor and anti-aligned with the target delta.

This means the next scientific question is **not** “can a bigger transition module work?” The next question is:

> Can a neural transition operator exactly reproduce, then carefully exceed, the train-only frozen-latent action-ridge floor without leakage, sign errors, or loss/optimization mismatch?

If the answer is no, the correct output is a debugging/audit closure report, not another JEPA architecture.

## Required reference facts from Phase 4

Read and preserve these facts from:

```text
outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md
outputs/autoresearch_bioflow_jepa_phase4/final_report.md
```

Minimum facts to carry into `BASELINE_REGISTRY.md`:

```text
Dataset: synth_genetic_anchor_lite
Eval split: test_heldout_perturbation
Latent cache model: BTJ001 BioTech-JEPA checkpoint
Protected model of record: rank-3 train-split-only PLS raw-linear readout
Forbidden shortcuts checked: no condition_key, no biological-key one-hot, no test target means, no pooled train+test targets
```

Teacher delta statistics:

```text
train delta_effective_rank = 13.5627
train delta_mean_norm = 0.4310
train delta_std_mean = 0.0846
train source_to_target_cosine_mean = 0.8977

eval delta_effective_rank = 11.7819
eval delta_mean_norm = 0.4252
eval delta_std_mean = 0.0832
eval source_to_target_cosine_mean = 0.9031
```

Frozen-latent simple floors:

```text
eval action_ridge_delta:
  transition_source_cosine_improvement = +0.0057
  transition_to_target_recall@1 = 0.4815
  transition_to_target_median_rank = 2.0000
  delta_cosine = 0.3980
  delta_magnitude_ratio = 0.7744
  delta_prediction_effective_rank = 10.2835

eval action_low_rank_ridge:
  transition_source_cosine_improvement = +0.0046
  transition_to_target_recall@1 = 0.4074
  transition_to_target_median_rank = 2.0000
  delta_cosine = 0.3877
  delta_magnitude_ratio = 0.7585
  delta_prediction_effective_rank = 6.7681

train action_ridge_delta:
  transition_source_cosine_improvement = +0.0769
  transition_to_target_recall@1 = 0.7778
  delta_cosine = 0.8483
  delta_magnitude_ratio = 0.8535
  delta_prediction_effective_rank = 11.6312
```

Frozen-latent optimization audit:

```text
raw_delta_mse, 20 steps:
  transition_source_cosine_improvement = +0.0403
  delta_cosine = 0.6699
  delta_magnitude_ratio = 1.1511
  delta_prediction_effective_rank = 9.7235

endpoint_cosine, 20 steps:
  transition_source_cosine_improvement = +0.0607
  delta_cosine = 0.1229
  delta_magnitude_ratio = 16.3994
  delta_prediction_effective_rank = 8.3893

whitened_delta_mse, 20 steps:
  transition_source_cosine_improvement = +0.0386
  delta_cosine = 0.6032
  delta_magnitude_ratio = 0.8577
  delta_prediction_effective_rank = 9.6204

whitened_hinge_direction, 20 steps:
  transition_source_cosine_improvement = -0.7517
  delta_magnitude_ratio = 17.7108
  source_improvement_hinge_violation_fraction = 1.0000
```

BFJ001 failed with:

```text
transition_source_cosine_improvement = -0.0104
delta_cosine = -0.1054
delta_prediction_effective_rank = 7.6852
source_improvement_hinge_violation_fraction = 0.7000
Decision label = BFJ_TIER1_DISCARD_NO_SIGNAL
```

Interpretation:

```text
The delta target has signal.
A linear train-only action-ridge baseline has small but real positive eval signal.
The neural vector-field candidate failed below this floor.
Therefore Phase 5 must become a contract-first operator-learning phase.
```

## Literature ideas to consult and record

Write these into `papers_consulted.md` with the specific mechanism extracted and whether it is used, rejected, or deferred.

Use the papers as conceptual guidance only. Do not overfit implementation to paper names.

### JEPA and action-conditioned world models

- V-JEPA 2 / V-JEPA 2-AC: self-supervised latent prediction and action-conditioned latent world modeling.
  - Mechanism to extract: latent prediction target, stop-gradient teacher target, action-conditioned transition in latent space.
  - Mapping here: `control z_bio + genetic action -> perturbed teacher z_bio`.
  - URL: https://arxiv.org/abs/2506.09985

### Flow matching and vector-field learning

- Flow Matching for Generative Modeling.
  - Mechanism to extract: learn vector fields against known training trajectories instead of reconstructing observations.
  - Mapping here: only use vector fields after the simpler frozen-latent operator contracts pass.
  - URL: https://arxiv.org/abs/2210.02747

- Efficient Flow Matching using Latent Variables.
  - Mechanism to extract: latent-variable structure can make transport/vector-field learning more efficient.
  - Mapping here: defer until BioOperator passes action-ridge floor; do not launch a new flow first.
  - URL: https://arxiv.org/html/2505.04486v2

### Controlled Koopman / linearized latent dynamics

- Koopman/control-inspired latent dynamics.
  - Mechanism to extract: a nonlinear system may be easier to predict if the learned latent evolves under a low-rank or structured action-conditioned linear operator.
  - Mapping here: use a control-affine low-rank operator as the first learnable transition head, not an unconstrained MLP/vector field.
  - Example reference: https://openreview.net/forum?id=fkrYDQaHOJ

### Single-cell perturbation transport

- CellOT.
  - Mechanism to extract: control-to-perturbed maps should preserve population structure, not only mean embeddings.
  - Mapping here: population/prototype transport is deferred until the single-condition latent operator contract passes.
  - URL: https://www.nature.com/articles/s41592-023-01969-x

### Schrödinger bridge / stochastic transport for single-cell perturbations

- Distributional Transport for Single-Cell Perturbation Prediction.
  - Mechanism to extract: stochastic dynamic mappings can model unpaired control and perturbed populations.
  - Mapping here: only consider stochastic population JEPA after deterministic operator floor is passed.
  - URL: https://arxiv.org/html/2511.13124v1

## Non-negotiable identity constraints

A Phase 5 candidate is not acceptable unless all are true:

```text
encoder_path_used = 1.0
pls_raw_linear_main_path_used = 0.0
condition_key_feature_present = 0.0
biological_key_one_hot_present = 0.0
test_target_mean_used_for_fit = 0.0
pooled_train_test_target_used_for_fit = 0.0
teacher_stop_gradient_verified = 1.0
EMA target encoders present if encoders are trained
separate z_bio and z_tech retained when full JEPA is used
```

For frozen-latent operator floors, EMA encoders are not trained, but the cached latents must come from the previously valid JEPA teacher/target path and the report must say:

```text
frozen_latent_operator_only = 1.0
encoder_training_skipped = 1.0
```

## Phase 5 thesis

Build **BioOperator-JEPA**:

```text
BioOperator-JEPA =
  real cross-modal action-conditioned JEPA backbone
  + train-only frozen-latent transition floor
  + ridge-calibrated action operator
  + low-rank control-affine transition head
  + strict sign/gradient/operator contracts
```

The architectural novelty is not another unconstrained delta MLP. It is a JEPA world model whose perturbation transition head is forced to pass a reproducible analytical operator floor before being allowed to become nonlinear or end-to-end.

Core transition form:

```text
z_pred = z_control + Δz

Δz = b(a) + A(a) * whiten(z_control)

A(a) = sum_r alpha_r(a) * U_r V_r^T
```

where:

```text
z_control = biological latent of control condition
z_target = biological teacher latent of perturbed condition
a = perturbation action descriptor, not condition key
delta_teacher = z_target - z_control
b(a) = action-specific additive shift
A(a) = action-conditioned low-rank control operator
```

Do not implement stochastic/prototype population transport until this deterministic operator passes the frozen-latent floor.

## Required files

Create and maintain:

```text
outputs/autoresearch_biooperator_jepa_phase5/research_journal.md
outputs/autoresearch_biooperator_jepa_phase5/results.tsv
outputs/autoresearch_biooperator_jepa_phase5/BASELINE_REGISTRY.md
outputs/autoresearch_biooperator_jepa_phase5/papers_consulted.md
outputs/autoresearch_biooperator_jepa_phase5/operator_contract_audit.md
outputs/autoresearch_biooperator_jepa_phase5/architectural_changes_log.md
outputs/autoresearch_biooperator_jepa_phase5/identity_violations_considered.md
outputs/autoresearch_biooperator_jepa_phase5/final_report.md
```

`results.tsv` must include at least:

```text
commit	experiment_num	stage	family	tier_reached	decision_label	status	dataset	eval_split	seed_list	primary_metric	secondary_metric	protected_metric_summary	architectural_change	description
```

Add these Phase 5-specific columns when available:

```text
operator_train_transition_improvement
operator_eval_transition_improvement
operator_train_delta_cosine
operator_eval_delta_cosine
operator_eval_recall_at_1
operator_eval_median_rank
operator_predicted_delta_rank
action_ridge_floor_gap
sign_contract_pass
ridge_equivalence_pass
source_improvement_hinge_violation_fraction
```

## Stage A: reproduce the frozen-latent operator floor

Before writing new model architecture, implement or patch a script:

```text
scripts/run_biooperator_contract_audit.py
```

It must:

1. Load or recompute the same frozen latent cache used in Phase 4.
2. Reproduce the Phase 4 action-ridge and low-rank ridge baselines.
3. Save predictions for every baseline.
4. Compute all transition metrics from a single common function.
5. Write `operator_contract_audit.md`.

Required metrics:

```text
transition_source_cosine_improvement
absolute_target_cosine
delta_cosine
delta_magnitude_ratio
delta_prediction_effective_rank
transition_to_target_recall@1
transition_to_target_median_rank
source_improvement_hinge_violation_fraction
```

Required Stage A pass gate:

```text
abs(reproduced_eval_action_ridge_transition_improvement - 0.0057) <= 0.0020
reproduced_eval_action_ridge_delta_cosine >= 0.35
reproduced_eval_action_ridge_rank >= 8.0
no forbidden shortcut detected
```

If Stage A fails, stop immediately with:

```text
PHASE5_STOP_OPERATOR_FLOOR_NOT_REPRODUCED
```

Do not implement BioOperator-JEPA if the analytical floor cannot be reproduced.

## Stage B: sign, gradient, and loss contract tests

Add tests before training any new neural transition model.

Recommended files:

```text
tests/test_biooperator_contracts.py
tests/test_biooperator_losses.py
```

Minimum tests:

### B1. Delta sign contract

Construct toy latents:

```text
z_source = [1, 0]
z_target = [0, 1]
delta_teacher = z_target - z_source
```

A predictor initialized to `delta_teacher` must improve source-to-target cosine.

A predictor initialized to `-delta_teacher` must degrade source-to-target cosine.

The test must fail if source and target are accidentally swapped.

### B2. One-step gradient contract

For raw delta MSE and whitened delta MSE, one optimizer step from small random initialization must increase mean target cosine on a toy batch.

### B3. Hinge sign contract

The source-improvement hinge must penalize predictions that are worse than source-as-target and must not reward exploding magnitude. The Phase 4 `whitened_hinge_direction` result was catastrophic; this test must catch that failure pattern.

### B4. Ridge equivalence contract

Train a tiny neural linear head on frozen train latents with the same input features as `action_ridge_delta`. It must match the ridge floor within tolerance on train and not anti-align on eval.

### B5. Magnitude contract

Any operator prediction with `delta_magnitude_ratio > 2.0` must be flagged unless a specific experiment declares a stochastic/prototype transport exception. The Phase 4 endpoint-cosine audit had magnitude ratio `16.3994`; do not let that become a silent “success.”

Stage B pass gate:

```text
all focused tests pass
sign_contract_pass = 1.0
gradient_contract_pass = 1.0
hinge_contract_pass = 1.0
ridge_equivalence_contract_pass = 1.0
```

If Stage B fails, stop with:

```text
PHASE5_STOP_OPERATOR_CONTRACT_FAILURE
```

## Stage C: implement the smallest neural operator that can match ridge

Only after Stages A and B pass, implement:

```text
perturb_jepa/models/biooperator_jepa.py
perturb_jepa/training/biooperator_losses.py
perturb_jepa/evaluation/biooperator_metrics.py
scripts/train_biooperator_jepa.py
scripts/evaluate_biooperator_jepa.py
```

### Required modules

#### `DeltaWhitening`

Train-split-only whitening transform for delta targets:

```text
fit on train delta_teacher only
apply to train/eval prediction and target deltas
save mean/std/eigen cutoff
never fit on eval/test
```

#### `ActionFeatureBuilder`

Builds action features without condition-key leakage:

```text
synthetic: perturbation descriptor / legal action descriptor only
Norman: gene multi-hot or gene-set action descriptor only
chemical screens: compound descriptor + concentration only when concentration is real
never condition_key
never biological_key
never exact target-key one-hot
```

#### `RidgeTransitionFloor`

A non-neural baseline module or utility that can be called by training/evaluation scripts. It should be able to produce the exact Stage A predictions.

Purpose:

```text
serves as the floor
serves as initialization target
serves as distillation teacher
serves as regression-test fixture
```

#### `NeuralActionRidgeHead`

A neural head constrained to be ridge-equivalent:

```text
input: action features
output: delta in z_bio space
architecture: single Linear or low-rank Linear
initialization: zeros or closed-form ridge weights if shape permits
loss: raw/whitened delta MSE + delta cosine
```

This is not the final architecture. It is the required bridge from analytical floor to neural operator.

#### `LowRankControlAffineOperator`

Main BioOperator transition head:

```text
input: z_control_bio, action_features
output: delta_z_bio

components:
  additive_action_shift b(a)
  action-conditioned low-rank linear map A(a) z_control
  optional small residual MLP initialized to zero

constraints:
  low-rank factors spectral-normalized or norm-clipped
  residual contribution ratio logged
  zero-action/control action produces near-zero delta
  no batch id as transition shortcut
```

Suggested formula:

```python
z = whiten_state(z_control_bio)
a = action_encoder(action_features)
shift = shift_head(a)
coeff = softmax_or_tanh_coefficients(a)
linear_delta = sum_r coeff[r] * (z @ V[r]) @ U[r].T
residual_delta = residual_scale * residual_mlp(concat(z, a))
delta = unwhiten_delta(shift + linear_delta + residual_delta)
z_pred = normalize_or_layernorm(z_control_bio + delta)
```

#### `BioOperatorJEPA`

Full candidate wrapper that keeps BioTech-JEPA identity but replaces the failed transition head.

Required outputs:

```text
z_control_bio
z_target_bio_teacher
predicted_delta_bio
predicted_target_bio
ridge_floor_delta_optional
transition_identity_report
```

### Required losses

Use a conservative weighted mixture:

```text
L_operator =
  1.0 * whitened_delta_mse
+ 0.5 * delta_direction_cosine_loss
+ 0.5 * endpoint_latent_jepa_cosine_loss
+ 0.2 * ridge_floor_distillation_loss during warm start
+ 0.1 * source_improvement_hinge, capped and sign-tested
+ 0.05 * delta_rank_variance_floor
```

Do **not** let endpoint cosine alone dominate. Phase 4 showed endpoint cosine can improve while delta magnitude explodes.

Log unweighted and weighted losses separately:

```text
unweighted_whitened_delta_mse
weighted_whitened_delta_mse
unweighted_delta_direction
weighted_delta_direction
unweighted_endpoint_jepa
weighted_endpoint_jepa
unweighted_ridge_distillation
weighted_ridge_distillation
unweighted_source_hinge
weighted_source_hinge
weighted_operator_to_main_ratio
```

## Stage D: experiment sequence

Run these in order. Do not skip ahead.

### BOJ000: Contract audit reproduction

```text
family = diagnostics
stage = StageA
candidate = no model training
```

Pass gate: Stage A gate above.

Decision labels:

```text
PHASE5_OPERATOR_FLOOR_REPRODUCED
PHASE5_STOP_OPERATOR_FLOOR_NOT_REPRODUCED
```

### BOJ001: Frozen-latent neural ridge equivalence

```text
family = neural_operator_floor
stage = StageC
candidate = NeuralActionRidgeHead only
encoders = frozen / not used
```

Goal: prove neural training can match the analytical ridge floor.

Pass gate:

```text
train_transition_source_cosine_improvement >= +0.0600
train_delta_cosine >= 0.75
eval_transition_source_cosine_improvement >= +0.0035
eval_delta_cosine >= 0.30
eval_delta_prediction_effective_rank >= 8.0
eval_delta_magnitude_ratio between 0.4 and 1.6
no anti-alignment
```

If BOJ001 fails, stop with:

```text
PHASE5_STOP_NEURAL_OPERATOR_BELOW_RIDGE_FLOOR
```

### BOJ002: Frozen-latent low-rank control-affine operator

```text
family = control_affine_operator
candidate = LowRankControlAffineOperator
encoders = frozen / not used
```

Goal: beat or match action ridge with a biologically plausible control-affine operator.

Pass gate:

```text
eval_transition_source_cosine_improvement >= max(+0.0057, BOJ001_eval_improvement - 0.0010)
eval_delta_cosine >= BOJ001_eval_delta_cosine - 0.05
eval_transition_to_target_recall@1 >= 0.45
eval_delta_prediction_effective_rank >= 8.0
source_improvement_hinge_violation_fraction <= 0.60
residual_contribution_ratio <= 0.30 unless explicitly justified
```

If BOJ002 is worse than BOJ001, retain BOJ001 and stop. Do not hide a controlled operator failure by moving end-to-end.

Decision labels:

```text
TIER1_KEEP_OPERATOR_MATCHES_FLOOR
TIER1_DISCARD_OPERATOR_BELOW_FLOOR
```

### BOJ003: BioOperator-JEPA with frozen encoders

```text
family = jepa_backbone_operator
candidate = BioOperator-JEPA transition head attached to frozen BioTech-JEPA encoders
```

Goal: verify the full model wiring does not destroy the operator floor.

Pass gate:

```text
eval_transition_source_cosine_improvement >= BOJ002_eval_improvement - 0.0010
eval_delta_cosine >= BOJ002_eval_delta_cosine - 0.05
encoder_path_used = 1.0
pls_raw_linear_main_path_used = 0.0
condition_key_feature_present = 0.0
teacher_stop_gradient_verified = 1.0
```

If BOJ003 fails, stop and write exactly where the floor was lost:

```text
operator standalone -> model wrapper
loss weighting
normalization
source/target batch construction
teacher target mismatch
```

### BOJ004: BioOperator-JEPA end-to-end JEPA fine-tune

Only run if BOJ003 passes.

```text
family = end_to_end_biooperator_jepa
candidate = encoders + z_bio/z_tech + operator transition
```

Use a small learning rate for encoders and a larger one for operator head. Keep ridge distillation for the first warmup segment, then anneal.

Pass gate:

```text
eval_transition_source_cosine_improvement >= +0.0100
eval_delta_cosine >= 0.35
eval_transition_to_target_recall@1 >= 0.50
eval_delta_prediction_effective_rank >= 8.0
batch_allocation_gap remains non-negative if batch labels available
image_to_rna_recall@1 does not collapse below previous healthy value when imaging is present
no magnitude explosion: delta_magnitude_ratio <= 1.8
no forbidden shortcuts
```

Decision labels:

```text
TIER1_KEEP_CONTROLLED_SIGNAL
TIER1_DISCARD_NO_SIGNAL
TIER1_DISCARD_BELOW_OPERATOR_FLOOR
TIER1_DISCARD_MAGNITUDE_EXPLOSION
TIER1_DISCARD_IDENTITY_VIOLATION
```

### BOJ005: Norman RNA-only diagnostic

Only run if BOJ004 passes on synthetic.

For Norman:

```text
batch = ignored unless real batch/acquisition metadata is recovered
dose = fixed guide presence, not chemical concentration
action = gene multi-hot / gene-pair descriptor
modality = RNA-only
```

Goal: diagnostic only. Norman cannot validate imaging or batch disentanglement from the current processed h5ad.

Report:

```text
transition_source_cosine_improvement
transition_to_target_recall@1
transition_to_target_median_rank
target z_bio effective rank
delta magnitude ratio
action descriptor coverage
exact condition-key leakage check
```

Do not promote from Norman-only diagnostic.

## Hard stop conditions

Stop immediately and write `final_report.md` if any occurs:

```text
Stage A cannot reproduce action-ridge floor.
Any sign/gradient/hinge contract fails.
BOJ001 learned neural floor underperforms action_ridge_delta by more than tolerance.
Any candidate has negative eval transition_source_cosine_improvement.
Any candidate has negative eval delta_cosine.
Any candidate has delta_magnitude_ratio > 2.0 without declared stochastic transport exception.
Any candidate uses condition_key, biological_key one-hot, test target means, or pooled train+test targets.
Any candidate promotes PLS/raw-linear readout as the JEPA path.
Any candidate fails teacher stop-gradient identity checks.
```

## Promotion rule

No Phase 5 Tier 1 run can promote the model. Phase 5 can only produce one of:

```text
PHASE5_OPERATOR_CONTRACT_FAILURE_CLOSE
PHASE5_OPERATOR_FLOOR_REPRODUCED_BUT_NEURAL_FAILURE_CLOSE
PHASE5_OPERATOR_MATCHES_FLOOR_READY_FOR_TIER2
PHASE5_BIOOPERATOR_TIER1_KEEP_READY_FOR_TIER2
SEARCH_CLOSED_NO_NEW_BASELINE
```

The protected rank-3 PLS raw-linear readout remains model of record unless a future Tier 3 pass supersedes it.

## Final report template

At closure, write:

```markdown
# BioOperator-JEPA Phase 5 Final Report

## Decision label

<exact label>

## Model of record

Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass explicitly supersedes it.

## What was tested

- Stage A operator floor reproduction
- Stage B sign/gradient/loss contracts
- BOJ001 neural ridge-equivalence result
- BOJ002 low-rank control-affine result, if run
- BOJ003 frozen-backbone integration result, if run
- BOJ004 end-to-end JEPA result, if run
- BOJ005 Norman diagnostic, if run

## Key metrics

| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | decision |
|---|---:|---:|---:|---:|---:|---|

## Floor comparison

Compare every neural candidate to:

```text
eval action_ridge_delta improvement = +0.0057
eval action_ridge_delta delta_cosine = 0.3980
eval action_ridge_delta rank = 10.2835
```

## What failed or passed

State whether failure occurred in:

```text
latent cache
metric reproduction
sign convention
loss gradient
ridge neuralization
control-affine operator
full JEPA wrapper
end-to-end encoder training
```

## Recommendation

One of:

```text
Close architecture search and debug data/metrics.
Retain frozen operator floor only as audit reference.
Proceed to Tier 2 multi-seed BioOperator-JEPA.
Redesign dataset/split because current low-compute synthetic has too little transition signal.
```
```

## Minimal launch command suggestions

Use exact scripts after implementation; adapt only paths/devices as needed.

```bash
python scripts/run_biooperator_contract_audit.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --latent-cache outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit \
  --output-dir outputs/autoresearch_biooperator_jepa_phase5/contract_audit \
  --device cpu
```

```bash
python scripts/train_biooperator_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --mode frozen_neural_ridge \
  --seed 0 \
  --steps 100 \
  --device cuda \
  --output-dir outputs/autoresearch_biooperator_jepa_phase5/experiments/BOJ001_frozen_neural_ridge_seed0
```

```bash
python scripts/train_biooperator_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --mode frozen_control_affine \
  --seed 0 \
  --steps 100 \
  --operator-rank 8 \
  --device cuda \
  --output-dir outputs/autoresearch_biooperator_jepa_phase5/experiments/BOJ002_frozen_control_affine_seed0
```

```bash
python scripts/train_biooperator_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --mode end_to_end_jepa \
  --seed 0 \
  --steps 200 \
  --operator-rank 8 \
  --encoder-lr 1e-4 \
  --operator-lr 1e-3 \
  --device cuda \
  --output-dir outputs/autoresearch_biooperator_jepa_phase5/experiments/BOJ004_end_to_end_seed0
```
