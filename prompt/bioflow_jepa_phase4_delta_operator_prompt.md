# Phase 4 Amendment: BioFlow-JEPA / Controlled Vector-Field JEPA

## Direct instruction

You are continuing the JEPA research loop after Phase 3. Read this prompt completely and apply it verbatim.

Do **not** continue Phase 3 by training BMJ001 longer, increasing model size, adding action-token complexity, adding population prototypes, or launching BMJ002-BMJ006. Phase 3 stopped correctly. The failure is now specifically the **transition operator optimization**, not JEPA identity, not data loading, and not image-branch health.

Your next candidate is **BioFlow-JEPA**:

> A real cross-modal, action-conditioned JEPA biological world model that replaces the failed one-shot delta MLP with a controlled latent vector field / transport operator in `z_bio` space.

The goal is not to build a generative diffusion model. The goal is to make the JEPA transition path actually learn the informative teacher delta that Phase 3 proved exists.

---

## Current evidence you must preserve

### Model of record

The protected model of record remains:

```text
rank-3 train-split-only PLS raw-linear readout
```

It is a protected baseline and audit reference. It is **not** a JEPA model and must not be used as the main BioFlow-JEPA representation path.

### Phase 3 closure facts

Phase 3 ended with:

```text
Decision label: SEARCH_CLOSED_NO_NEW_BASELINE
No Phase 3 candidate promoted.
BMJ001 decision: TIER1_DISCARD_NO_SIGNAL
```

BMJ001 remained a real JEPA identity candidate:

```text
encoder_path_used = 1.0
pls_raw_linear_main_path_used = 0.0
condition_key_feature_present = 0.0
teacher_stop_gradient_verified = 1.0
separate_z_bio_z_tech_latents_present = 1.0
held_out_action_descriptor_valid = 1.0
```

Stage A diagnostics passed:

```text
image branch: IMAGE_BRANCH_AUDIT_HEALTHY
transition target: DELTA_TARGET_HAS_HEADROOM
action descriptor: ACTION_DESCRIPTOR_VALID
overall: PHASE3_DIAGNOSTICS_COMPLETE_PROCEED
```

The transition target audit found real target headroom:

```text
held-out delta teacher effective rank = 10.1424
held-out delta teacher std mean = 0.0819
held-out delta target NN recall@1 = 1.0000
held-out delta target batch-probe accuracy = 0.3125 vs majority 0.4375
```

BMJ001 failed because the learned transition operator degraded the source state:

```text
transition_source_cosine_improvement = -0.1695
transition_to_target_recall@1 = 0.0968
delta_cosine = -0.0332
absolute_target_cosine = 0.6891
delta_teacher_effective_rank = 12.4196
delta_prediction_effective_rank = 4.7813
target_z_bio_effective_rank = 8.1391
image_to_RNA_recall@1 = 0.1290
RNA_to_image_recall@1 = 0.0968
batch_allocation_gap = 0.0645
```

The Phase 3 final recommendation was:

```text
Close Phase 3 as currently specified.
The next amendment should target the delta predictor optimization itself before adding action-token or population complexity.
The failure pattern is: delta targets are informative, but the first learned delta operator anti-aligns with the teacher delta and degrades target prediction.
```

### Phase 2 facts still active

BioTech-JEPA gave useful factorization evidence, but not a promotion:

```text
BTJ001 synthetic genetic anchor:
RNA->image recall@1 = 0.1875
image->RNA recall@1 = 0.0000
transition-to-target recall@1 = 0.4375
transition source cosine improvement = +0.0161
joint z_bio batch-probe accuracy = 0.1875
joint z_tech batch-probe accuracy = 0.4375
batch allocation gap = +0.2500
joint z_bio effective rank = 7.5103
```

Norman remains useful only as an RNA-only genetic perturbation smoke/diagnostic dataset:

```text
cells = 91,205
genes = 5,045
conditions = 284
cell type = A549 only
no exposed batch/acquisition metadata
chemical dose ignored; guide presence fixed
```

Do not claim Norman validates imaging or batch disentanglement.

---

## Literature ideas to use

Record these in `papers_consulted.md` with one row per source: citation/link, field, technique extracted, how it maps to BioFlow-JEPA, and whether the experiment used it.

### JEPA / predictive representation learning

Use JEPA as the model identity. Prediction must happen in latent representation space using online/context encoders and stop-gradient EMA target encoders.

Relevant sources:

- I-JEPA / Image-based Joint-Embedding Predictive Architecture, CVPR 2023: predict latent target-block representations from context blocks, not pixels.
  - URL: https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf
- V-JEPA, 2024: feature-prediction-first video JEPA without reconstruction as the main objective.
  - URL: https://arxiv.org/abs/2404.08471
- V-JEPA 2, 2025: action-conditioned latent world model direction.
  - URL: https://arxiv.org/abs/2506.09985

Mapping:

```text
cell state = latent biological state z_bio
action = perturbation / guide / gene pair / drug descriptor
future state = perturbed target z_bio from EMA teacher
```

### Flow matching and rectified flow

Phase 3's one-shot delta predictor anti-aligned with the teacher delta. Replace that endpoint-only MLP with a vector-field objective that learns the local velocity from source to target in latent space.

Relevant sources:

- Flow Matching for Generative Modeling, 2022.
  - URL: https://arxiv.org/abs/2210.02747
- Conditional Flow Matching / OT-CFM, 2023.
  - URL: https://arxiv.org/abs/2302.00482
- Rectified Flow, 2022/2023.
  - URL: https://arxiv.org/abs/2209.03003

Mapping:

```text
source distribution = control z_bio population
target distribution = perturbed z_bio population
action-conditioned velocity = perturbation-conditioned biological response field
endpoint rollout = predicted perturbed z_bio
```

Do not implement a full generative model. Implement the smallest controlled latent vector field needed to fix the delta operator.

### Koopman / control / latent dynamics

Use a low-rank action-conditioned operator as a stabilizing inductive bias and baseline for the vector field.

Relevant sources:

- Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition, NeurIPS 2017.
  - URL: https://arxiv.org/abs/1710.04340
- Deep learning for universal linear embeddings of nonlinear dynamics, Nature Communications 2018.
  - URL: https://www.nature.com/articles/s41467-018-07210-0
- Koopman Operators in Robot Learning, 2025 survey.
  - URL: https://arxiv.org/html/2408.04200v2

Mapping:

```text
z_bio is the learned latent coordinate
action-conditioned operator K(a) moves z_bio_control toward z_bio_perturbed
low-rank bilinear form gives a stable first candidate before expressive MLP dynamics
```

### Single-cell perturbation modeling

Use these as biological constraints and baseline comparators, not as excuses to abandon JEPA.

Relevant sources:

- CellOT / neural optimal transport for single-cell perturbation responses.
  - URL: https://www.nature.com/articles/s41592-023-01969-x
- GEARS / graph-enhanced genetic perturbation prediction.
  - URL: https://www.nature.com/articles/s41587-023-01905-6
- CPA / compositional perturbation autoencoder.
  - URL: https://www.embopress.org/doi/full/10.15252/msb.202211517

Mapping:

```text
CellOT -> population-level transport metrics and nulls
GEARS -> gene/gene-pair action descriptor priors
CPA -> compositional action/context discipline
```

### Anti-collapse and student/teacher stabilization

Use these only as stability components, not as the primary scientific claim.

Relevant sources:

- BYOL, 2020: online/target networks with EMA target.
  - URL: https://arxiv.org/abs/2006.07733
- VICReg, 2021/2022: variance/covariance regularization to avoid collapse.
  - URL: https://arxiv.org/abs/2105.04906
- Barlow Twins, 2021: redundancy reduction via cross-correlation.
  - URL: https://arxiv.org/abs/2103.03230
- DINO, 2021: self-distillation with momentum encoder and ViT representation diagnostics.
  - URL: https://arxiv.org/abs/2104.14294

---

## Non-negotiable JEPA identity rules

A model is not BioFlow-JEPA unless all of these are true:

```text
online/context encoder path used = 1.0
raw-linear PLS main path used = 0.0
condition_key feature present = 0.0
biological_key one-hot feature present = 0.0
teacher target latents are stop-gradient = 1.0
EMA target encoders are present = 1.0
transition prediction is in latent z_bio space = 1.0
transition target is EMA teacher z_bio or teacher delta = 1.0
z_bio and z_tech remain separate = 1.0
z_bio is used for retrieval and transition = 1.0
z_tech is not used as a shortcut into biological transition = 1.0
```

Raw RNA/count/image reconstruction may exist only as auxiliary diagnostics. It must not dominate the model identity or promotion claim.

---

## Forbidden shortcuts

Do not use:

```text
condition_key as an input
biological_key one-hot as an input
exact target-key lookup
test target means
train+test pooled targets
batch id as a transition input for biological z_bio
raw-linear PLS as main representation path
metadata leakage from eval/test split
Norman batch claims without real batch metadata
chemical dose claims for Norman dose_val = 1 / 1+1
```

Do not run:

```text
BMJ002-BMJ006 before fixing delta optimization
more BMJ001 steps as the default next experiment
larger MLP only
higher batch-invariance weight only
population prototype transition before delta operator passes
GEARS/action-token complexity before delta operator passes
```

---

## Required output directory

Create:

```text
outputs/autoresearch_bioflow_jepa_phase4/
```

Initialize or update:

```text
outputs/autoresearch_bioflow_jepa_phase4/results.tsv
outputs/autoresearch_bioflow_jepa_phase4/research_journal.md
outputs/autoresearch_bioflow_jepa_phase4/architectural_changes_log.md
outputs/autoresearch_bioflow_jepa_phase4/family_allocation.md
outputs/autoresearch_bioflow_jepa_phase4/BASELINE_REGISTRY.md
outputs/autoresearch_bioflow_jepa_phase4/papers_consulted.md
outputs/autoresearch_bioflow_jepa_phase4/external_resources.md
outputs/autoresearch_bioflow_jepa_phase4/identity_violations_considered.md
outputs/autoresearch_bioflow_jepa_phase4/DELTA_OPERATOR_AUDIT.md
outputs/autoresearch_bioflow_jepa_phase4/final_report.md  # only when stop condition fires
```

---

## Phase 4 research question

```text
Can a real cross-modal, action-conditioned JEPA learn a biological transition operator by fitting an action-conditioned latent vector field in z_bio space, such that predicted transitions move control states toward held-out perturbed teacher states rather than anti-aligning with teacher deltas?
```

This is narrower than the prior Phase 3 goal. The point is to fix this concrete failure:

```text
teacher delta informative, predicted delta anti-aligned, predicted delta rank collapsed
```

---

## Stage A: Inventory and fixed baselines

Before any architecture change, read these files if present:

```text
outputs/autoresearch_biomech_jepa_phase3/final_report.md
outputs/autoresearch_biomech_jepa_phase3/results.tsv
outputs/autoresearch_biotech_jepa_phase2/final_report.md
outputs/autoresearch_biotech_jepa_phase2/research_journal.md
outputs/autoresearch_bioaction_jepa_v1/final_report.md
outputs/autoresearch_synth_lite/model_of_record.md
```

Also read repository files:

```text
perturb_jepa/models/biotech_jepa.py
perturb_jepa/training/biotech_losses.py
perturb_jepa/training/biotech_trainer.py
perturb_jepa/evaluation/biotech_metrics.py
perturb_jepa/training/bioaction_batches.py
perturb_jepa/training/norman_biotech_batches.py
scripts/train_biotech_jepa.py
scripts/evaluate_biotech_jepa.py
```

Write `BASELINE_REGISTRY.md` with at least:

```text
Protected PLS model of record: rank-3 train-split-only raw-linear readout
BTJ001 transition_source_cosine_improvement: +0.0161
BTJ001 transition_to_target_recall@1: 0.4375
BTJ001 batch_allocation_gap: +0.2500
BMJ001 transition_source_cosine_improvement: -0.1695
BMJ001 transition_to_target_recall@1: 0.0968
BMJ001 delta_cosine: -0.0332
BMJ001 delta_prediction_effective_rank: 4.7813
BMJ001 delta_teacher_effective_rank: 12.4196
BMJ001 image_to_RNA_recall@1: 0.1290
BMJ001 RNA_to_image_recall@1: 0.0968
Norman BTJ002 transition_source_cosine_improvement: +0.0313
Norman BTJ002 transition_to_target_recall@1: 0.0625
```

If any value differs in raw files, record the discrepancy and use the raw artifact value.

---

## Stage B: Delta operator audit before model changes

This is mandatory. Do not write BioFlow model code until this audit is complete.

Create:

```text
outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/
```

Write:

```text
DELTA_OPERATOR_AUDIT.md
DELTA_LATENT_CACHE_SUMMARY.md
DELTA_BASELINE_RESULTS.tsv
DELTA_OPTIMIZATION_DIAGNOSTICS.tsv
REOPENING_DECISION.md
```

### B1. Cache teacher latents

Implement or reuse a script:

```text
scripts/cache_bioflow_teacher_latents.py
```

For `synth_genetic_anchor_lite/test_heldout_perturbation`, cache train and eval condition-pair teacher latents:

```text
source_z_bio_teacher
source_z_bio_online
target_z_bio_teacher
target_z_bio_online
source_z_tech_teacher
target_z_tech_teacher
action_descriptor
perturbation_id  # label only, not one-hot input unless already part of action descriptor contract
cell_line_id     # context label only unless allowed by prior BioTech config
batch_id         # diagnostics only
split
condition labels # diagnostics only, never model input
```

For Norman, cache RNA-only teacher latents if available from current BioTech code. Norman is diagnostic only.

### B2. Compute teacher-delta statistics

For every train/eval split:

```text
delta = target_z_bio_teacher - source_z_bio_teacher
```

Report:

```text
delta mean norm
delta std mean
delta effective rank
delta covariance spectrum
delta cosine nearest-neighbor recall@1
delta batch-probe accuracy if batch labels exist
delta perturbation/action-probe accuracy
delta source-target cosine gap distribution
fraction of near-zero deltas
```

If target deltas are tiny, anisotropic, or rank-deficient, state that explicitly.

### B3. Baseline transition operators

Before neural training, evaluate simple transition operators on the cached teacher latents.

Baselines:

```text
source_as_target_null: z_pred = source_z_bio
mean_delta_null: z_pred = source_z_bio + mean_train_delta
action_mean_delta: z_pred = source_z_bio + mean_train_delta_by_action, fallback global mean
action_ridge_delta: ridge regression from [source_z_bio, action_descriptor] to delta
action_low_rank_ridge: low-rank ridge/SVD regression to delta
action_knn_delta: nearest train action/source delta, train only
oracle_train_delta_upper_bound: train split only, diagnostic ceiling; never promotion candidate
```

Metrics:

```text
transition_source_cosine_improvement
transition_to_target_recall@1
transition_to_target_median_rank
delta_cosine
delta_magnitude_ratio
absolute_target_cosine
delta_prediction_effective_rank
batch_allocation_gap if z_tech available
```

Decision:

```text
If no simple baseline can produce positive transition_source_cosine_improvement on train or eval, pause architecture search and write final_report.md.
If action_ridge_delta or action_low_rank_ridge gives positive improvement, use it as an initialization and comparator.
```

### B4. Gradient/sign audit

Before full model training, run a frozen-encoder transition-only optimization for a few steps using current BMJ/BioTech latents.

Audit questions:

```text
Does one optimizer step improve train transition cosine?
Is predicted delta initially anti-aligned with teacher delta?
Do gradients on the final transition layer point in the expected sign?
Does L2/cosine loss on raw delta create magnitude collapse?
Does endpoint cosine alone ignore delta direction?
Does delta whitening improve sign and rank?
Does source-improvement hinge prevent negative updates?
```

Write exact conclusions in `DELTA_OPERATOR_AUDIT.md`.

### B5. Reopening decision

Write `REOPENING_DECISION.md` with one of:

```text
PHASE4_DELTA_OPERATOR_REOPEN_APPROVED
PHASE4_DELTA_OPERATOR_REOPEN_DENIED
```

Reopen only if all are true:

```text
teacher delta has measurable rank and variance
at least one simple baseline has positive transition_source_cosine_improvement OR train-only transition optimization improves in the first 20 steps
there is a targeted fix beyond larger MLP
no leakage from condition_key/test target means
```

If reopening is denied, write `final_report.md` and stop.

---

## Stage C: Implement BioFlow-JEPA only if Stage B passes

### New files

Implement the smallest viable code path, preferably by extending BioTech-JEPA rather than duplicating everything.

Recommended files:

```text
perturb_jepa/models/bioflow_jepa.py
perturb_jepa/training/bioflow_losses.py
perturb_jepa/training/bioflow_trainer.py
perturb_jepa/evaluation/bioflow_metrics.py
scripts/train_bioflow_jepa.py
scripts/evaluate_bioflow_jepa.py
scripts/cache_bioflow_teacher_latents.py
scripts/run_delta_operator_audit.py
tests/test_bioflow_vector_field.py
tests/test_delta_whitening.py
tests/test_bioflow_source_improvement.py
tests/test_bioflow_identity.py
```

### Config

Create `BioFlowJEPAConfig` with fields:

```python
@dataclass(frozen=True)
class BioFlowJEPAConfig:
    base_biotech_config: BioTechJEPAConfig
    transition_mode: str = "vector_field"  # vector_field | low_rank_koopman | hybrid
    flow_steps: int = 4
    flow_tau_samples: int = 1
    use_delta_whitening: bool = True
    delta_whitening_rank: int | None = None
    use_tangent_projection: bool = True
    use_source_improvement_hinge: bool = True
    source_improvement_margin: float = 0.02
    use_action_conditioned_film: bool = True
    use_low_rank_operator: bool = True
    low_rank_operator_rank: int = 8
    freeze_encoders_for_transition_steps: int = 100
    transition_lr_multiplier: float = 3.0
    endpoint_loss_weight: float = 1.0
    velocity_loss_weight: float = 1.0
    delta_direction_loss_weight: float = 2.0
    delta_magnitude_loss_weight: float = 0.2
    source_improvement_loss_weight: float = 2.0
    delta_rank_variance_weight: float = 0.1
    action_negative_loss_weight: float = 0.2
    zero_action_identity_weight: float = 0.5
    z_tech_leakage_penalty_weight: float = 0.0  # diagnostics first, do not overuse
```

### Core module: DeltaWhitening

Implement a train-split-only delta whitener:

```python
class DeltaWhitening(nn.Module):
    def fit(delta_train: Tensor) -> DeltaWhitening: ...
    def whiten(delta: Tensor) -> Tensor: ...
    def unwhiten(delta_w: Tensor) -> Tensor: ...
```

Rules:

```text
fit only on train teacher deltas
save mean, basis, singular/std values
never fit on eval/test deltas
record explained variance
```

Purpose:

```text
BMJ001 delta targets had std mean 0.0819 and prediction rank collapse.
Whitening should stop small/anisotropic deltas from being ignored by endpoint cosine.
```

### Core module: ActionConditionedVectorField

Implement:

```python
class ActionConditionedVectorField(nn.Module):
    def forward(
        self,
        z_tau: Tensor,
        tau: Tensor,
        action: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        ...
```

Required design:

```text
input: z_tau in z_bio space or whitened delta/tangent space
input: tau scalar embedding
input: action descriptor / perturbation embedding
optional input: source z_bio context
output: velocity in z_bio delta space or whitened delta space
```

Use FiLM or gated residual conditioning, not only naive concatenation:

```text
hidden = MLP(LayerNorm(z_tau + tau_embedding))
action_scale, action_shift = action_mlp(action).chunk(2)
hidden = hidden * (1 + tanh(action_scale)) + action_shift
velocity = final_layer(hidden)
```

Initialization:

```text
final layer near zero if no ridge warm-start
or initialize final layer from action_ridge_delta / low-rank operator if Stage B shows it helps
log initial delta norm, cosine, and source improvement
```

### Core module: LowRankActionKoopman

Implement a simple action-conditioned low-rank operator:

```python
class LowRankActionKoopman(nn.Module):
    def forward(self, z_source: Tensor, action: Tensor) -> Tensor:
        # returns delta or next_z
        g = action_to_rank_gates(action)              # [B, r]
        left = z_source @ V                           # [B, r]
        bilinear_delta = (left * g) @ U.T             # [B, D]
        bias_delta = action_to_bias(action)           # [B, D]
        return bilinear_delta + bias_delta
```

Purpose:

```text
A low-rank controllable transition should be harder to anti-align than an unconstrained MLP.
It also gives a mechanistic audit: which action components control which latent directions?
```

### Core module: FlowIntegrator

Implement deterministic integration:

```python
def integrate_vector_field(vf, z0, action, *, steps: int, context=None):
    z = z0
    for k in range(steps):
        tau = torch.full((z.shape[0],), k / steps, device=z.device, dtype=z.dtype)
        v = vf(z, tau, action, context=context)
        z = z + v / steps
        if normalize_endpoint:
            z = F.normalize(z, dim=-1)
    return z
```

Use Euler first. Add RK4 only if Euler passes smoke tests.

### Tangent projection for normalized latents

If `z_bio` is normalized, delta vectors should live in a tangent-like space. Implement optional projection:

```python
def tangent_project(v, z):
    return v - (v * z).sum(dim=-1, keepdim=True) * z
```

Endpoint:

```python
z_pred = F.normalize(z_source + delta_pred, dim=-1)
```

Record whether tangent projection is on.

---

## Stage D: BioFlow-JEPA losses

Use online source context and EMA teacher target. Stop gradient on all teacher targets.

Definitions:

```text
s_online = online/context z_bio(control)
s_teacher = EMA teacher z_bio(control).detach()
t_teacher = EMA teacher z_bio(perturbed).detach()
a = action descriptor / action embedding
true_delta = t_teacher - s_teacher
```

Sample:

```text
tau ~ Uniform(0, 1)
z_tau = (1 - tau) * s_teacher + tau * t_teacher
velocity_target = true_delta
velocity_pred = v_theta(z_tau, tau, a, context=s_online)
z_pred = integrate v_theta from s_online to tau=1
pred_delta = z_pred - s_online
```

Loss terms:

### 1. Velocity JEPA loss

```python
L_velocity = mse_or_smooth_l1(whiten(velocity_pred), whiten(velocity_target))
```

### 2. Delta direction loss

```python
L_delta_direction = 1 - cosine(pred_delta, true_delta.detach())
```

If cosine is negative, add a stronger penalty:

```python
L_anti_align = relu(-cosine(pred_delta, true_delta.detach()))
```

### 3. Endpoint latent JEPA loss

```python
L_endpoint = 1 - cosine(z_pred, t_teacher)
```

### 4. Source-improvement hinge

This is mandatory because BMJ001 made the source worse.

```python
source_cos = cosine(s_online, t_teacher).detach()
pred_cos = cosine(z_pred, t_teacher)
L_improve = relu(source_cos + margin - pred_cos)
```

### 5. Magnitude calibration

```python
L_mag = smooth_l1(log(norm(pred_delta) + eps), log(norm(true_delta) + eps))
```

Also log:

```text
delta_magnitude_ratio = mean(norm(pred_delta) / norm(true_delta))
```

### 6. Predicted-delta variance/rank floor

Use VICReg-style variance/covariance on predicted deltas. Do not let predicted deltas collapse to a 4.78 effective-rank space again.

```python
L_delta_var = variance_floor(pred_delta)
L_delta_cov = covariance_offdiag(pred_delta)
```

Diagnostics:

```text
delta_prediction_effective_rank
delta_teacher_effective_rank
rank_ratio
std_mean_pred_delta
std_mean_teacher_delta
```

### 7. Action-negative contrast

For a batch, permute actions across samples:

```text
z_wrong = transition(s_online, shuffled_action)
```

Add a margin loss so the correct action endpoint is closer to the target than wrong-action endpoints:

```python
L_action_neg = relu(cos(z_wrong, t_teacher) - cos(z_pred, t_teacher) + margin)
```

### 8. Zero/control action identity

For control/no-op action pairs, require near-zero transition:

```python
L_zero = ||transition(s, zero_action) - s||
```

Use only when a true zero/no-op action is present.

### 9. Preserve BioTech JEPA losses

Retain but downweight:

```text
RNA program JEPA
image region JEPA
RNA->image JEPA
image->RNA JEPA
z_bio/z_tech separation diagnostics
anti-collapse diagnostics
```

Do not let reconstruction dominate.

---

## Stage E: Experiments

Use exact labels. Update `results.tsv` after every run.

### BFJ000: Delta operator audit

Type: diagnostic only.

Run:

```text
scripts/run_delta_operator_audit.py
```

Decision labels:

```text
PHASE4_DELTA_OPERATOR_REOPEN_APPROVED
PHASE4_DELTA_OPERATOR_REOPEN_DENIED
```

Stop if denied.

### BFJ001: Frozen-encoder BioFlow transition probe

Hypothesis:

```text
BMJ001 failed because the endpoint delta MLP optimization anti-aligned with informative teacher deltas. A frozen-encoder vector-field transition trained on whitened teacher deltas plus source-improvement hinge should produce positive transition improvement without changing the representation.
```

Constraints:

```text
freeze RNA/image encoders and EMA target encoders
train transition module only
use synth_genetic_anchor_lite/test_heldout_perturbation
no Norman yet
no population prototypes
no action-token graph complexity
```

Run minimal:

```bash
python scripts/train_bioflow_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 50 \
  --eval-steps 5 \
  --batch-size 2 \
  --bag-size 3 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --transition-mode vector_field \
  --use-delta-whitening \
  --use-source-improvement-hinge \
  --freeze-encoders \
  --output-dir outputs/autoresearch_bioflow_jepa_phase4/experiments/BFJ001_frozen_vector_field_seed0 \
  --save-checkpoint
```

Tier 1 keep criteria:

```text
transition_source_cosine_improvement >= +0.0200
delta_cosine > 0.0500
delta_prediction_effective_rank >= 8.0000 OR rank_ratio >= 0.65
transition_to_target_recall@1 >= 0.2500
predicted delta magnitude ratio between 0.25 and 2.50
image_to_RNA recall@1 not below 0.0500
RNA_to_image recall@1 not below 0.0500
condition_key_feature_present = 0.0
pls_raw_linear_main_path_used = 0.0
teacher_stop_gradient_verified = 1.0
```

Hard discard:

```text
transition_source_cosine_improvement <= 0.0000
delta_cosine <= 0.0000
delta_prediction_effective_rank < 5.0000
source-improvement hinge active on >80% eval samples after training
```

If BFJ001 fails a hard discard, write `final_report.md` and stop.

### BFJ002: Ridge/Koopman warm-start transition

Run only if BFJ001 fails narrowly or Stage B showed ridge/low-rank operator is strong.

Hypothesis:

```text
A low-rank action-conditioned Koopman/ridge warm-start provides the correct transition sign and rank before nonlinear vector-field refinement.
```

Add:

```text
low-rank action operator
train-split ridge initialization if available
zero-init residual vector field
```

Keep criteria:

```text
must beat BFJ001 on transition_source_cosine_improvement OR rescue BFJ001 hard discard
must keep delta_cosine positive
must not reduce delta_prediction_effective_rank below BFJ001
```

### BFJ003: End-to-end BioFlow-JEPA with slow encoder update

Run only if BFJ001 or BFJ002 passes.

Hypothesis:

```text
Once transition sign and rank are stable, slow end-to-end encoder training can improve cross-modal and transition metrics without reintroducing batch leakage or collapse.
```

Constraints:

```text
transition module LR >= encoder LR
encoder LR small
EMA target update preserved
z_bio/z_tech split preserved
no condition key
```

Tier 1 keep:

```text
transition_source_cosine_improvement >= +0.0300
delta_cosine > 0.0750
transition_to_target_recall@1 >= max(0.2500, BMJ001 + 0.10)
batch_allocation_gap >= +0.1000
z_bio effective rank >= 6.0000
cross-modal recall in both directions >= 0.0500
```

### BFJ004: Norman RNA-only transition diagnostic

Run only if a synthetic BFJ candidate passes Tier 1.

Norman constraints:

```text
RNA-only
batch ignored unless real batch metadata recovered
dose fixed as guide presence, not chemical concentration
action input is gene multi-hot or gene/gene-pair descriptor, not condition-key one-hot
```

Goal:

```text
show code path and transition loss behaves on real processed Norman h5ad
not a promotion candidate
not evidence for imaging or batch disentanglement
```

Keep criteria:

```text
transition_source_cosine_improvement > 0.0000
delta_cosine > 0.0000
target_z_bio_effective_rank not collapsed
no test rows used for training
```

### BFJ005: Multi-seed Tier 2

Run only if BFJ003 passes cleanly.

Seeds:

```text
0, 1, 2
```

Datasets:

```text
synth_genetic_anchor_lite/test_heldout_perturbation
synth_batch_confound_lite/test as confounding audit only
Norman RNA-only diagnostic optional
```

Tier 2 pass requires:

```text
mean transition_source_cosine_improvement >= +0.0300
all seeds transition_source_cosine_improvement > 0.0000
mean delta_cosine > 0.0500
std of primary metric not larger than the mean effect
no seed has delta rank collapse
no seed has condition-key feature leakage
batch_allocation_gap mean >= +0.1000 on synthetic with batch labels
```

Tier 2 still cannot promote. It only permits designing Tier 3.

---

## Evaluation metrics to implement or preserve

### Core transition metrics

```text
transition_source_cosine_improvement
transition_to_target_recall@1
transition_to_target_recall@5
transition_to_target_median_rank
absolute_target_cosine
delta_cosine
delta_magnitude_ratio
delta_prediction_effective_rank
delta_teacher_effective_rank
delta_rank_ratio
source_improvement_hinge_violation_fraction
```

### JEPA identity metrics

```text
encoder_path_used
pls_raw_linear_main_path_used
condition_key_feature_present
biological_key_onehot_present
teacher_stop_gradient_verified
EMA_target_present
transition_target_stop_gradient
```

### Cross-modal metrics

```text
RNA->image recall@1
RNA->image recall@5
image->RNA recall@1
image->RNA recall@5
```

### Batch/technical disentanglement metrics

```text
z_bio batch-probe accuracy
z_tech batch-probe accuracy
batch_allocation_gap = z_tech_batch_probe - z_bio_batch_probe
z_bio effective rank
z_tech effective rank
```

### Nulls and baselines

Report every candidate against:

```text
source_as_target_null
mean_delta_null
action_mean_delta
action_ridge_delta
action_low_rank_ridge
action_knn_delta
protected PLS geometry baseline where applicable
BTJ001
BMJ001
```

---

## Required tests

Before any training run, pass focused tests:

```bash
pytest \
  tests/test_bioflow_vector_field.py \
  tests/test_delta_whitening.py \
  tests/test_bioflow_source_improvement.py \
  tests/test_bioflow_identity.py \
  tests/test_biotech_jepa_model.py \
  tests/test_norman_biotech_batches.py
```

Test requirements:

```text
DeltaWhitening fit uses train split only.
DeltaWhitening inverse approximately reconstructs held-out deltas without fitting on held-out data.
Vector field output has correct shape.
Integration with zero velocity returns source.
Source-improvement hinge is zero only when prediction improves over source by margin.
Teacher target tensors have requires_grad=False.
Condition key tensors are not accepted as model inputs.
PLS raw-linear path is not used by BioFlow-JEPA main path.
Tangent projection returns vector orthogonal to source latent when enabled.
Action-negative loss increases when actions are shuffled.
```

---

## Documentation requirements

Every experiment journal entry must include:

```text
hypothesis
implementation files changed
initialization details
whether encoders were frozen
transition module parameter count
loss weights
teacher target construction
train/eval split names
null baselines
metrics
identity checks
decision label
learning
stop-condition check
```

Every architecture change must log:

```text
raw transition contribution ratio
post-whitening contribution ratio
endpoint delta contribution ratio
pred_delta_norm / true_delta_norm
source-improvement hinge violation fraction
rank ratio
whether any cap or clamp is active
```

---

## Decision labels

Use only these labels for Phase 4:

```text
PHASE4_DELTA_OPERATOR_REOPEN_APPROVED
PHASE4_DELTA_OPERATOR_REOPEN_DENIED
BFJ_TIER1_KEEP_CONTROLLED_SIGNAL
BFJ_TIER1_DISCARD_NO_SIGNAL
BFJ_TIER1_DISCARD_ANTI_ALIGNED_DELTA
BFJ_TIER1_DISCARD_RANK_COLLAPSE
BFJ_TIER1_DISCARD_IDENTITY_VIOLATION
BFJ_TIER1_DISCARD_BATCH_LEAKAGE
BFJ_TIER2_PASS_CLEAN_DO_NOT_PROMOTE_YET
BFJ_TIER2_FAIL_SIGNAL_NOT_RETAINED
BFJ_TIER2_FAIL_HIGH_VARIANCE
BFJ_TIER2_FAIL_RANK_OR_BATCH_REGRESSION
SEARCH_CLOSED_NO_NEW_BASELINE
```

---

## Stop conditions

Stop immediately and write `final_report.md` if any occurs:

```text
Stage B reopening denied.
BFJ001 transition_source_cosine_improvement <= 0.0000.
BFJ001 delta_cosine <= 0.0000.
BFJ001 delta_prediction_effective_rank < 5.0000.
Any model uses condition_key/biological_key one-hot input.
Any model uses test target means or pooled train+test targets.
Any model uses raw-linear PLS as BioFlow main path.
Two consecutive BFJ Tier 1 candidates fail by anti-aligned delta.
Delta whitening or source-improvement hinge appears incorrectly implemented.
```

Do not continue autonomously after a stop condition.

---

## What counts as success

A Phase 4 success is not promotion. It is permission to continue.

Minimum useful success:

```text
A frozen or lightly trained BioFlow transition operator produces positive source improvement, positive delta cosine, and non-collapsed predicted-delta rank on held-out perturbation without violating JEPA identity.
```

A strong Phase 4 result:

```text
transition_source_cosine_improvement >= +0.0300
delta_cosine > +0.0750
delta_prediction_effective_rank >= 8.0 or rank_ratio >= 0.65
transition_to_target_recall@1 improves over BMJ001 and approaches or exceeds BTJ001
batch_allocation_gap remains directionally correct
cross-modal retrieval does not collapse
```

Promotion remains prohibited until a later Tier 3 no-regression/generalization phase.

---

## Final launch text for this phase

When starting from the repository root, run the Phase 4 process in this order:

```bash
# 1. Inventory and delta audit
python scripts/run_delta_operator_audit.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --output-dir outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit

# 2. Only if REOPENING_DECISION.md says PHASE4_DELTA_OPERATOR_REOPEN_APPROVED:
pytest \
  tests/test_bioflow_vector_field.py \
  tests/test_delta_whitening.py \
  tests/test_bioflow_source_improvement.py \
  tests/test_bioflow_identity.py \
  tests/test_biotech_jepa_model.py \
  tests/test_norman_biotech_batches.py

# 3. Only if tests pass:
python scripts/train_bioflow_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 50 \
  --eval-steps 5 \
  --batch-size 2 \
  --bag-size 3 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --transition-mode vector_field \
  --use-delta-whitening \
  --use-source-improvement-hinge \
  --freeze-encoders \
  --output-dir outputs/autoresearch_bioflow_jepa_phase4/experiments/BFJ001_frozen_vector_field_seed0 \
  --save-checkpoint
```

Remember: the first objective is to stop the learned delta from anti-aligning. Do not add complexity until the transition operator can move source latents toward target teacher latents in the correct direction.
