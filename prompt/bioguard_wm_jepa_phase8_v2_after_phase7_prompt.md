# Phase 8 v2 Codex Prompt: BioGuard-WM-JEPA After Phase 7

## Direct instruction

Read and apply this file verbatim as the active Phase 8 amendment.

This file **replaces** the earlier Phase 8 prompt. The earlier prompt was written before the Phase 7 result was available. Phase 7 is now complete and must be treated as a hard constraint, not background context.

Start Phase 8 v2 in:

```text
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/
```

The paper `2512.24497v3.pdf` has been placed somewhere in the repository by the user. Locate it before coding:

```bash
find . -iname '2512.24497v3.pdf' -o -iname '*jepa*wm*.pdf' -o -iname '*24497*.pdf'
```

If the paper is absent, write:

```text
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/PAPER_MISSING.md
```

and stop without architecture changes.

Do **not** continue discarded Phase 4/5/6/7 candidates by training longer. Do **not** launch a full end-to-end JEPA run unless the frozen-latent predictor gates pass first. Do **not** promote any Phase 8 v2 result. Phase 8 v2 can only produce one of these outcomes:

```text
1. a clean negative result: no safe residual exists above the full-ridge floor under current data;
2. a frozen-latent action-AdaLN residual that safely exceeds the floor and is eligible for a later full JEPA test;
3. a full BioGuard-WM-JEPA Tier 1 candidate only if the frozen-latent gates pass first.
```

---

## Phase 7 update that changes the plan

Phase 7 ended with:

```text
PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA
```

The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.

The protected full train-only action-ridge transition floor is:

```text
transition improvement = 0.0057
delta cosine = 0.3980
recall@1 = 0.4815
delta rank = 10.2835
magnitude ratio = 0.7744
```

Phase 7 ran:

```text
BSG000 floor reproduction
BSG001 residual target/split audit
BSG002 spectral residual CV selection
BSG003 kernel residual CV selection
BSG004 program residual CV selection
```

BSG005-BSG008 were not run because **no residual passed train-only cross-fitted selection**.

The deployed output for BSG002-BSG004 was the protected zero-residual floor fallback:

```text
prediction = source_z_bio + ridge_floor_delta
residual_scale = 0
residual_gate = 0
```

Phase 7 train-internal cross-fit outcomes:

```text
spectral:
  selected = False
  cv_lcb_transition_gap = -0.000207
  cv_lcb_recall_gap = -0.041111
  mean_transition_gap = -0.000150
  mean_recall_gap = -0.013889
  action_negative_gap = -0.000150

kernel:
  selected = False
  cv_lcb_transition_gap = 0.000095
  cv_lcb_recall_gap = -0.044454
  mean_transition_gap = 0.000400
  mean_recall_gap = 0.000000
  action_negative_gap = 0.000400

program:
  selected = False
  cv_lcb_transition_gap = 0.000000
  cv_lcb_recall_gap = 0.000000
  mean_transition_gap = 0.000000
  mean_recall_gap = 0.000000
  action_negative_gap = 0.000000
```

Leakage audit passed: eval/test target rows were not used for fitting, whitening/statistics, residual calibration, residual selection, or candidate choice.

No full BioGuard-JEPA candidate was trained in Phase 7. Operator-only probes cannot promote the model.

**Implication for Phase 8 v2:**

The new paper can still help, but Phase 8 must not treat action-AdaLN/RoPE as permission to keep trying residuals blindly. The only allowed new residual is a paper-motivated **action-AdaLN + RoPE JEPA-WM predictor** with floor-preserving initialization, fixed context contract, and train-only cross-fitted risk gating. If it does not pass, close the search under the current data.

---

## Scientific question

The Phase 8 v2 question is:

> Does the JEPA-WM predictor recipe from `2512.24497v3.pdf`—action-conditioned AdaLN, RoPE, fixed train/eval context length, latent endpoint L2/cosine losses, and rollout-contract discipline—produce a residual transition operator that safely improves over the protected train-only full-ridge transition floor on held-out perturbations?

The null hypothesis is now strong:

> Under the current synthetic genetic-anchor data, the full action-ridge floor may be the limit; learned residuals should default to zero unless train-only calibration proves otherwise.

A clean negative answer is acceptable and should stop the loop.

---

## Required files to read before coding

Read these if present. If any file is missing, record it in `missing_context.md` and continue only if safe.

```text
bioguard_wm_jepa_phase8_full_codex_prompt.md
bioguard_jepa_phase7_safe_residual_prompt.md
outputs/autoresearch_bioguard_jepa_phase7/final_report.md
outputs/autoresearch_bioguard_jepa_phase7/results.tsv
outputs/autoresearch_bioguard_jepa_phase7/residual_calibration_report.md
outputs/autoresearch_bioguard_jepa_phase7/residual_selection_report.md
outputs/autoresearch_bioguard_jepa_phase7/BASELINE_REGISTRY.md

final_report (12).md
final_report (11).md
final_report (8).md
final_report (7).md
final_report (6).md
final_report (5).md
DELTA_OPERATOR_AUDIT.md
BIOTECH_JEPA_CODE_INDEX.md
NORMAN_CONTEXT_AUDIT.md
research_journal (14).md
research_journal (15).md
architectural_changes_log (7).md
CURRENT_STATUS_AND_BEST_MODEL_CODE (1).md
FULL_ARCHITECTURE_CODE_BUNDLE.md
SKILL (2).md
```

Then read the paper and write:

```text
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/papers_consulted.md
```

The paper entry must include:

```text
Paper: What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?
Local path: <path>
Extracted lessons:
1. JEPA-WM identity is latent predictive dynamics, not reconstruction/reward/value/policy heads.
2. The predictor/dynamics model is the main engineering object.
3. Action conditioning with AdaLN + RoPE is a strong predictor recipe.
4. Multistep rollout can improve robustness, but target contracts must be exact.
5. Train/eval context length mismatch is invalid.
6. L2 endpoint latent cost is a strong planning/calibration cost.
7. Deterministic predictors can average multimodal futures; residual deployment needs abstention/risk control.
Mapping to this repo:
- state = z_bio;
- action = perturbation/gene/drug descriptor;
- future state = perturbed teacher z_bio;
- protected transition floor = train-only full action-ridge delta;
- residual = optional floor-preserving correction, selected only through train-only calibration.
```

---

## Protected model and protected transition floor

### Model of record

The protected model of record remains:

```text
rank-3 train-split-only PLS raw-linear readout
```

PLS is audit-only. It may be used as a protected baseline or short bootstrap teacher, but it must not become the BioGuard-WM-JEPA main representation path.

### Transition floor

The protected transition floor remains:

```text
full train-only action-ridge delta
```

Known floor values:

```text
transition improvement = 0.0057
delta cosine = 0.3980
recall@1 = 0.4815
delta rank = 10.2835
magnitude ratio = 0.7744
```

Any nonzero residual candidate must preserve or improve this floor. If not, discard it.

---

## Non-negotiable identity and leakage rules

### Real JEPA identity

A full candidate is a real JEPA only if it has:

```text
online/context encoders
EMA target encoders
stop-gradient teacher latents
latent-space target prediction losses
RNA program JEPA when RNA is present
image region JEPA when imaging is present
RNA -> image and image -> RNA latent prediction when both modalities are present
action-conditioned control -> perturbed teacher-state prediction
separate z_bio / z_tech where batch/acquisition metadata exists
encoder readouts as the main representation path
```

Operator-only probes are allowed, but they cannot promote the model.

### Forbidden shortcuts

Do not use any of these as model inputs, residual targets, calibration features, or implicit lookup keys:

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

`condition_key` and `biological_key` may appear only in metadata, grouping reports, leakage reports, and post-hoc evaluation tables.

### Floor preservation rule

Every transition prediction must have the exact form:

```text
floor_z = source_z_bio + ridge_floor_delta
pred_z = floor_z + residual_gate * residual_scale * residual_delta
```

When either:

```text
residual_gate = 0
```

or:

```text
residual_scale = 0
```

then:

```text
pred_z == floor_z
```

within:

```text
max_abs_error <= 1e-6
cosine_error <= 1e-6
transition_metric_gap <= 1e-6
```

---

## Phase 8 v2 architecture

The new candidate family is:

```text
BioGuard-WM-JEPA v2 =
  real cross-modal/action-conditioned JEPA backbone
  + z_bio / z_tech separation where applicable
  + protected train-only full-ridge transition floor
  + floor-preserving action-AdaLN + RoPE JEPA-WM residual predictor
  + fixed context-length contract
  + train-only action-grouped cross-fitted residual calibration
  + zero-default floor fallback
  + optional biological two-step rollout only after one-step gates pass
```

The new predictor is not allowed to replace the floor. It may only add a residual on top of the floor after passing calibration.

### Default context contract

Use fixed context length `3` by default:

```text
token 1 = source/control z_bio
token 2 = ridge-floor predicted z_bio
token 3 = learned residual query token
```

Optional context tokens are allowed only if train and eval use the same layout and `context_contract.json` records the layout exactly.

Train/eval context mismatch is a hard stop.

---

## Required output files

Create and maintain:

```text
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/results.tsv
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/research_journal.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/architectural_changes_log.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/family_allocation.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/BASELINE_REGISTRY.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/papers_consulted.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/external_resources.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/identity_violations_considered.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/context_contract.json
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/residual_calibration_report.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/rollout_contract_report.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/leakage_audit.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/REOPENING_DECISION.md
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/final_report.md
```

`results.tsv` must include:

```text
experiment_id
dataset
eval_split
seed
mode
status
predictor_type
context_length_train
context_length_eval
rollout_steps
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
cv_lcb_delta_cosine_gap
mean_transition_gap
mean_recall_gap
residual_gate_mean
residual_gate_nonzero_fraction
residual_scale
action_negative_gap
rollout_loss_validated
context_contract_validated
leakage_status
identity_status
decision_label
```

---

# Implementation plan

Implement in this order. Do not jump to full JEPA before frozen-latent probes pass.

## Stage A: ingest Phase 7 and reproduce the floor

1. Parse Phase 7 report and write `PHASE7_STATUS_LOCKED.md`.
2. Reproduce the protected full-ridge floor from cached train-only teacher latents.
3. Reproduce that BSG002-BSG004 deploy as zero-residual floor fallback.
4. Write `BASELINE_REGISTRY.md` with exact floor values and provenance.
5. Write `REOPENING_DECISION.md`.

Architecture reopening is allowed only if:

```text
PHASE8V2_REOPEN_PREDICTOR_ASSAY_APPROVED
```

Criteria:

```text
paper found and summarized
Phase 7 report found or equivalent values reconstructed
full-ridge floor reproduced
floor wrapper exactness passed
no leakage detected
previous residual failures locked and not rerun
```

If not, stop and write `final_report.md`.

---

## Stage B: implement predictor primitives

Add:

```text
perturb_jepa/models/jepawm_predictor.py
```

Define:

```python
BioJEPAWMContextConfig
RotaryEmbedding
RotarySelfAttention
ActionAdaLNBlock
ActionAdaLNPredictor
FloorPreservingJEPAWMTransitionHead
```

### `BioJEPAWMContextConfig`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class BioJEPAWMContextConfig:
    z_dim: int
    action_dim: int
    predictor_dim: int = 128
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    context_length: int = 3
    use_uncertainty_token: bool = False
    use_rope: bool = True
    adaln_zero: bool = True
    residual_output_zero_init: bool = True
    max_context_length_seen_at_train: int = 3
```

Rules:

```text
context_length must equal the number of tokens used by both train and eval.
max_context_length_seen_at_train >= context_length.
If eval context length differs from train context length, raise ValueError.
```

### RoPE

Implement RoPE without new dependencies:

```python
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
```

`RotaryEmbedding.forward(seq_len, device, dtype)` returns cos/sin tensors broadcastable to:

```text
[B, heads, T, head_dim]
```

`RotarySelfAttention` must:

```text
use explicit qkv projections
reshape to [B, heads, T, head_dim]
apply RoPE to q/k when enabled
use scaled dot-product attention
support optional boolean attention mask
return [B, T, D]
```

Do not add FlashAttention-specific code.

### `ActionAdaLNBlock`

Use action-conditioned AdaLN in every block:

```python
class ActionAdaLNBlock(nn.Module):
    def __init__(self, dim: int, action_dim: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0, *, use_rope: bool = True, adaln_zero: bool = True): ...
    def forward(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...
```

Required forward logic:

```python
mod = self.action_to_mod(action)  # [B, 6 * D]
shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

x = self.norm1(tokens)
x = x * (1 + scale1[:, None, :]) + shift1[:, None, :]
h = self.attn(x)
tokens = tokens + gate1[:, None, :] * h

x = self.norm2(tokens)
x = x * (1 + scale2[:, None, :]) + shift2[:, None, :]
h = self.mlp(x)
tokens = tokens + gate2[:, None, :] * h
return tokens
```

If `adaln_zero=True`, initialize the final modulation linear layer to zero.

### `ActionAdaLNPredictor`

Inputs:

```python
source_z: torch.Tensor          # [B, z_dim]
floor_z: torch.Tensor           # [B, z_dim]
action_features: torch.Tensor   # [B, action_dim]
prev_context_z: torch.Tensor | None = None
uncertainty: torch.Tensor | None = None
```

Output:

```python
residual_delta: torch.Tensor    # [B, z_dim]
aux: dict[str, torch.Tensor]
```

Required components:

```text
action_encoder: action_features -> predictor_dim
source_proj: z_dim -> predictor_dim
floor_proj: z_dim -> predictor_dim
context_proj: z_dim -> predictor_dim
uncertainty_proj if enabled
type embeddings: source/floor/query/context/uncertainty
query_token
blocks: ModuleList[ActionAdaLNBlock]
out_norm
out_head: predictor_dim -> z_dim
```

If `residual_output_zero_init=True`, initialize `out_head.weight` and `out_head.bias` to zero. At initialization:

```text
residual_delta == 0
FloorPreservingJEPAWMTransitionHead output == floor_z
```

### `FloorPreservingJEPAWMTransitionHead`

Inputs:

```python
source_z: torch.Tensor
action_features: torch.Tensor
ridge_floor_delta: torch.Tensor | None = None
floor_z: torch.Tensor | None = None
residual_gate: torch.Tensor | float = 0.0
residual_scale: torch.Tensor | float = 0.0
prev_context_z: torch.Tensor | None = None
uncertainty: torch.Tensor | None = None
```

Output dict:

```python
{
  "pred_z": pred_z,
  "floor_z": floor_z,
  "ridge_floor_delta": ridge_floor_delta,
  "residual_delta_raw": residual_delta_raw,
  "residual_delta_scaled": residual_gate * residual_scale * residual_delta_raw,
  "residual_gate": residual_gate_tensor,
  "residual_scale": residual_scale_tensor,
}
```

Rules:

```text
If floor_z is absent, compute floor_z = source_z + ridge_floor_delta.
If both floor_z and ridge_floor_delta are absent, raise ValueError.
If residual_gate or residual_scale is zero, pred_z must equal floor_z exactly.
```

---

## Stage C: losses

Add:

```text
perturb_jepa/training/bioguard_wm_losses.py
```

Define:

```python
cosine_endpoint_loss(pred_z, target_z)
l2_endpoint_loss(pred_z, target_z)
delta_cosine_loss(pred_delta, teacher_delta)
source_improvement_hinge(pred_z, source_z, target_z, margin=0.0)
floor_gap_hinge(pred_z, floor_z, source_z, target_z, margin=0.0)
action_negative_contrast_loss(pred_z, target_z, negative_pred_z, margin=0.05)
multistep_rollout_loss(step_predictions, step_targets, mode="last_gradient_only")
```

Definitions:

```python
cosine_endpoint_loss = 1 - cosine_similarity(pred_z, target_z.detach()).mean()
l2_endpoint_loss = mse(pred_z, target_z.detach())
delta_cosine_loss = 1 - cosine_similarity(pred_delta, teacher_delta.detach()).mean()
```

`source_improvement_hinge`:

```python
source_cos = cosine(source_z, target_z.detach())
pred_cos = cosine(pred_z, target_z.detach())
return relu(source_cos + margin - pred_cos).mean()
```

`floor_gap_hinge`:

```python
floor_cos = cosine(floor_z, target_z.detach())
pred_cos = cosine(pred_z, target_z.detach())
return relu(floor_cos + margin - pred_cos).mean()
```

Multistep rollout rule:

```text
For step k, compare prediction with teacher target z_{t+k}.
Never compare step k prediction to the previous prediction.
Never compare step 2 to z_{t+1} if intended target is z_{t+2}.
```

---

## Stage D: train-only cross-fitted calibration

Add:

```text
perturb_jepa/training/bioguard_wm_calibration.py
```

Define:

```python
make_action_grouped_folds(records, n_folds=3, action_key="action_id", seed=0)
compute_transition_metrics(source_z, target_z, floor_z, pred_z)
select_residual_scale_crossfit(fold_metrics, scale_grid, min_lcb_transition_gap=0.0001, min_lcb_recall_gap=0.0, min_lcb_delta_cosine_gap=0.0, require_action_negative_gap=True)
fit_calibrated_residual_gate(...)
```

Action-grouped fold rules:

```text
Group by action identity, perturbation id, gene-pair descriptor, or action descriptor hash.
Never place the same action group in both train and calibration fold for the same split.
If fewer than 3 action groups exist, use 2 folds.
If fewer than 2 action groups exist, write INSUFFICIENT_ACTION_GROUPS and default residual_scale=0.
```

Scale grid:

```python
scale_grid = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
```

Lower confidence bound:

```python
lcb = mean - 1.96 * std / sqrt(max(n_folds, 1))
```

A nonzero residual scale may be selected only if all conditions hold:

```text
LCB transition floor gap >= 0.0001
LCB recall@1 floor gap >= 0.0
LCB delta cosine floor gap >= 0.0
action_negative_gap > 0.0
no individual fold has recall floor gap < -0.05
no leakage flag
no context-contract violation
```

If no scale passes:

```text
residual_scale = 0.0
residual_gate = 0.0
status = CALIBRATION_DEFAULT_TO_FLOOR
```

Write:

```text
outputs/autoresearch_bioguard_wm_jepa_phase8_v2/residual_calibration_report.md
```

Do not loosen these gates during Phase 8 v2.

---

## Stage E: rollout utilities

Add:

```text
perturb_jepa/training/bioguard_wm_rollouts.py
```

Define:

```python
@dataclass
class BiologicalRolloutExample:
    source_key: str
    action_1: Any
    action_2: Any | None
    source_z: torch.Tensor
    target_1_z: torch.Tensor
    target_2_z: torch.Tensor | None
    action_1_features: torch.Tensor
    action_2_features: torch.Tensor | None
    metadata: dict
```

Functions:

```python
build_single_step_examples(...)
build_two_step_genetic_examples(...)
build_two_step_time_or_dose_examples(...)
validate_rollout_examples_no_leakage(...)
```

Allowed two-step biological examples:

```text
control -> single perturbation -> double perturbation, train-only/cross-fitted
control -> early/weak response -> late/strong response, only if labels exist without leakage
```

Forbidden pseudo-rollout claims:

```text
Do not claim control -> floor -> residual is a biological two-step rollout.
It may be used only as an internal predictor-contract test.
```

If no valid two-step biological examples exist, write:

```text
NO_VALID_TWO_STEP_ROLLOUTS.md
```

and skip rollout without failing Phase 8 v2.

---

## Stage F: optional full JEPA wrapper

Add only after BGWM002 passes with a positive residual:

```text
perturb_jepa/models/bioguard_wm_jepa.py
```

Define:

```python
BioGuardWMJEPAConfig
BioGuardWMJEPA
```

Wrapper contract:

```text
1. Reuse existing BioTech/BioGuard-style RNA/image encoders and EMA teachers where available.
2. Produce z_bio and z_tech using existing branches where available.
3. Produce action_features using existing genetic/drug descriptor encoders.
4. Compute or load train-only ridge_floor_delta from the protected floor head.
5. Apply FloorPreservingJEPAWMTransitionHead on z_bio only.
6. Keep z_tech out of the biological transition except optional uncertainty diagnostics.
7. Compute cross-modal JEPA losses as before when both modalities are present.
8. Compute transition JEPA loss against stop-gradient perturbed teacher z_bio.
9. Use calibrated residual gate/scale. Default to zero unless calibration approves.
```

No reconstruction/count loss may become the main objective. They may be auxiliary only.

---

## Stage G: CLI scripts

Add or update:

```text
scripts/train_bioguard_wm_jepa.py
scripts/evaluate_bioguard_wm_jepa.py
scripts/run_bioguard_wm_phase8_v2.py
```

`train_bioguard_wm_jepa.py` arguments:

```text
--dataset
--eval-split
--seed
--device
--steps
--batch-size
--bag-size
--shared-dim
--bio-dim
--tech-dim
--predictor-dim
--predictor-depth
--predictor-heads
--context-length
--rollout-steps
--residual-scale
--calibration-mode {none,crossfit,fixed_zero}
--ridge-floor-path
--latent-cache-path
--output-dir
--save-checkpoint
--paper-path
```

Default synthetic settings:

```text
--dataset synth_genetic_anchor_lite
--eval-split test_heldout_perturbation
--steps 100
--batch-size 2
--bag-size 3
--shared-dim 32
--bio-dim 24
--tech-dim 8
--predictor-dim 64
--predictor-depth 6
--predictor-heads 4
--context-length 3
--rollout-steps 1
--calibration-mode crossfit
```

`run_bioguard_wm_phase8_v2.py` must orchestrate BGWM000-BGWM006, update logs after each experiment, and stop when a stop condition fires.

---

# Required tests

Add or update:

```text
tests/test_jepawm_rope.py
tests/test_action_adaln_predictor.py
tests/test_floor_preserving_jepawm_head.py
tests/test_bioguard_wm_calibration.py
tests/test_bioguard_rollout_contracts.py
tests/test_bioguard_wm_context_contract.py
tests/test_bioguard_wm_no_leakage.py
tests/test_bioguard_wm_identity.py
tests/test_phase7_status_locked.py
```

Run focused tests before training:

```bash
pytest \
  tests/test_jepawm_rope.py \
  tests/test_action_adaln_predictor.py \
  tests/test_floor_preserving_jepawm_head.py \
  tests/test_bioguard_wm_calibration.py \
  tests/test_bioguard_rollout_contracts.py \
  tests/test_bioguard_wm_context_contract.py \
  tests/test_bioguard_wm_no_leakage.py \
  tests/test_bioguard_wm_identity.py \
  tests/test_phase7_status_locked.py
```

## Test requirements

### RoPE

Verify:

```text
RotaryEmbedding shapes are correct.
RotarySelfAttention preserves [B,T,D].
Same seed/input gives same output in eval mode.
No NaNs.
```

### AdaLN-zero

Verify:

```text
ActionAdaLNBlock with adaln_zero=True does not explode at init.
ActionAdaLNPredictor with residual_output_zero_init=True returns residual_delta exactly zero or within 1e-7.
```

### Floor preservation

Construct random tensors:

```python
source_z = torch.randn(B, D)
floor_delta = torch.randn(B, D) * 0.1
floor_z = source_z + floor_delta
```

Verify:

```text
residual_gate=0 -> pred_z == floor_z
residual_scale=0 -> pred_z == floor_z
nonzero gate/scale can change pred_z
```

### Context contract

Train/eval context mismatch must raise:

```text
ValueError("context length mismatch")
```

`context_contract.json` must record:

```text
context_length_train
context_length_eval
token_layout
use_uncertainty_token
rollout_steps
```

### Rollout target bug guard

Create toy states:

```text
z0 = [0, 0]
z1 = [1, 0]
z2 = [1, 1]
```

The test must fail if step-2 prediction is compared to `z1` or the previous prediction. It must pass only when step-2 is compared to `z2`.

### Calibration defaults to floor

Create fold metrics similar to Phase 7 where transition may slightly improve but recall LCB is negative. Verify:

```text
selected residual_scale = 0.0
status = CALIBRATION_DEFAULT_TO_FLOOR
```

### Calibration selects residual when safe

Create fold metrics where all folds improve transition, recall, and delta cosine above floor. Verify selected scale > 0.

### No leakage

Build records containing forbidden keys. Verify model input builders drop/ignore:

```text
condition_key
biological_key
exact target keys
eval target means
batch id as biological shortcut
```

### Phase 7 status locked

Verify parser recognizes:

```text
PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA
no residual passed selection
BSG005-BSG008 not run
floor values match expected values
```

---

# Experiment sequence and gates

## BGWM000: paper, Phase 7, and floor audit

Purpose:

```text
Confirm paper availability, Phase 7 status, floor reproduction, and no leakage.
```

Run:

```bash
python scripts/run_bioguard_wm_phase8_v2.py \
  --stage BGWM000 \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cpu \
  --paper-path <located 2512.24497v3.pdf> \
  --output-root outputs/autoresearch_bioguard_wm_jepa_phase8_v2
```

Pass requires:

```text
paper summarized
Phase 7 status locked
full ridge floor reproduced
floor metrics written
no leakage detected
REOPENING_DECISION.md says PHASE8V2_REOPEN_PREDICTOR_ASSAY_APPROVED
```

If fail, stop.

Decision label:

```text
BGWM000_KEEP_AUDIT_REOPEN_PREDICTOR_ASSAY
```

---

## BGWM001: zero-residual action-AdaLN smoke

Purpose:

```text
Verify the action-AdaLN + RoPE predictor and transition head preserve the full-ridge floor exactly when residual is disabled.
```

Run:

```bash
python scripts/train_bioguard_wm_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 1 \
  --batch-size 2 \
  --bag-size 3 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --predictor-depth 6 \
  --predictor-heads 4 \
  --context-length 3 \
  --rollout-steps 1 \
  --calibration-mode fixed_zero \
  --residual-scale 0.0 \
  --output-dir outputs/autoresearch_bioguard_wm_jepa_phase8_v2/experiments/BGWM001_zero_residual_seed0
```

Pass requires:

```text
floor_gap_transition within ±1e-6
floor_gap_recall within ±1e-6
pred_z equals floor_z when residual_scale=0
context contract valid
leakage audit pass
```

If fail, stop.

Decision label:

```text
BGWM001_KEEP_ZERO_RESIDUAL_CONTRACT
```

---

## BGWM002: frozen-latent action-AdaLN residual with train-only gate

Purpose:

```text
Test exactly one new paper-motivated residual candidate: action-AdaLN + RoPE predictor on frozen teacher latents.
```

This is the only new residual candidate allowed in Phase 8 v2.

Run:

```bash
python scripts/train_bioguard_wm_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 100 \
  --batch-size 2 \
  --bag-size 3 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --predictor-depth 6 \
  --predictor-heads 4 \
  --context-length 3 \
  --rollout-steps 1 \
  --calibration-mode crossfit \
  --output-dir outputs/autoresearch_bioguard_wm_jepa_phase8_v2/experiments/BGWM002_frozen_action_adaln_seed0 \
  --save-checkpoint
```

Loss weights:

```text
endpoint_l2_weight = 1.0
endpoint_cosine_weight = 1.0
delta_cosine_weight = 0.5
source_improvement_hinge_weight = 0.2
floor_gap_hinge_weight = 0.5
action_negative_weight = 0.1
anti_collapse_weight = 0.05
```

Train-only calibration gates:

```text
LCB transition floor gap >= 0.0001
LCB recall@1 floor gap >= 0.0
LCB delta cosine floor gap >= 0.0
action_negative_gap > 0.0
no individual fold recall gap < -0.05
no leakage
context contract valid
```

If cross-fitted calibration selects:

```text
residual_scale = 0
```

then stop with:

```text
BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN
```

This is a valid negative result, not an implementation failure.

If calibration selects:

```text
residual_scale > 0
```

then perform held-out scoring once. Keep only if:

```text
held-out transition improvement >= 0.0057
held-out recall@1 >= 0.4815
held-out delta cosine >= 0.3980
held-out delta rank >= 0.9 * 10.2835
magnitude ratio in [0.5, 1.5]
no leakage
context contract valid
```

If selected residual falls below floor on held-out scoring, stop with:

```text
BGWM002_DISCARD_ACTION_ADALN_CALIBRATION_FALSE_POSITIVE
```

If selected residual preserves or improves floor, continue with:

```text
BGWM002_KEEP_SAFE_ACTION_ADALN_RESIDUAL
```

---

## BGWM003: biological two-step rollout probe

Run only if BGWM002 keeps a positive residual.

Purpose:

```text
Test whether multistep biological rollout improves the already-safe residual without target bugs.
```

Allowed rollouts:

```text
control -> single perturbation -> double perturbation
control -> early/weak response -> late/strong response if labels exist
```

Run:

```bash
python scripts/train_bioguard_wm_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 100 \
  --batch-size 2 \
  --bag-size 3 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --predictor-depth 6 \
  --predictor-heads 4 \
  --context-length 3 \
  --rollout-steps 2 \
  --calibration-mode crossfit \
  --output-dir outputs/autoresearch_bioguard_wm_jepa_phase8_v2/experiments/BGWM003_two_step_rollout_seed0 \
  --save-checkpoint
```

Pass requires:

```text
rollout target contract tests pass
context length train == context length eval
held-out transition improvement >= BGWM002
held-out recall@1 >= BGWM002
no floor regression
no leakage
```

If no valid two-step biological examples exist:

```text
BGWM003_SKIP_NO_VALID_BIOLOGICAL_ROLLOUTS
```

---

## BGWM004: full BioGuard-WM-JEPA wrapper

Run only if BGWM002 keeps a positive residual. BGWM003 is optional if no valid two-step examples exist.

Purpose:

```text
Integrate the safe action-AdaLN residual head into the real encoder-first JEPA model.
```

Run:

```bash
python scripts/train_bioguard_wm_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 200 \
  --batch-size 2 \
  --bag-size 3 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --predictor-depth 6 \
  --predictor-heads 4 \
  --context-length 3 \
  --rollout-steps 1 \
  --calibration-mode crossfit \
  --output-dir outputs/autoresearch_bioguard_wm_jepa_phase8_v2/experiments/BGWM004_full_jepa_seed0 \
  --save-checkpoint
```

Full loss recipe:

```text
rna_program_jepa_weight = 1.0
image_region_jepa_weight = 1.0
rna_to_image_jepa_weight = 1.0
image_to_rna_jepa_weight = 1.0
transition_endpoint_l2_weight = 1.0
transition_endpoint_cosine_weight = 1.0
transition_delta_cosine_weight = 0.5
floor_gap_hinge_weight = 0.5
source_improvement_hinge_weight = 0.2
vicreg_variance_weight = 0.05
vicreg_covariance_weight = 0.05
bio_tech_orthogonality_weight = 0.05
count_decoder_weight = 0.05  # auxiliary only, if count decoder exists
reconstruction_weight = 0.0
```

Pass requires:

```text
real JEPA identity report passes
encoder path used = 1.0
raw-linear PLS main path used = 0.0
condition-key feature present = 0.0
teacher stop-gradient verified = 1.0
transition improvement >= protected floor
recall@1 >= protected floor
delta cosine >= protected floor
residual calibration still defaults to floor if unsafe
batch leakage does not exceed prior failure thresholds
```

If fail, stop.

---

## BGWM005: multi-seed Tier 1.5 check

Run only if BGWM004 passes.

Seeds:

```text
0, 1, 2
```

Run:

```bash
python scripts/run_bioguard_wm_phase8_v2.py \
  --stage BGWM005 \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seeds 0 1 2 \
  --device cuda \
  --output-root outputs/autoresearch_bioguard_wm_jepa_phase8_v2
```

Pass requires:

```text
mean transition improvement >= floor mean
mean recall@1 >= floor mean
mean delta cosine >= floor mean
std not larger than claimed effect
no seed-specific collapse
no leakage
```

Still no promotion. A pass only permits a future Tier 2/Tier 3 plan.

---

## BGWM006: Norman RNA-only diagnostic

Run only if synthetic BGWM004 or BGWM005 passes.

Norman constraints:

```text
RNA-only
A549-only
batch ignored unless real batch/acquisition metadata is recovered
chemical dose ignored; guide presence/composition only
action descriptor = gene multi-hot or gene-pair descriptor
no condition-key one-hot
GEARS-style train/test condition splits respected
```

Run:

```bash
python scripts/train_bioguard_wm_jepa.py \
  --dataset norman \
  --eval-split test \
  --seed 0 \
  --device cuda \
  --steps 200 \
  --batch-size 2 \
  --bag-size 8 \
  --shared-dim 64 \
  --bio-dim 48 \
  --tech-dim 0 \
  --predictor-dim 128 \
  --predictor-depth 6 \
  --predictor-heads 4 \
  --context-length 3 \
  --rollout-steps 1 \
  --calibration-mode crossfit \
  --output-dir outputs/autoresearch_bioguard_wm_jepa_phase8_v2/experiments/BGWM006_norman_rna_only_seed0 \
  --save-checkpoint
```

Norman cannot validate imaging or batch disentanglement from the processed h5ad alone. Report that limitation.

---

# Metrics

Compute exactly:

```python
transition_source_cosine_improvement = mean(cos(pred_z, target_z)) - mean(cos(source_z, target_z))
transition_floor_cosine_improvement = mean(cos(floor_z, target_z)) - mean(cos(source_z, target_z))
floor_gap_transition = transition_source_cosine_improvement - transition_floor_cosine_improvement

delta_cosine = mean(cos(pred_z - source_z, target_z - source_z))
floor_delta_cosine = mean(cos(floor_z - source_z, target_z - source_z))
floor_gap_delta_cosine = delta_cosine - floor_delta_cosine

recall@1 = nearest-neighbor retrieval of target_z among eval target gallery using pred_z
floor_recall@1 = same retrieval using floor_z
floor_gap_recall = recall@1 - floor_recall@1

delta_rank = effective_rank(pred_z - source_z)
magnitude_ratio = mean(norm(pred_z - source_z)) / mean(norm(target_z - source_z))
```

Effective rank:

```python
s = torch.linalg.svdvals(centered_matrix)
p = s / s.sum().clamp_min(eps)
effective_rank = torch.exp(-(p * p.clamp_min(eps).log()).sum())
```

Action-negative gap:

```text
Pair source_z with wrong action descriptor from the batch.
Report mean correct cosine - wrong-action cosine.
Must be > 0 for a nonzero residual to pass.
```

---

# Logging and reports

Every experiment directory must contain:

```text
config.json
metrics_train.jsonl
metrics_eval.json
jepa_identity_report.md
context_contract.json
leakage_report.md
model_card.md
```

`leakage_report.md` must answer:

```text
Were eval target rows used for fitting? no/yes
Were eval target means used? no/yes
Were pooled train+test stats used? no/yes
Was condition_key used as model input? no/yes
Was biological_key used as model input? no/yes
Was exact target-key one-hot used? no/yes
Was batch id used as biological shortcut? no/yes
Was raw-linear PLS used as main path? no/yes
```

Any `yes` means discard unless explicitly label-only diagnostic.

`jepa_identity_report.md` must include:

```text
online_context_encoders_present
ema_target_encoders_present
teacher_stop_gradient_verified
latent_prediction_loss_present
rna_program_jepa_present
image_region_jepa_present
cross_modal_jepa_present
action_conditioned_transition_jepa_present
encoder_path_used
raw_linear_pls_main_path_used
ridge_floor_fallback_present
residual_floor_preservation_verified
```

---

# Stop conditions

Stop immediately and write `final_report.md` if any occur:

```text
paper missing
Phase 7 result not found and cannot be reconstructed
floor not reproduced
floor preservation test fails
context train/eval mismatch
rollout target contract fails
leakage detected
condition_key or biological_key used as input
residual scale selected from eval/test information
BGWM002 selected residual but held-out floor gap is negative
full JEPA identity fails
batch leakage dominates z_bio beyond prior failure thresholds
transition improvement negative
magnitude ratio > 2.0 or < 0.25 for nonzero residual
delta rank collapses below 0.5 * floor delta rank
```

When stopping, `final_report.md` must include:

```text
decision label
model of record status
what was implemented
what was tested
exact metrics
why the stop fired
whether the failure is architecture, data, calibration, leakage, or implementation
recommended next action
```

---

# Decision labels

Use exact labels:

```text
PHASE8V2_PAPER_MISSING_STOP
PHASE8V2_PHASE7_STATUS_MISSING_STOP
PHASE8V2_FLOOR_REPRODUCTION_FAIL_STOP
PHASE8V2_REOPEN_PREDICTOR_ASSAY_APPROVED
BGWM000_KEEP_AUDIT_REOPEN_PREDICTOR_ASSAY
BGWM001_FAIL_FLOOR_PRESERVATION
BGWM001_KEEP_ZERO_RESIDUAL_CONTRACT
BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN
BGWM002_DISCARD_ACTION_ADALN_CALIBRATION_FALSE_POSITIVE
BGWM002_KEEP_SAFE_ACTION_ADALN_RESIDUAL
BGWM003_SKIP_NO_VALID_BIOLOGICAL_ROLLOUTS
BGWM003_DISCARD_ROLLOUT_CONTRACT_FAIL
BGWM003_KEEP_TWO_STEP_ROLLOUT
BGWM004_DISCARD_FULL_JEPA_IDENTITY_FAIL
BGWM004_DISCARD_FULL_JEPA_BELOW_FLOOR
BGWM004_KEEP_FULL_BIOGUARD_WM_JEPA_TIER1
BGWM005_KEEP_MULTI_SEED_READY_FOR_TIER2_PLAN
BGWM006_NORMAN_DIAGNOSTIC_COMPLETE
PHASE8V2_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA
SEARCH_CLOSED_NO_NEW_BASELINE
```

---

# Commands from repo root

## 1. Focused tests

```bash
pytest \
  tests/test_jepawm_rope.py \
  tests/test_action_adaln_predictor.py \
  tests/test_floor_preserving_jepawm_head.py \
  tests/test_bioguard_wm_calibration.py \
  tests/test_bioguard_rollout_contracts.py \
  tests/test_bioguard_wm_context_contract.py \
  tests/test_bioguard_wm_no_leakage.py \
  tests/test_bioguard_wm_identity.py \
  tests/test_phase7_status_locked.py
```

## 2. BGWM000-BGWM002 run

```bash
python scripts/run_bioguard_wm_phase8_v2.py \
  --stage through_BGWM002 \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --output-root outputs/autoresearch_bioguard_wm_jepa_phase8_v2
```

CPU fallback:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python scripts/run_bioguard_wm_phase8_v2.py \
  --stage through_BGWM002 \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cpu \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --output-root outputs/autoresearch_bioguard_wm_jepa_phase8_v2
```

## 3. Continue only if gates pass

```bash
python scripts/run_bioguard_wm_phase8_v2.py \
  --stage continue_if_gates_pass \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seeds 0 1 2 \
  --device cuda \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --output-root outputs/autoresearch_bioguard_wm_jepa_phase8_v2
```

---

# What not to do

Do not:

```text
train longer on BSJ004, BFJ001, BOJ002, BMJ001, or Phase 7 residuals;
rerun spectral/kernel/program residuals as if Phase 7 did not happen;
launch Norman before synthetic gates pass;
use residuals that fail cross-fitted calibration;
select residual scale on held-out eval;
change held-out split semantics;
use exact biological-key lookup;
use raw-linear PLS as the main representation;
turn the method into reconstruction-first autoencoding;
claim promotion from Tier 1;
ignore context train/eval mismatch;
add diffusion/stochastic heads in Phase 8 v2.
```

---

# Final report template

When the phase stops, write:

```markdown
# BioGuard-WM-JEPA Phase 8 v2 Final Report

## Decision label
<exact label>

## Model of record
Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass supersedes it.

## Phase 7 status integration
- Phase 7 decision:
- Floor values:
- Residual candidates locked as failed:

## Paper integration
- Paper path:
- Lessons used:
- What was implemented from the paper:
- What was deliberately not implemented:

## What was implemented
- Files changed:
- New classes:
- New scripts:
- New tests:

## What was tested
- BGWM000:
- BGWM001:
- BGWM002:
- BGWM003:
- BGWM004:
- BGWM005:
- BGWM006:

## Key metrics
| experiment | transition improvement | delta cosine | recall@1 | delta rank | floor gap transition | floor gap recall | residual scale | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## JEPA identity
- online/context encoders:
- EMA target encoders:
- stop-gradient target latents:
- latent transition prediction:
- cross-modal JEPA:
- raw-linear PLS main path used:

## Leakage status
- eval target rows used for fitting:
- condition_key input:
- biological_key input:
- eval target means:
- pooled train+test stats:

## Main interpretation
<concise scientific interpretation>

## Recommendation
<continue / amend / close>
```

---

# Success definition

Phase 8 v2 is successful if it answers the predictor question cleanly.

A positive result:

```text
Action-AdaLN + RoPE JEPA-WM residual passes train-only cross-fitted calibration,
preserves or improves the full-ridge floor on held-out perturbations,
and integrates into the real JEPA wrapper without identity/leakage failure.
```

A negative result:

```text
No safe residual exists above the floor under train-only calibration.
```

Do not hide a negative result by launching more variants. Stop, report, and recommend the next mechanism only after the failure is identified.
