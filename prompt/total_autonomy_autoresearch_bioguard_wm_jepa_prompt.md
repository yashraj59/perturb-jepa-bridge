# Total-Autonomy Autoresearch Prompt: BioGuard-WM-JEPA and Beyond

## Read This First

You are Codex operating as a **fully autonomous computational research agent** for a biological ML model. Your task is to continue research on the user's JEPA-based perturbation model with maximum autonomy.

The user explicitly wants an autonomous loop that does not wait for human amendments after every failure. Therefore, this prompt uses:

```text
AUTONOMY_MODE = CONTINUOUS_DEBATE_COUNCIL
```

This means ordinary scientific stop conditions do **not** end the overall research program. Instead, they end the current candidate/family/phase, trigger a documented Debate Council, write a self-contained amendment, and continue. Only hard escalation triggers may halt execution.

This is still a disciplined research loop. It is not random exploration, not metric hacking, not a license to corrupt splits, not a license to overwrite the protected model of record, and not a license to make biological or clinical claims. The autonomy is about removing the user from the inner-loop decision process, not about removing scientific controls.

---

## Core Research Goal

Build a **real, novel, cross-modal, action-conditioned JEPA world model** for paired or condition-paired scRNA and cell imaging perturbation data.

The model should learn:

```text
control biological state + perturbation action -> perturbed biological state
```

while also learning cross-modal predictive structure:

```text
RNA context -> image latent target
image context -> RNA latent target
joint context -> missing modality latent target
```

The desired end state is a model that can beat the current protected transition floor and cross-modal baselines on held-out perturbation settings without leakage, while preserving biological interpretability, latent rank, and no-regression constraints.

The current candidate line is called:

```text
BioGuard-WM-JEPA
```

But you may invent and implement successor architectures if evidence supports them. Every successor must preserve the real-JEPA identity constraints below.

---

## Absolute Model-Of-Record Rule

The protected model of record remains:

```text
Protected rank-3 train-split-only PLS raw-linear readout
```

This protected model of record is not a learned JEPA promotion. It is the protected baseline/reference. It remains active unless and until a candidate passes Tier 3 exactly.

Rules:

1. Tier 1 never promotes.
2. Tier 2 never promotes.
3. Operator-only probes never promote.
4. Exact-key lookup baselines never promote as JEPA.
5. PLS raw-linear readouts never become the JEPA main path.
6. Only a Tier 3 pass with all no-regression gates satisfied can supersede the model of record.
7. If any candidate improves one metric but fails protected gates, it is a useful failure, not a new baseline.

---

## Current Evidence To Lock Before Starting

Read all current reports before coding. At minimum read these files if present:

```text
final_report (12).md
final_report (11).md
final_report (8).md
final_report (7).md
final_report (6).md
final_report (5).md
final_report (4).md
DELTA_OPERATOR_AUDIT.md
BIOTECH_JEPA_CODE_INDEX.md
NORMAN_CONTEXT_AUDIT.md
research_journal (14).md
research_journal (15).md
architectural_changes_log (7).md
CURRENT_STATUS_AND_BEST_MODEL_CODE (1).md
FULL_ARCHITECTURE_CODE_BUNDLE.md
```

If these files live under `outputs/` or have slightly different names, locate them with:

```bash
find . -iname '*final_report*' -o -iname '*DELTA_OPERATOR_AUDIT*' -o -iname '*BIOTECH_JEPA_CODE_INDEX*' -o -iname '*NORMAN_CONTEXT_AUDIT*' -o -iname '*research_journal*' -o -iname '*architectural_changes_log*'
```

Lock these facts in `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/LOCKED_PRIOR_FACTS.md`:

```text
Phase 1 BioAction-JEPA:
- Real JEPA identity was implemented.
- Stop reason: batch leakage dominated latent state.
- Protected PLS remained model of record.

Phase 2 BioTech-JEPA:
- z_bio / z_tech separation was implemented.
- Synthetic genetic-anchor audit reopened architecture search.
- Norman is RNA-only in processed h5ad; batch and chemical dose cannot be validated there.

Phase 3 BioMechanistic-JEPA:
- Delta targets had headroom.
- BMJ001 delta operator anti-aligned / no useful signal.

Phase 4 BioFlow-JEPA:
- Action-ridge simple baseline was positive.
- Neural vector-field candidate failed with negative signal.

Phase 5 BioOperator-JEPA:
- Neural ridge-equivalence reproduced the train-only action-ridge floor.
- Low-rank control-affine operator fell below floor.

Phase 6 BioSpectral-JEPA:
- Floor wrapper and rank ladder preserved floor.
- Spectral residual improved train fit but fell below floor on held-out transition/retrieval.

Phase 7 BioGuard-JEPA:
- Decision: PHASE7_CLOSE_FLOOR_IS_LIMIT_UNDER_CURRENT_DATA.
- No residual passed train-only cross-fitted selection.
- Spectral, kernel, and program residuals all deployed zero-residual floor fallback.
- Leakage audit passed.
- No full BioGuard-JEPA candidate was trained.
```

Lock these protected full-ridge transition floor values:

```text
transition_source_cosine_improvement = 0.0057
selected_delta_cosine = 0.3980
transition_to_target_recall@1 = 0.4815
delta_prediction_effective_rank = 10.2835
delta_magnitude_ratio = 0.7744
```

These are the minimum transition floor values for any residual/operator/wrapper candidate unless a fresh train-only baseline reproduction shows a discrepancy and you document why.

---

## Paper To Read Locally

The user will place the paper in the repository. Locate it:

```bash
find . -iname '2512.24497v3.pdf' -o -iname '*24497*.pdf' -o -iname '*jepa*wm*.pdf'
```

Read it before implementing Phase 8. Write a file:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/papers_consulted.md
```

Include at least this paper entry:

```text
Title: What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?
Local file: <path>
Extracted implementation lessons:
- JEPA-WMs are predictive latent dynamics models.
- The predictor/dynamics model is central, not just the encoder.
- AdaLN + RoPE is a strong action-conditioning architecture.
- Multistep rollout can help robustness but needs correct target contracts.
- Train/eval context length must match.
- L2/cosine latent endpoint costs are natural for calibration.
- Deterministic predictors can average multimodal futures; residuals need uncertainty or abstention.
Mapping to this project:
- Build an action-AdaLN + RoPE residual predictor.
- Enforce fixed context contract.
- Add biological two-step rollout only after one-step gates pass.
- Keep floor-preserving residual and train-only calibration.
```

If internet access is available, also inspect and cite local notes or official pages/papers for:

```text
I-JEPA
V-JEPA
V-JEPA 2 action-conditioned world model
DINO-WM / JEPA-WM
CellOT / optimal-transport perturbation maps
GEARS / gene interaction perturbation prediction
CPA / compositional perturbation autoencoder
scVI/sysVI for biological/technical variation separation
VICReg / Barlow Twins / BYOL / DINO for collapse avoidance
Conformal risk control / safe policy improvement / cross-fitting
```

If internet is unavailable, do not fabricate citations. Write `external_resources.md` saying external search was unavailable and proceed from local artifacts.

---

## Hard Safety And Scope Boundaries

This is computational model research only.

Do not:

1. provide wet-lab protocols;
2. propose operational biological manipulation steps;
3. make clinical, diagnostic, treatment, patient-specific, or deployment-facing claims;
4. process protected health information unless the user has explicitly provided a de-identified authorized dataset;
5. use test/eval target rows for fitting, whitening, hyperparameter selection, calibration, residual selection, or model choice;
6. use `condition_key`, `biological_key`, exact target-key one-hot features, eval target means, or pooled train+test statistics;
7. modify locked data splits/evaluators to make results easier;
8. present a Tier 1/Tier 2 result as biological validation;
9. silently promote a model;
10. use PLS raw-linear readouts as the main JEPA representation path.

---

## Hard Escalation Triggers

Because the user requested “without stopping,” normal candidate/family/phase failures must not stop the overall program. They trigger Debate Council continuation.

However, immediately halt and write `HARD_ESCALATION_REPORT.md` only for these hard triggers:

```text
HARD_ESCALATE_SAFETY_OR_WETLAB
HARD_ESCALATE_CLINICAL_OR_PHI
HARD_ESCALATE_FORBIDDEN_LEAKAGE_CANNOT_BE_FIXED
HARD_ESCALATE_LOCKED_DATA_SPLIT_CORRUPTED
HARD_ESCALATE_REPO_CANNOT_RUN_ANY_TESTS_AFTER_REPAIR_ATTEMPTS
HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED
HARD_ESCALATE_EXTERNAL_RESOURCE_LICENSE_BLOCKER
HARD_ESCALATE_USER_EXPLICITLY_TERMINATED_RUN
```

Everything else is a pivot event, not a human-wait stop.

---

## Autonomy Mode: Continuous Debate Council

Whenever any of these ordinary stop-like events occurs:

```text
candidate below floor
three Tier 1 failures in same family
family exhausted
metric ambiguity
repeated mode collapse
residual overfit
cross-modal branch collapse
operator contract failure
no residual selected
no Tier 3 winner after a phase
```

perform this sequence without asking the user:

1. Finish the current experiment.
2. Write all normal logs.
3. Write `phase_closure_report_<NNN>.md` rather than a final project stop.
4. Convene Debate Council.
5. Produce `debate_council_<NNN>.md`.
6. Produce `SESSION_AMENDMENT_<NNN>.md`.
7. Append the amendment to `autoresearch.md` under `## Session Amendments`.
8. Continue with the next smallest evidence-based experiment.

### Debate Council Roles

Use these five roles. They are role prompts for your own internal planning, but their outputs must be documented.

```text
Architect:
  proposes a concrete model/mechanism change.

Skeptic:
  argues why the current direction may be wrong; identifies leakage, metric, and overfit risks.

Methodologist:
  focuses on baselines, splits, statistics, calibration, and no-regression gates.

Biologist:
  focuses on perturbation biology, gene programs, population structure, and interpretability.

Monitor:
  does not propose mechanisms; enforces process, summarizes vote, checks identity and hard escalation triggers.
```

### Debate Council Output

Each council file must include:

```text
1. Trigger.
2. Evidence summary from last experiments.
3. Independent proposals from Architect/Skeptic/Methodologist/Biologist.
4. Steelman of at least two opposing proposals.
5. Three-round debate summary.
6. Scoring table:
   - novelty against current loop
   - feasibility
   - identity preservation
   - expected effect size
   - falsifiability
   - leakage risk
   - compute cost
7. Monitor decision.
8. Exact next amendment.
9. Next experiment command or implementation target.
```

### Council Decision Rule

If at least one proposal has average score >= 0.65 and no dimension below 0.40:

```text
COUNCIL_EXECUTE
```

If all proposals are below 0.65 but at least one diagnostic/audit proposal is feasible:

```text
COUNCIL_EXECUTE_METRIC_OR_INTERNAL_AUDIT
```

If all model proposals are weak and all diagnostics are exhausted:

```text
COUNCIL_EXECUTE_DATA_OR_BENCHMARK_REDESIGN
```

If three councils in a row propose near-identical mechanisms:

```text
COUNCIL_MONOCULTURE_DETECTED_RUN_DIAGNOSTIC_OR_DATA_REDIRECTION
```

Do **not** halt for low confidence unless a hard escalation trigger fires. Low confidence means switch to diagnostics, metric investigation, benchmark redesign, or literature search.

---

## Required Output Directory

Create:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/
```

Initialize these files:

```text
autoresearch.md
LOCKED_PRIOR_FACTS.md
BASELINE_REGISTRY.md
results.tsv
research_journal.md
architectural_changes_log.md
family_allocation.md
papers_consulted.md
external_resources.md
identity_violations_considered.md
AUTONOMY_LEDGER.md
MODEL_OF_RECORD.md
LEAKAGE_RULES.md
NO_STOP_POLICY.md
HARD_ESCALATION_REPORT.md  # only if needed
insights/
councils/
amendments/
phase_reports/
experiments/
metrics/
artifacts/
```

`NO_STOP_POLICY.md` must explicitly state:

```text
Ordinary scientific failures do not stop the project. They trigger documented autonomous debate, amendment, and continuation.
Hard safety/compliance/repo-corruption/compute triggers still halt.
```

---

## Results TSV Schema

`results.tsv` must include at least:

```text
experiment_id
phase
family
tier
seed
dataset
eval_split
mode
status_label
transition_improvement
delta_cosine
recall_at_1
median_rank
delta_rank
magnitude_ratio
rna_to_image_recall_at_1
image_to_rna_recall_at_1
z_bio_batch_probe
z_tech_batch_probe
batch_allocation_gap
effective_rank_z_bio
effective_rank_target
collapse_flag
leakage_flag
floor_gap_transition
floor_gap_recall
floor_gap_delta_cosine
selected_residual_scale
calibration_lcb_transition_gap
calibration_lcb_recall_gap
action_negative_gap
context_contract_pass
jepa_identity_pass
notes
artifact_dir
```

---

## Real JEPA Identity Contract

A candidate may call itself JEPA only if it satisfies all of these:

```text
1. Uses online/context encoder path for the main representation.
2. Uses EMA target encoder(s), or a justified frozen teacher target path for frozen-latent operator probes.
3. Uses stop-gradient teacher latent targets.
4. Predicts latent targets, not raw reconstruction as the primary objective.
5. Has explicit predictor modules, not just encoders.
6. Has action-conditioned control -> perturbed transition prediction.
7. Has RNA -> image and image -> RNA latent prediction when both modalities are present.
8. Keeps `condition_key`/`biological_key` label-only and never input features.
9. Does not use PLS raw-linear readouts as the main representation path.
10. Logs identity checks in every experiment artifact.
```

Create or update tests:

```text
tests/test_jepa_identity_contract.py
```

The test must fail if:

```text
raw_linear_pseudobulk or raw_linear_pooled is the main candidate path
teacher targets require gradients
condition_key/biological_key appears in model input tensors
transition prediction is missing
cross-modal prediction is missing for paired RNA+image synthetic data
reconstruction loss dominates the main objective
```

---

## Current Implementation Entry Points

Use current code as the base. Locate files from `BIOTECH_JEPA_CODE_INDEX.md` and related reports.

Expected existing or related files:

```text
perturb_jepa/models/biotech_jepa.py
perturb_jepa/training/biotech_losses.py
perturb_jepa/training/biotech_trainer.py
perturb_jepa/evaluation/biotech_metrics.py
perturb_jepa/training/bioaction_batches.py
perturb_jepa/training/norman_biotech_batches.py
perturb_jepa/training/synthetic_biology_lite.py
scripts/train_biotech_jepa.py
scripts/evaluate_biotech_jepa.py
```

Add new code in small, testable modules rather than bloating existing files.

---

## Phase 8 Starting Point

Start with a controlled implementation of:

```text
BioGuard-WM-JEPA Phase 8 v3: Action-AdaLN + RoPE Predictor Assay
```

Purpose:

```text
Give the JEPA-WM paper's predictor recipe one clean, floor-preserving, leakage-safe chance to beat the Phase 7 floor.
```

This is not “make the residual bigger.” It is a predictor-design assay under strict floor preservation and train-only calibration.

---

# Implementation Plan, Phase 8 v3

## New Files To Add

Add these files unless equivalent files already exist:

```text
perturb_jepa/models/jepawm_rope.py
perturb_jepa/models/action_adaln_predictor.py
perturb_jepa/models/bioguard_wm_jepa.py
perturb_jepa/training/bioguard_wm_losses.py
perturb_jepa/training/bioguard_wm_calibration.py
perturb_jepa/evaluation/bioguard_wm_metrics.py
perturb_jepa/training/bioguard_wm_rollouts.py
scripts/run_bioguard_wm_total_autonomy.py
scripts/train_bioguard_wm_jepa.py
scripts/evaluate_bioguard_wm_jepa.py
```

Add tests:

```text
tests/test_jepawm_rope.py
tests/test_action_adaln_predictor.py
tests/test_floor_preserving_jepawm_head.py
tests/test_bioguard_wm_calibration.py
tests/test_bioguard_rollout_contracts.py
tests/test_bioguard_wm_context_contract.py
tests/test_bioguard_wm_no_leakage.py
tests/test_bioguard_wm_identity.py
tests/test_total_autonomy_council.py
tests/test_phase7_status_locked.py
```

---

## Module: `perturb_jepa/models/jepawm_rope.py`

Implement RoPE utilities.

Required API:

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0): ...
    def forward(self, seq_len: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]: ...


def rotate_half(x: torch.Tensor) -> torch.Tensor: ...


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor: ...
```

Shape contract:

```text
x shape: [batch, heads, tokens, head_dim]
cos/sin shape may broadcast to x.
head_dim must be even.
```

Tests:

```text
- preserves shape
- preserves norm approximately
- deterministic across calls
- fails on odd head_dim
- works on CPU and CUDA if available
```

---

## Module: `perturb_jepa/models/action_adaln_predictor.py`

Implement action-conditioned predictor blocks.

### `ActionAdaLNBlock`

Required behavior:

```text
- Transformer-style block.
- Input tokens: [B, T, D].
- Action/context conditioning vector: [B, A].
- Uses action-conditioned adaptive LayerNorm in every block.
- Uses RoPE on attention Q/K.
- Supports AdaLN-zero initialization.
- Starts near identity when zero-initialized.
```

Suggested implementation:

```python
class ActionAdaLNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        action_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        adaln_zero: bool = True,
    ) -> None:
        ...
```

Use custom QKV attention so RoPE can be applied:

```python
qkv = self.qkv(normed_tokens)
q, k, v = qkv.chunk(3, dim=-1)
q/k/v -> [B, H, T, Dh]
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
attn = softmax(q @ k^T / sqrt(Dh))
```

AdaLN modulation:

```python
shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = action_mlp(action).chunk(6, dim=-1)
```

AdaLN-zero:

```text
Initialize the final action modulation layer to zero.
Initialize residual gates so output equals input at start.
```

### `ActionAdaLNRoPEPredictor`

Required API:

```python
class ActionAdaLNRoPEPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        action_dim: int,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_context_tokens: int = 4,
        output_dim: int | None = None,
        adaln_zero: bool = True,
    ) -> None:
        ...

    def forward(
        self,
        tokens: torch.Tensor,
        action: torch.Tensor,
        *,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

Output:

```text
Return one predicted latent vector [B, output_dim] from a learned target query token or the final token.
```

Context contract must be enforced:

```text
The predictor must know the exact number/order of context tokens used at train time.
If eval provides a different token count or token type order, raise an error unless explicitly configured and trained.
```

---

## Module: `perturb_jepa/models/bioguard_wm_jepa.py`

Implement config and floor-preserving transition wrapper.

### `BioGuardWMJEPAConfig`

Fields:

```python
@dataclass(frozen=True)
class BioGuardWMJEPAConfig:
    bio_dim: int = 24
    action_dim: int = 32
    tech_dim: int = 8
    predictor_dim: int = 64
    predictor_depth: int = 6
    predictor_heads: int = 4
    context_tokens: tuple[str, ...] = (
        "control_z_bio",
        "ridge_floor_z_bio",
        "action_token",
    )
    use_uncertainty_token: bool = False
    adaln_zero: bool = True
    residual_scale_init: float = 0.0
    max_residual_scale: float = 1.0
    preserve_floor_exactly: bool = True
    floor_head_trainable: bool = False
    detach_floor: bool = True
    residual_target_mode: str = "teacher_delta_minus_floor_delta"
    loss_endpoint_weight: float = 1.0
    loss_delta_cosine_weight: float = 1.0
    loss_source_improvement_hinge_weight: float = 0.2
    loss_residual_norm_weight: float = 0.01
    vicreg_weight: float = 0.0
```

### `RidgeFloorHead`

Purpose:

```text
Represent the protected full train-only action-ridge transition floor.
```

Required behavior:

```text
- Fit only on train rows / cached train teacher latents.
- Store coefficients and preprocessing stats as buffers.
- No eval target rows are used.
- Returns `floor_delta` and `floor_endpoint = control_z_bio + floor_delta`.
- Can exactly reproduce previous action-ridge floor metrics.
```

### `FloorPreservingJEPAWMTransitionHead`

Required forward signature:

```python
def forward(
    self,
    control_z_bio: torch.Tensor,
    action_features: torch.Tensor,
    *,
    floor_delta: torch.Tensor,
    z_tech: torch.Tensor | None = None,
    residual_scale_override: float | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    ...
```

Required outputs:

```text
floor_delta
floor_endpoint
raw_residual_delta
residual_delta
residual_scale
predicted_delta
predicted_endpoint
context_tokens
context_contract_hash
```

Floor preservation invariant:

```python
if residual_scale == 0:
    predicted_delta == floor_delta
    predicted_endpoint == control_z_bio + floor_delta
```

This must be exact up to floating point tolerance. Write tests that check `max_abs_diff < 1e-7`.

Context token order:

```text
token 1: control_z_bio

token 2: ridge_floor_z_bio = control_z_bio + floor_delta

token 3: action_token = projection(action_features)

optional token 4: uncertainty_or_tech_token = projection(z_tech or calibration stats)
```

Do not evaluate with a token length/order that was not used during training.

---

## Module: `perturb_jepa/training/bioguard_wm_losses.py`

Implement:

```python
def endpoint_latent_loss(predicted_endpoint, teacher_target): ...
def delta_cosine_loss(predicted_delta, teacher_delta): ...
def source_improvement_hinge_loss(control, predicted_endpoint, teacher_target, margin=0.0): ...
def residual_norm_penalty(residual_delta, floor_delta): ...
def action_negative_contrast_loss(...): ...
def bioguard_wm_transition_loss(outputs, batch, weights): ...
```

Loss philosophy:

```text
The learned residual is optimized only relative to the protected floor.
It must not be rewarded for train fit if it worsens calibrated held-out performance.
```

Residual target:

```text
teacher_residual_delta = teacher_delta - floor_delta
```

Do not train residual to predict the full delta unless the floor delta is explicitly included and the floor-preservation test passes.

---

## Module: `perturb_jepa/training/bioguard_wm_calibration.py`

Implement train-only action-grouped cross-fitting.

Required classes/functions:

```python
@dataclass
class CalibrationResult:
    selected: bool
    selected_scale: float
    cv_lcb_transition_gap: float
    cv_lcb_recall_gap: float
    cv_lcb_delta_cosine_gap: float
    mean_transition_gap: float
    mean_recall_gap: float
    mean_delta_cosine_gap: float
    action_negative_gap: float
    fold_rows: list[dict]
    decision_label: str


def make_action_group_folds(action_ids, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]: ...


def calibrate_residual_scale(
    floor_predictions,
    residual_predictions,
    teacher_targets,
    action_ids,
    candidate_scales=(0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
    min_lcb_transition_gap=0.0001,
    min_lcb_recall_gap=0.0,
    min_lcb_delta_cosine_gap=0.0,
    max_fold_recall_drop=0.05,
) -> CalibrationResult: ...
```

Gate:

```text
A nonzero residual may be selected only if all are true:
- cv_lcb_transition_gap >= +0.0001
- cv_lcb_recall_gap >= 0.0
- cv_lcb_delta_cosine_gap >= 0.0
- action_negative_gap > 0.0
- no individual fold recall gap < -0.05
- leakage report PASS
- context contract PASS
```

Otherwise:

```text
selected = False
selected_scale = 0.0
decision_label = BGWM_RESIDUAL_REJECTED_USE_FLOOR
```

Never tune residual scale on eval/test target rows.

---

## Module: `perturb_jepa/training/bioguard_wm_rollouts.py`

Add biological multistep rollout support, but only activate after one-step Phase 8 gates pass.

Allowed pseudo-rollouts:

```text
control -> single perturbation -> double perturbation
control -> ridge-floor prediction -> residual-corrected endpoint
control -> early/weak response -> late/strong response, only when labels exist without leakage
```

Required contract test:

```text
Given z0, action_a, action_b, z_a_teacher, z_ab_teacher:
- first step trains P(z0, action_a) -> z_a_teacher
- second step trains P(P(z0, action_a), action_b) -> z_ab_teacher
- never trains second step to reproduce its own previous prediction
- never uses eval target means
```

If the data does not contain valid single/double perturbation rollouts, log:

```text
ROLLOUT_NOT_AVAILABLE_NO_VALID_TRAIN_ONLY_CHAIN
```

and do not fabricate pseudo-chains.

---

## Module: `perturb_jepa/evaluation/bioguard_wm_metrics.py`

Required metrics:

```text
transition_source_cosine_improvement
absolute_target_cosine
delta_cosine
transition_to_target_recall@1
transition_to_target_recall@5
transition_to_target_median_rank
delta_prediction_effective_rank
delta_magnitude_ratio
floor_gap_transition
floor_gap_recall@1
floor_gap_delta_cosine
floor_gap_delta_rank
source_as_target_null metrics
action_ridge_floor metrics
mean_delta_null metrics
action_negative_gap
z_bio effective rank
z_tech effective rank
z_bio batch probe
z_tech batch probe
batch allocation gap
RNA->image recall@1
image->RNA recall@1
collapse diagnostics
leakage report
context contract report
JEPA identity report
```

Every evaluation artifact must include:

```text
metrics.json
metrics.md
leakage_report.md
jepa_identity_report.md
context_contract_report.md
calibration_report.md
```

---

## Scripts

### `scripts/run_bioguard_wm_total_autonomy.py`

This is the main orchestrator. It should:

1. create output root;
2. read locked facts;
3. locate local paper;
4. run tests;
5. reproduce floor;
6. run Phase 8 v3;
7. evaluate gates;
8. if failure, write phase report and run Debate Council;
9. automatically continue into next amendment;
10. repeat until hard escalation or external termination.

Command interface:

```bash
python scripts/run_bioguard_wm_total_autonomy.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seeds 0 1 2 \
  --device cuda \
  --output-root outputs/autoresearch_total_autonomy_bioguard_wm_jepa \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --autonomy-mode continuous_debate_council \
  --continue-after-failure true
```

CPU fallback:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python scripts/run_bioguard_wm_total_autonomy.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seeds 0 1 2 \
  --device cpu \
  --output-root outputs/autoresearch_total_autonomy_bioguard_wm_jepa \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --autonomy-mode continuous_debate_council \
  --continue-after-failure true
```

### Orchestrator Loop Pseudocode

```python
while True:
    if hard_escalation_triggered():
        write_hard_escalation_report()
        break

    ensure_logs_exist()
    ensure_model_of_record_locked()
    run_or_verify_step0_baselines()
    run_focused_tests()

    plan = load_current_plan_or_latest_amendment()
    experiment = select_next_smallest_experiment(plan)

    write_pre_experiment_hypothesis(experiment)
    implement_if_needed(experiment)
    run_smoke_tests(experiment)
    run_tier1(experiment)
    evaluate_gates(experiment)
    update_all_logs(experiment)

    if experiment.tier1_passes:
        run_tier2(experiment)
        update_all_logs(experiment)

    if experiment.tier2_passes:
        run_tier3(experiment)
        update_all_logs(experiment)

    if experiment.tier3_passes_all_gates:
        promote_model_of_record(experiment)
        write_promotion_report(experiment)
        # Continue research on next frontier unless externally terminated.
        trigger = "TIER3_WIN_CONTINUE_TO_NEXT_FRONTIER"
        run_debate_council(trigger)
        append_amendment_and_continue()
        continue

    if ordinary_stop_like_event(experiment):
        write_phase_closure_report(experiment)
        run_debate_council(trigger=experiment.decision_label)
        append_amendment_and_continue()
        continue
```

---

# Phase 8 v3 Experiment Sequence

Run focused tests first:

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
  tests/test_total_autonomy_council.py \
  tests/test_phase7_status_locked.py
```

## BGWM000: Lock And Reproduce Floor

Hypothesis:

```text
Before any new predictor, the full action-ridge transition floor must be reproducible exactly.
```

Expected metrics:

```text
transition_improvement = 0.0057 +/- small tolerance
selected_delta_cosine = 0.3980 +/- small tolerance
recall@1 = 0.4815 +/- small tolerance
delta_rank = 10.2835 +/- small tolerance
magnitude_ratio = 0.7744 +/- small tolerance
```

Decision labels:

```text
BGWM000_PASS_FLOOR_REPRODUCED
BGWM000_FAIL_FLOOR_NOT_REPRODUCED_TRIGGER_AUDIT
```

If fail, do not halt. Run a metric/provenance audit via Debate Council.

## BGWM001: Predictor Unit Assay

Hypothesis:

```text
Action-AdaLN + RoPE predictor can learn residual targets on train folds while preserving exact floor at residual scale zero.
```

Run only train-internal evaluation.

Requirements:

```text
- no eval/test target rows used
- floor preservation at residual_scale=0 exact
- context contract pass
- gradients nonzero through predictor
- no output magnitude explosion
```

Decision labels:

```text
BGWM001_PASS_UNIT_ASSAY
BGWM001_FAIL_UNIT_ASSAY_IMPLEMENTATION
BGWM001_FAIL_CONTEXT_CONTRACT
BGWM001_FAIL_FLOOR_PRESERVATION
```

If fail, repair implementation and retest. If repeated, Debate Council pivot.

## BGWM002: Frozen-Latent Action-AdaLN Residual Calibration

Hypothesis:

```text
Compared with spectral/kernel/program residuals, an action-AdaLN + RoPE JEPA-WM residual may generalize better because action conditioning reaches every predictor block and the context contract is fixed.
```

Procedure:

```text
- use cached/frozen z_bio latents
- fit full ridge floor on train only
- train residual predictor on train folds only
- select residual scale via action-grouped cross-fitting only
- evaluate held-out perturbation only after scale selected
```

Gate for nonzero residual:

```text
cv_lcb_transition_gap >= +0.0001
cv_lcb_recall_gap >= 0.0
cv_lcb_delta_cosine_gap >= 0.0
action_negative_gap > 0.0
no individual fold recall gap < -0.05
leakage_report PASS
context_contract PASS
```

Held-out guard:

```text
If a nonzero residual is deployed and held-out floor_gap_transition < 0 or recall@1 < floor recall@1, discard it and continue with Debate Council.
```

Decision labels:

```text
BGWM002_KEEP_SAFE_NONZERO_RESIDUAL
BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN
BGWM002_DISCARD_HELDOUT_BELOW_FLOOR
BGWM002_DISCARD_LEAKAGE_OR_CONTEXT_FAIL
```

If no safe residual is selected, do not run full JEPA wrapper yet. Trigger Debate Council.

## BGWM003: Full Frozen-Backbone BioGuard-WM-JEPA Wrapper

Run only if BGWM002 selects nonzero residual safely.

Hypothesis:

```text
The residual predictor can be wrapped into real JEPA architecture while preserving frozen-latent transition floor.
```

Requirements:

```text
- online/context encoder path used
- EMA teacher target present
- stop-gradient teacher targets
- transition JEPA loss latent-only main objective
- floor preserved
- cross-modal RNA/image losses active where modalities exist
```

Decision labels:

```text
BGWM003_KEEP_FULL_JEPA_WRAPPER_TIER1
BGWM003_DISCARD_IDENTITY_FAIL
BGWM003_DISCARD_BELOW_FLOOR
BGWM003_DISCARD_RETRIEVAL_OR_BATCH_REGRESSION
```

## BGWM004: End-To-End Small JEPA Fine-Tune

Run only if BGWM003 passes.

Hypothesis:

```text
Small end-to-end JEPA fine-tuning can improve z_bio transition targets without losing floor or leaking batch.
```

Constraints:

```text
- PLS audit-only
- floor head frozen
- residual gate initialized to safe selected value
- low LR on encoders
- EMA teachers update normally
- reconstruction/count losses auxiliary only
```

Decision labels:

```text
BGWM004_TIER1_KEEP_END_TO_END_SIGNAL
BGWM004_DISCARD_BELOW_FLOOR
BGWM004_DISCARD_BATCH_LEAKAGE
BGWM004_DISCARD_COLLAPSE
```

## BGWM005: Two-Step Biological Rollout Probe

Run only if BGWM004 passes or Debate Council authorizes an isolated rollout probe.

Allowed chains:

```text
control -> single perturbation -> double perturbation
control -> early/weak -> late/strong, if present
control -> floor -> residual endpoint
```

Decision labels:

```text
BGWM005_KEEP_ROLLOUT_SIGNAL
BGWM005_NOT_AVAILABLE_NO_VALID_CHAINS
BGWM005_DISCARD_ROLLOUT_TARGET_CONTRACT_FAIL
BGWM005_DISCARD_BELOW_ONE_STEP
```

## BGWM006: Norman RNA-Only Diagnostic

Run only after synthetic evidence is non-destructive.

Norman limitations:

```text
- RNA-only
- A549-only in current processed h5ad
- no exposed batch/acquisition metadata
- dose_val is guide-composition notation, not chemical dose
```

Norman can test genetic action generalization only. It cannot validate imaging or batch disentanglement.

Decision labels:

```text
BGWM006_NORMAN_DIAGNOSTIC_SIGNAL
BGWM006_NORMAN_NO_SIGNAL
BGWM006_NORMAN_DATA_LIMITATION
```

No Norman-only result can promote.

---

# Beyond Phase 8: Autonomous Research Families

If Phase 8 fails or plateaus, continue automatically using these families. Do not ask the user for the next phase. Use Debate Council to select the next smallest experiment.

## Family A: Predictor Architecture Repair

Motivation:

```text
The JEPA-WM paper suggests predictor architecture matters. Previous residuals may have failed from weak action conditioning and mismatched context contracts.
```

Allowed mechanisms:

```text
A1. Action-AdaLN + RoPE predictor depth ladder: 3, 6, 9, 12.
A2. Sequence-conditioned action tokens with RoPE.
A3. Feature-conditioned action + RoPE.
A4. AdaLN-zero versus AdaLN nonzero initialization.
A5. Context length W=2,3,5, but train/eval must match.
```

Forbidden:

```text
bigger MLP without predictor diagnostic
context length mismatch
unbounded scale increases
```

## Family B: Safe Residual Selection And Abstention

Motivation:

```text
Phase 7 showed residuals overfit train and fail conservative CV selection.
```

Allowed mechanisms:

```text
B1. Per-action residual abstention gate.
B2. Uncertainty token from calibration fold variance.
B3. Mixture of small residual experts with floor fallback.
B4. Conformal-style risk gate using train-only calibration.
B5. Safe policy improvement-style baseline preservation.
```

Gate:

```text
Must deploy floor fallback when uncertain.
```

## Family C: Representation Repair Before Operator Learning

Motivation:

```text
Transition operators may be limited by z_bio representation quality.
```

Allowed mechanisms:

```text
C1. Refit z_bio teacher using contrastive-free latent prediction.
C2. Local token-level cross-modal JEPA targets.
C3. Image->RNA repair branch with modality dropout.
C4. Cross-batch consensus teacher when valid anchors exist.
C5. z_bio/z_tech orthogonality with capacity controls.
```

Must preserve:

```text
transition floor audit
batch leakage gates
effective rank
```

## Family D: Population Transport JEPA

Motivation:

```text
Single endpoint prediction may be too mean-like; perturbation response is population/distributional.
```

Allowed mechanisms:

```text
D1. Prototype-level transition JEPA.
D2. Latent OT map from control prototype distribution to perturbed prototype distribution.
D3. Sinkhorn loss on train folds only.
D4. Population diversity preservation.
D5. Source-as-target / mean-shrinkage null comparisons.
```

Must log:

```text
cluster coverage
prototype entropy
population spread
rare-state preservation if available
```

## Family E: Program And Graph Action Priors

Motivation:

```text
Genetic perturbations have structure. A gene multi-hot descriptor may be insufficient for held-out perturbations.
```

Allowed mechanisms:

```text
E1. ProgramActionEncoder using train-only gene program membership.
E2. Graph message passing over gene/action descriptors.
E3. GEARS-style gene interaction descriptors.
E4. Reactome/GO/MSigDB program summaries if resources are available and logged.
E5. Low-rank + sparse action effects with floor fallback.
```

Forbidden:

```text
hard-coded gene symbol hacks
forcing named genes to move specific directions
unlogged resource versions
```

## Family F: Metric And Data Redesign

Motivation:

```text
When the floor is hard to beat, architecture may be blocked by benchmark size/noise/split design.
```

Allowed tasks:

```text
F1. Technical duplicate ceiling.
F2. Bootstrap evaluation noise.
F3. Held-out perturbation difficulty stratification.
F4. Synthetic benchmark expansion with more perturbations and cross-batch anchors.
F5. Paired RNA/image split redesign.
F6. New real dataset loader only if data is already available or can be downloaded under acceptable license.
```

Rules:

```text
Do not modify old split semantics to make old results look better.
Create new benchmark names if split/data changes.
```

## Family G: Self-Supervised Pretraining

Motivation:

```text
Small synthetic supervised transition data may not support richer JEPA operators.
```

Allowed mechanisms:

```text
G1. RNA program block JEPA pretraining.
G2. Image region JEPA pretraining.
G3. Cross-modal condition-bag JEPA pretraining.
G4. Masked modality prediction with latent targets only.
G5. PLS bootstrap annealed to zero.
```

Forbidden:

```text
raw reconstruction as main objective
PLS as final main path
```

## Family H: Stochastic Or Multi-Hypothesis JEPA

Motivation:

```text
Deterministic predictors may average multimodal biological futures.
```

Allowed mechanisms:

```text
H1. Mixture-of-experts residual with abstention.
H2. Top-k latent hypotheses evaluated by endpoint distance.
H3. Small latent noise injection with calibration.
H4. Quantile/risk-aware residual gate.
```

Must preserve:

```text
floor fallback
no leakage
no mode collapse
```

## Family I: Count-Aware Auxiliary Decoding

Motivation:

```text
Downstream biology needs count/expression metrics, but JEPA identity must remain latent-prediction-first.
```

Allowed mechanisms:

```text
I1. NB/Poisson auxiliary decoder with low weight.
I2. Program-level count decoder.
I3. Direction-of-effect auxiliary loss.
I4. Top-DE overlap diagnostic only.
```

Forbidden:

```text
count decoder becoming main model identity
training against eval target means
```

## Family J: Literature-Generated Novel Family

When Debate Council concludes existing families are exhausted, run a literature scan and propose one new family. Requirements:

```text
- must address a specific observed failure
- must preserve JEPA identity
- must include a smallest falsifiable Tier 1 experiment
- must include no-leakage gates
- must include why it is not a near-duplicate of failed families
```

---

# Tiered Evaluation Gates

## Tier 1: Single-Seed / Fast Probe

Default:

```text
seed = 0
steps = low-compute smoke unless family requires cached latent probe
primary dataset = synth_genetic_anchor_lite
eval split = test_heldout_perturbation
```

Tier 1 keep requires:

```text
- JEPA identity pass if candidate is full JEPA
- leakage report pass
- transition improvement >= protected floor OR non-destructive diagnostic with strong reason
- if residual/operator: floor_gap_transition >= 0 unless explicitly diagnostic-only
- recall@1 >= floor recall@1 unless explicitly diagnostic-only
- delta cosine >= floor delta cosine unless explicitly diagnostic-only
- no collapse flag
- effective rank not below protected floor by more than 20%
- no context contract violation
- no forbidden feature usage
```

Tier 1 can keep a diagnostic-only experiment without beating the floor only if it answers a clear question needed for the next amendment. Diagnostic-only keeps must not promote or deepen into expensive training without council approval.

## Tier 2: Multi-Seed Validation

Default:

```text
seeds = 0, 1, 2
same split and eval metrics
paired comparison to floor where possible
```

Tier 2 pass requires:

```text
- mean transition improvement above floor by >= +0.002 absolute OR council-defined meaningful threshold
- no seed below floor by > 0.001 transition gap
- recall@1 mean >= floor recall@1
- delta cosine mean >= floor delta cosine
- std not larger than claimed effect
- leakage pass all seeds
- identity pass all seeds
- no collapse
```

Tier 2 does not promote.

## Tier 3: Promotion / No-Regression / Generalization

A candidate can become new model of record only if all are true:

```text
1. Passes Tier 2.
2. Beats protected transition floor on held-out perturbation.
3. Preserves or improves cross-modal retrieval.
4. Preserves z_bio/z_tech separation or passes batch leakage audit.
5. Passes a second validation setting:
   - another synthetic split, or
   - expanded synthetic benchmark, or
   - real paired RNA/image dataset if available, or
   - Norman RNA-only diagnostic as supplementary but not sole validation.
6. Beats source-as-target, mean-delta, action-mean, action-ridge, and simple table baselines where applicable.
7. Has no condition-key leakage.
8. Has a domain report with biological metrics.
9. Has reproducible commands and artifacts.
```

Only then update `MODEL_OF_RECORD.md`.

---

# Biological Metrics

Track layered biological metrics whenever expression/count targets are available:

```text
distributional:
  MSE/MAE/correlation/MMD/Wasserstein if available

direction:
  delta cosine
  signed DE agreement
  direction accuracy
  logFC correlation

program/marker:
  program recovery
  pathway/module consistency
  top50 DE overlap
  gene-set coherence

population:
  cluster/prototype coverage
  entropy/spread
  rare-state preservation if available

manifold:
  neighbor overlap
  centroid shift
  density preservation

nulls:
  source-as-target
  global mean
  train action mean
  action ridge
  variance-shrunk prediction
  exact-key table when exact support exists, but never as JEPA
```

Do not claim biological validity from Tier 1/Tier 2.

---

# Leakage Rules

Create `LEAKAGE_RULES.md` and enforce in tests.

Forbidden in model inputs, training targets, calibration, and selection:

```text
condition_key
biological_key
exact target-key one-hot features
eval/test target means
eval/test target rows for fitting
eval/test whitening/statistics
pooled train+test statistics
batch id as biological transition shortcut
raw-linear PLS as candidate main representation path
```

Allowed:

```text
batch labels for diagnostics
batch labels for z_tech auxiliary head if explicitly logged and not used by transition shortcut
train-only control rows for held-out perturbation controls when data contract already supports it
train-only action descriptors
gene multi-hot action descriptors for Norman
```

Every experiment must write:

```text
leakage_report.md
```

with:

```text
forbidden_inputs_present: true/false
forbidden_target_statistics_present: true/false
train_rows_used_for_fit: count
val_rows_used_for_calibration: count
eval_rows_used_for_scoring_only: count
condition_key_role: label_only/input/absent
batch_id_role: diagnostic/z_tech_aux/input_to_transition/absent
verdict: PASS/FAIL
```

---

# Documentation Discipline

## `research_journal.md` Entry Template

Use this for every experiment:

```markdown
## <experiment_id>: <title>

**Hypothesis**:

**Family**:

**Implementation**:

**Initialization / identity preservation**:

**Data and split**:

**Command**:

**Metrics**:

**Diagnostics**:

**Leakage report**:

**Decision label**:

**Learning**:

**Next action**:
```

## `architectural_changes_log.md`

For every code change log:

```text
file changed
new class/function
parameter count
initialization
expected effect
observed effect
cost/memory impact
identity risk
```

## `AUTONOMY_LEDGER.md`

Every autonomous continuation must log:

```text
cycle number
trigger
council file
amendment file
next action
why user was not asked
hard escalation check result
```

## Insight Briefs

Every 10 experiments or major pivot write:

```text
insights/INSIGHT_BRIEF_<NNN>.md
```

Include:

```text
what has been learned
what mechanisms are retired/cooled
what remains plausible
metric/baseline caveats
next council recommendation
```

---

# Autonomy Amendments

When a new plan is selected, write:

```text
amendments/SESSION_AMENDMENT_<NNN>.md
```

Template:

```markdown
# Session Amendment <NNN>: <Title>

## Trigger

## Evidence

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout unless Tier 3 has superseded it.

## Retired / Cooled Families

## New Or Reopened Family

## Exact Next Experiment

## Implementation Tasks

## Gates

## Do-Not-Run List

## Hard Escalation Check

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.
```

Append this amendment to root `autoresearch.md`.

---

# What To Do If Phase 8 v3 Fails

If BGWM002 selects residual scale zero:

```text
- record BGWM002_CLOSE_NO_SAFE_RESIDUAL_AFTER_ACTION_ADALN
- do not force residual
- run Debate Council
- likely next: metric/data redesign, stochastic abstention, representation repair, or population transport
```

If BGWM002 nonzero residual is selected by train-only calibration but held-out falls below floor:

```text
- record BGWM002_DISCARD_HELDOUT_BELOW_FLOOR
- analyze calibration mismatch
- run Debate Council
- next likely: stricter calibration, uncertainty gate, or larger train fold data
```

If BGWM003 full wrapper fails JEPA identity:

```text
- repair identity first
- do not count as scientific failure until implementation is valid
```

If repeated residual approaches fail:

```text
- cool all residual-only families
- run metric/data/representation audit
- consider benchmark expansion or pretraining
```

---

# What To Do If A Candidate Wins

If a candidate passes Tier 3:

1. Write `PROMOTION_REVIEW.md`.
2. Update `MODEL_OF_RECORD.md`.
3. Preserve all artifacts.
4. Write biological domain report.
5. Continue research automatically to the next frontier via Debate Council.

Do not halt after a Tier 3 win unless the user explicitly terminated the run. The next frontier may be:

```text
- real paired RNA/image data integration
- Norman RNA-only robustness
- expanded synthetic benchmark
- count-aware auxiliary decoder
- population transport
- cross-modal repair
- paper-quality ablations
```

---

# First Commands To Run

From repo root, after the paper has been placed in `papers/` or another repo directory:

```bash
mkdir -p outputs/autoresearch_total_autonomy_bioguard_wm_jepa
find . -iname '2512.24497v3.pdf' -o -iname '*24497*.pdf' -o -iname '*jepa*wm*.pdf'
```

Then run initial tests after implementation:

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
  tests/test_total_autonomy_council.py \
  tests/test_phase7_status_locked.py
```

Then launch:

```bash
python scripts/run_bioguard_wm_total_autonomy.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seeds 0 1 2 \
  --device cuda \
  --output-root outputs/autoresearch_total_autonomy_bioguard_wm_jepa \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --autonomy-mode continuous_debate_council \
  --continue-after-failure true
```

CPU fallback:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python scripts/run_bioguard_wm_total_autonomy.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seeds 0 1 2 \
  --device cpu \
  --output-root outputs/autoresearch_total_autonomy_bioguard_wm_jepa \
  --paper-path "$(find . -iname '2512.24497v3.pdf' | head -n 1)" \
  --autonomy-mode continuous_debate_council \
  --continue-after-failure true
```

---

# Launch Instruction For Codex

Paste this into Codex:

```markdown
Read and apply `total_autonomy_autoresearch_bioguard_wm_jepa_prompt.md` verbatim.

Operate in `AUTONOMY_MODE = CONTINUOUS_DEBATE_COUNCIL`.

The user does not want the research loop to stop after ordinary failed candidates. Therefore, ordinary stop conditions become documented pivot events: write the closure report for the candidate/phase, convene Debate Council, write an amendment, append it to `autoresearch.md`, and continue.

Only hard escalation triggers may halt: safety/wet-lab/clinical/PHI issues, forbidden leakage that cannot be fixed, corrupted locked splits, inability to run any repo tests after repair attempts, compute/storage exhaustion, external license blockers, or explicit user termination.

Start in:

outputs/autoresearch_total_autonomy_bioguard_wm_jepa/

First, lock prior facts from the Phase 1-7 reports, reproduce the protected full action-ridge transition floor, locate and read `2512.24497v3.pdf`, write `papers_consulted.md`, initialize all required logs, and run focused tests.

Then implement Phase 8 v3: a floor-preserving BioGuard-WM-JEPA predictor assay using action-AdaLN + RoPE, fixed train/eval context contract, train-only action-grouped cross-fitted residual calibration, leakage audits, JEPA identity checks, and rollout target bug tests.

The protected rank-3 train-split-only PLS raw-linear readout remains the model of record unless a Tier 3 pass supersedes it. The protected full-ridge transition floor remains the transition floor for residual/operator candidates.

Do not use `condition_key`, `biological_key`, exact target-key one-hot features, eval target means, pooled train+test statistics, or PLS raw-linear heads as the candidate main representation path.

If Phase 8 fails, do not ask the user what to do next. Run the Debate Council procedure, write an amendment, and continue with the next evidence-based family.
```

