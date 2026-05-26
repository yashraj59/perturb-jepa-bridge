# BioAction-JEPA Phase 2 Amendment: BioTechnical Disentanglement Before Reopening Architecture Search

## Purpose

You are amending the previous BioAction-JEPA autoresearch loop after its clean stop condition.

The prior loop succeeded at implementing a real JEPA identity, but it did **not** produce a promotable model. The blocker is not "insufficient training" and not "JEPA identity." The blocker is that the learned shared/joint latent remains highly predictive of technical batch in exact and held-out-dose settings, and a direct batch-centroid invariance penalty did not fix it.

Your task is to open a **diagnostic-first Phase 2** focused on biological/technical disentanglement for a real JEPA model.

Do not launch another long version of the same Family A or Family F architecture. Do not promote anything unless the protected gates pass.

---

## Model Of Record

The protected model of record remains:

```text
rank-3 train-split-only PLS raw-linear readout
RNA readout: raw_linear_pseudobulk
Image readout: raw_linear_pooled
```

The prior BioAction-JEPA candidates are audit references only. They are not the model of record.

Family N and Family O remain expression/count-aware audit references only.

Only a Tier 3 pass can supersede the protected model of record.

---

## What The Previous Loop Established

Read these files first:

```text
final_report.md
architectural_changes_log.md
external_resources.md
outputs/autoresearch_bioaction_jepa_v1/step0_baselines/SUMMARY.md
outputs/autoresearch_bioaction_jepa_v1/BASELINE_REGISTRY.md
outputs/autoresearch_bioaction_jepa_v1/results.tsv
outputs/autoresearch_bioaction_jepa_v1/research_journal.md
```

Required facts to preserve:

```text
BioAction-JEPA identity: implemented and passed identity checks.
No BioAction candidate promoted.
No Tier 2 or Tier 3 BioAction run launched.
Stop condition: batch leakage dominated latent state across Family A and Family F.
Family F batch-centroid invariance at weights 0.5 and 5.0 did not fix the failure mode.
Protected PLS remains model of record.
```

The previous strongest useful signal:

```text
EXP001 held-out perturbation:
exact match fraction = 0.0
RNA->image recall@1 = 0.0625 vs zero-step 0.0000
image->RNA recall@1 = 0.09375 vs zero-step 0.03125
transition-source cosine improvement = +0.00968
joint batch probe balanced accuracy = 0.4787 vs majority 0.4063
```

The previous blocking failures:

```text
EXP002 held-out dose:
transition-source cosine improvement = +0.02624
RNA->image recall@1 = 0.03125 vs initial 0.0625
image batch probe balanced accuracy = 0.6524
joint batch probe balanced accuracy = 0.6316 vs majority 0.5000

EXP003 exact synth_micro/test:
exact match fraction = 1.0
transition-source cosine improvement = +0.04168
RNA->image recall@1 = 0.0000 vs initial 0.03125
joint batch probe balanced accuracy = 0.9412 vs majority 0.5313

EXP005 batch-invariance:
standalone joint batch probe remained 0.9412 vs majority 0.5313
RNA->image recall@1 remained 0.0000
```

---

## Phase 2 Thesis

The previous architecture tried to make a single shared latent both biologically predictive and batch-invariant. That is likely the wrong factorization.

A better model should explicitly allocate information into:

```text
z_bio  = biological state needed for cross-modal retrieval and perturbation transitions
z_tech = technical/acquisition state needed to explain batch, plate, imaging artifacts, library size, dropout, etc.
z_joint = controlled fusion, not a monolithic retrieval latent
```

The new model should not merely erase batch. It should give batch a legal place to go, then forbid that nuisance factor from driving biological retrieval and action-conditioned transitions.

The core novelty should be:

> A factorized, environment-aware, action-conditioned cross-modal JEPA that predicts technical-invariant biological teacher targets while separately modeling technical state.

Working name:

```text
BioTech-JEPA
```

---

## Literature Ideas To Extract

Do not implement whole papers. Extract one compatible mechanism at a time.

Record all papers in:

```text
outputs/autoresearch_biotech_jepa_phase2/papers_consulted.md
```

At minimum, record these families of ideas:

### JEPA / world models

- I-JEPA / V-JEPA: context encoders, EMA target encoders, stop-gradient latent targets, predictor queries, representation-space prediction.
- V-JEPA 2 / action-conditioned world models: action-conditioned transition prediction from current latent state to future/target latent state.

Extracted mechanism:

```text
Use perturbation/dose/time/cell-line as biological action tokens and train control z_bio + action -> perturbed teacher z_bio.
```

### Domain generalization and causal invariance

- Domain-adversarial learning / gradient reversal: useful baseline, but prior Phase 1 shows a simple batch-invariance loss is insufficient.
- Invariant Risk Minimization: use multiple environments and require stable predictors across environments rather than only making batch hard to decode.
- Causal/domain representation learning: learn invariant factors and environment-dependent factors separately.

Extracted mechanism:

```text
Do not adversarially scrub the entire latent. Split z_bio and z_tech. Make z_tech explain batch. Make z_bio satisfy environment-stable prediction constraints.
```

### Single-cell batch modeling

- scVI / scANVI / sysVI-style intuition: model biological and technical variation rather than treating batch as a nuisance to simply delete.
- Single-cell integration benchmarks: batch removal must preserve biological variation and not overmix conditions.

Extracted mechanism:

```text
Evaluate both batch removal and biological retention. A low batch probe is not enough if retrieval, direction, or program structure collapse.
```

### Perturbation prediction

- CellOT: perturbations are maps between control and perturbed cell-state distributions, not just mean shifts.
- GEARS / graph perturbation priors: perturbation actions should exploit gene/program graph structure, not only ID embeddings.
- CPA: factor perturbation, dose, time, and cellular context compositionally.

Extracted mechanism:

```text
Use latent transition JEPA as the main objective, with optional OT-style population alignment and graph/program action priors as secondary modules.
```

---

## Phase 2 Output Directory

Create:

```text
outputs/autoresearch_biotech_jepa_phase2/
```

Initialize or update:

```text
outputs/autoresearch_biotech_jepa_phase2/results.tsv
outputs/autoresearch_biotech_jepa_phase2/research_journal.md
outputs/autoresearch_biotech_jepa_phase2/architectural_changes_log.md
outputs/autoresearch_biotech_jepa_phase2/family_allocation.md
outputs/autoresearch_biotech_jepa_phase2/BASELINE_REGISTRY.md
outputs/autoresearch_biotech_jepa_phase2/papers_consulted.md
outputs/autoresearch_biotech_jepa_phase2/external_resources.md
outputs/autoresearch_biotech_jepa_phase2/identity_violations_considered.md
outputs/autoresearch_biotech_jepa_phase2/final_report.md
```

Do not overwrite Phase 1 outputs.

---

## Phase 2 Has Two Stages

### Stage 1: Diagnostic investigation only

Before changing the model architecture, run a data/model diagnostic audit.

Create:

```text
outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit/
```

Required files:

```text
INVENTORY.md
METHODS.md
SPLIT_AND_CONFOUNDING_AUDIT.md
RAW_SIGNAL_BATCH_AUDIT.md
TEACHER_TARGET_AUDIT.md
REPRESENTATION_AUDIT.md
REOPENING_DECISION.md
```

Stage 1 must answer:

1. Is batch intrinsically confounded with perturbation, dose, cell line, or split?
2. How much batch signal exists in raw RNA pseudobulk?
3. How much batch signal exists in raw pooled images?
4. How much batch signal exists in protected PLS latents?
5. How much batch signal exists at zero-step BioAction encoder initialization?
6. How much batch signal exists in Phase 1 trained online latents?
7. How much batch signal exists in Phase 1 EMA teacher targets?
8. Does batch leakage come from input data, target construction, batch sampling, loss geometry, or action conditioning?
9. Are there enough cross-batch replicates of the same biological key to learn invariant biological targets?
10. Are held-out perturbation controls falling back to train controls in a way that changes the batch distribution?

Do not modify production model code during Stage 1.

### Stage 2: Reopen model search only if Stage 1 passes

Architecture search may reopen only if `REOPENING_DECISION.md` says all of the following:

```text
1. There is measurable biological signal not fully explained by batch.
2. There are enough cross-batch biological anchors, or the audit defines a valid substitute.
3. Batch leakage source is identified well enough to test a targeted mechanism.
4. The proposed mechanism is not simply "increase invariance weight."
5. Required metrics and gates are updated with exact baselines.
```

If these are not satisfied, write `final_report.md` and stop.

---

## Stage 1 Diagnostic Tasks

### Task A: Split and confounding matrix

For each dataset/split:

```text
synth_micro/test
synth_heldout_perturbation_lite/test_heldout_perturbation
synth_dose_extrapolation_lite/test_heldout_dose
synth_batch_confound_lite/test
```

Compute contingency tables and Cramér's V / mutual information where possible:

```text
batch_id × perturbation_id
batch_id × dose
batch_id × cell_line_id
batch_id × condition_key
batch_id × split
batch_id × exact biological key
```

Also compute:

```text
cross_batch_replicate_count_per_bio_key
bio_keys_with_1_batch_only
bio_keys_with_2plus_batches
bio_keys_with_3plus_batches
fraction_of_eval_targets_with_cross_batch_train_anchor
fraction_of_eval_targets_with_only_same-batch_anchor
fraction_of_eval_targets_with_no_bio_anchor
```

### Task B: Raw signal batch probe

Train/evaluate simple probes on:

```text
raw RNA pseudobulk
raw observed counts pseudobulk
raw image pooled pixels
protected PLS RNA latent
protected PLS image latent
random Gaussian baseline
metadata-only baseline excluding batch
```

Use held-out rows for evaluation. Do not train probes on eval target labels beyond normal supervised probe labels.

Report:

```text
balanced accuracy
majority baseline
per-class recall
bootstrap CI if cheap
```

### Task C: Teacher target audit

Evaluate batch decodability of:

```text
RNA EMA teacher target
image EMA teacher target
joint EMA teacher target
RNA online z_bio / z_shared
image online z_bio / z_shared
joint online z_shared
transition predicted latent
```

For Phase 1 models, load checkpoints if available. If checkpoints are absent, explicitly state that the audit is limited to saved metrics and zero-step models.

### Task D: Technical duplicate / split-half ceiling

For each condition bag with enough cells, split cells into two half-bags and compute:

```text
RNA->RNA same-condition consistency
image->image same-condition consistency
RNA->image same-condition consistency
batch probe for half-bag embeddings
condition probe for half-bag embeddings
```

This tells whether cross-modal retrieval failure is due to architecture, sample noise, or impossible signal.

### Task E: Loss geometry audit

For Phase 1 training logs, estimate whether any loss dominated:

```text
unweighted_jepa_losses
weighted_jepa_losses
VICReg/Barlow/prototype losses
batch-centroid invariance loss
weighted_aux_to_main_ratio
gradient norms if available
```

If gradient norms were not logged, add logging for future runs but do not rerun Stage 1 only for gradient norms unless needed.

---

## Stage 2 Candidate Architecture: BioTech-JEPA

Implement only after Stage 1 reopening criteria pass.

### Core module

Create:

```text
perturb_jepa/models/biotech_jepa.py
```

with:

```python
@dataclass(frozen=True)
class BioTechJEPAConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int
    bio_dim: int
    tech_dim: int
    num_bio_prototypes: int
    num_tech_prototypes: int
    dropout: float
    ema_decay: float
    use_pls_bootstrap: bool
    pls_bootstrap_steps: int
    pls_bootstrap_final_weight: float
    use_environment_irm: bool
    use_tech_reconstruction: bool
    use_ot_population_loss: bool
    use_graph_action_prior: bool
```

Model outputs must include:

```text
rna_z_bio
image_z_bio
joint_z_bio
rna_z_tech
image_z_tech
joint_z_tech

rna_teacher_z_bio
image_teacher_z_bio
joint_teacher_z_bio
rna_teacher_z_tech
image_teacher_z_tech

rna_to_image_bio_prediction
image_to_rna_bio_prediction
joint_to_rna_bio_prediction
joint_to_image_bio_prediction

control_to_perturbed_bio_prediction
control_to_perturbed_tech_prediction_optional

batch_logits_from_z_bio_for_probe_only
batch_logits_from_z_tech
condition_logits_from_z_bio_for_probe_only
condition_logits_from_z_tech_for_probe_only

z_bio_variance
z_tech_variance
bio_tech_cross_covariance
```

### Required identity

A real BioTech-JEPA must have:

```text
online/context encoders
EMA target encoders
stop-gradient teacher targets
latent-space prediction
target-query predictors
cross-modal latent prediction
action-conditioned transition prediction
z_bio / z_tech factorization
```

It is not enough to add an adversarial batch loss to the old model.

### Factorized latent contract

The encoders must produce separate biological and technical heads:

```python
z = encoder(...)
z_bio = bio_projection(z)
z_tech = tech_projection(z)
```

Rules:

```text
z_bio is used for retrieval, biological JEPA targets, and perturbation transition.
z_tech is used for batch/technical prediction, optional reconstruction, and nuisance accounting.
z_joint is never allowed to be the sole retrieval representation.
```

### Teacher target construction

Teacher targets should be stop-gradient and EMA-based, but the target itself should be biological-factor aware.

Use one of these in priority order, based on Stage 1 feasibility:

#### Option 1: Cross-batch consensus teacher

For biological keys observed in multiple batches:

```text
teacher_z_bio_consensus = mean or attention-pooled EMA teacher z_bio across batches for the same biological key
```

Student context from one batch predicts the consensus teacher target.

This makes the biological target less batch-specific.

#### Option 2: Environment-swap teacher

Given two examples with the same biological key but different batch:

```text
student context: batch A
teacher target: same biological key, batch B
```

The student predicts the target biological latent, not the target technical latent.

#### Option 3: Synthetic oracle teacher only for synthetic diagnostics

If using synthetic data with known `z_bio`, use it only for diagnostics or an explicitly marked oracle upper bound. Do not promote a model that relies on synthetic oracle labels unavailable in real data.

---

## BioTech-JEPA Losses

Create:

```text
perturb_jepa/training/biotech_losses.py
```

Required terms:

### 1. Intra-modal biological JEPA

```text
RNA masked context -> RNA teacher z_bio target
image masked context -> image teacher z_bio target
```

### 2. Cross-modal biological JEPA

```text
RNA z_bio -> image teacher z_bio
image z_bio -> RNA teacher z_bio
joint z_bio -> RNA teacher z_bio
joint z_bio -> image teacher z_bio
```

### 3. Action-conditioned transition JEPA

```text
control z_bio + perturbation_action -> perturbed teacher z_bio
```

The action should include:

```text
perturbation_id or descriptor
cell_line_id/context
dose
time
optional graph/program prior
```

### 4. Technical allocation loss

Make `z_tech` useful for technical factors:

```text
z_tech -> batch_id
z_tech -> library size / dropout proxy if available
z_tech -> image intensity/technical summary if available
```

This is not for promotion; it is to prevent `z_bio` from needing to store technical variation.

### 5. Biological retention loss

Make `z_bio` preserve biological structure:

```text
condition/perturbation/cell-line/dose probe metrics as diagnostics
program recovery
direction-of-effect metrics
cross-modal retrieval
transition-source improvement
```

Do not use exact biological-key one-hot as a model input.

### 6. z_bio nuisance suppression

Use adversarial/probe penalty only on `z_bio`, not on the whole latent.

Start with small weights and log:

```text
unweighted_batch_adv_loss
weighted_batch_adv_loss
batch_adv_to_jepa_ratio
z_bio_batch_probe
z_tech_batch_probe
```

### 7. Orthogonality / redundancy control

```text
cross_covariance(z_bio, z_tech)
Barlow/VICReg for z_bio
variance floor for z_bio and z_tech
```

### 8. Environment-stable prediction loss

For each batch environment, compute transition loss separately:

```text
loss_e = transition_jepa_loss(batch_environment=e)
IRM_penalty = variance or gradient penalty across environments
GroupDRO_weighted_loss = emphasize worst batch environment
```

Do not over-optimize a single average if one batch fails.

### 9. Optional OT population alignment

If cheap and stable:

```text
distribution(control_to_perturbed_bio_prediction) ≈ distribution(perturbed_teacher_z_bio)
```

Use MMD/sliced Wasserstein or Sinkhorn-like approximations. This is secondary to JEPA identity.

### 10. Optional count decoder

Optional and low-weight:

```text
z_bio + action + z_tech -> NB count mean/dispersion
```

The count decoder cannot become the main model identity. Its role is diagnostic and downstream biological evaluation.

---

## Training Schedule

Use staged training:

### Stage 2A: Warm start

```text
steps: small
PLS bootstrap allowed only on z_bio, not final retrieval
PLS bootstrap weight anneals to zero
tech allocation loss active
batch suppression on z_bio weak
```

### Stage 2B: JEPA main training

```text
high weight: latent JEPA
high weight: cross-modal JEPA
high weight: transition JEPA
medium weight: environment-stable prediction
medium weight: technical allocation
low weight: z_bio batch adversary
low weight: count decoder / reconstruction
```

### Stage 2C: Probe-only evaluation

Freeze model and train probes:

```text
batch probe on z_bio
batch probe on z_tech
condition probe on z_bio
condition probe on z_tech
cross-modal retrieval
transition null comparison
program/direction metrics
```

---

## Scripts

Add:

```text
scripts/run_biotech_batch_audit.py
scripts/train_biotech_jepa.py
scripts/evaluate_biotech_jepa.py
```

The audit script must work before model code changes.

All scripts must support:

```text
--dataset
--seed
--eval-split
--device
--output-root
```

`--eval-split` must not be hardcoded to `test`.

---

## Tests

Add focused tests:

```text
tests/test_biotech_batch_audit.py
tests/test_biotech_jepa_model.py
tests/test_biotech_losses.py
tests/test_biotech_condition_pairs.py
tests/test_biotech_eval_split.py
```

Tests must verify:

```text
z_bio and z_tech shapes
EMA teachers frozen
teacher targets stop-gradient
cross-modal predictions exist
transition prediction exists
z_bio used for retrieval
z_tech not used for retrieval
condition_key/biological_key not used as input
eval split respected
batch labels not passed into z_bio predictor except for probe/loss diagnostics
```

---

## Baselines And Reference Values

Use existing Step 0 baselines as fixed references.

Known references:

```text
synth_micro/test:
exact match fraction = 1.0
protected PLS RNA->image recall@1 = 0.28125

synth_heldout_perturbation_lite/test_heldout_perturbation:
exact match fraction = 0.0
protected PLS RNA->image recall@1 = 0.1852

synth_dose_extrapolation_lite/test_heldout_dose:
exact match fraction = 0.0
protected PLS RNA->image recall@1 = 0.1806
```

Previous BioAction references:

```text
EXP001 held-out perturbation RNA->image recall@1 = 0.0625
EXP001 held-out perturbation transition improvement = +0.00968
EXP001 joint batch probe = 0.4787 vs majority 0.4063

EXP002 held-out dose RNA->image recall@1 = 0.03125
EXP002 transition improvement = +0.02624
EXP002 joint batch probe = 0.6316 vs majority 0.5000

EXP003 exact test RNA->image recall@1 = 0.0000
EXP003 joint batch probe = 0.9412 vs majority 0.5313
```

If any raw metric files disagree with these values, resolve by reading raw per-run JSON/TSV. Do not silently choose convenient values.

---

## Tier Gates

### Stage 1 Reopening Gate

Only reopen architecture search if the audit identifies a plausible, targeted mechanism.

If audit says the split is fundamentally confounded and there are no cross-batch biological anchors, stop and recommend dataset/split redesign.

### Tier 1: Single-seed targeted test

Run only after Stage 1 reopening passes.

Use:

```text
dataset = synth_heldout_perturbation_lite
eval_split = test_heldout_perturbation
seed = 0 or the same seed used in Phase 1 reference
```

Tier 1 keep requires all:

```text
real BioTech-JEPA identity: pass
exact match fraction: 0.0
RNA->image recall@1 >= max(0.10, previous BioAction heldout perturbation recall@1 + 0.03)
image->RNA recall@1 improves over previous BioAction heldout perturbation
transition-source cosine improvement >= +0.01
z_bio batch probe balanced accuracy <= majority + 0.07
z_tech batch probe balanced accuracy >= z_bio batch probe + 0.10, unless raw batch signal is too weak
no collapse: z_bio variance floor passes
no retrieval collapse
all required diagnostics present
```

Tier 1 discard if:

```text
z_bio batch probe > majority + 0.15
RNA->image recall@1 <= previous BioAction by more than 0.02
transition improvement <= 0.0
z_tech does not capture batch while z_bio does
loss ratios show batch/adversarial losses dominate JEPA
```

### Tier 2: Multi-seed

Run only if Tier 1 passes.

Seeds:

```text
0, 1, 2
```

Datasets:

```text
synth_heldout_perturbation_lite/test_heldout_perturbation
synth_dose_extrapolation_lite/test_heldout_dose
```

Pass requires:

```text
mean RNA->image recall@1 beats previous BioAction references on both splits
mean transition-source cosine improvement positive on both splits
z_bio batch probe <= majority + 0.07 on both splits
no seed has severe retrieval collapse
standard deviation smaller than the claimed effect where possible
```

### Tier 3: Promotion evaluation

Only Tier 3 can supersede the protected PLS model.

Run:

```text
synth_heldout_perturbation_lite/test_heldout_perturbation seeds 0/1/2
synth_dose_extrapolation_lite/test_heldout_dose seeds 0/1/2
synth_batch_confound_lite/test seeds 0/1/2
synth_micro/test seeds 0/1/2 as audit only, not sole promotion basis
```

Promotion requires all:

```text
z_bio is the main retrieval/transition representation
PLS not used as final representation path
no condition_key/biological_key one-hot inputs
no test target leakage
held-out perturbation and dose beat protected PLS recall@1 or show a justified JEPA-specific improvement on transition metrics while preserving retrieval within 5% of PLS
transition metrics beat source-as-target and previous BioAction references
z_bio batch probe <= majority + 0.07 on held-out splits and <= majority + 0.10 on batch-confound audit
z_tech captures technical/batch signal better than z_bio
program/direction metrics do not regress versus relevant references
count decoder, if enabled, remains auxiliary
```

If the model improves transition but not retrieval, mark as:

```text
TIER2_PASS_HIGH_RISK_DO_NOT_PROMOTE_YET
```

Do not rebase.

---

## Architecture Families

### Family B1: Factorized Bio/Tech Latent

Hypothesis:

```text
Batch leakage persists because the old model has no explicit technical branch, forcing technical signal into the shared latent.
```

Mechanism:

```text
Separate z_bio and z_tech. Use z_bio for JEPA/retrieval/transition. Use z_tech for technical prediction and nuisance accounting.
```

Stop/pivot:

```text
Retire if z_tech does not capture batch and z_bio remains batch-predictive after two controlled variants.
```

### Family B2: Cross-Batch Consensus Teacher

Hypothesis:

```text
Teacher targets themselves are batch-contaminated, so students faithfully learn batch-contaminated representations.
```

Mechanism:

```text
Construct z_bio teacher targets from same-biological-key examples across batches, using consensus or environment-swap targets.
```

Stop/pivot:

```text
Retire if Stage 1 shows insufficient cross-batch anchors or if consensus targets collapse retrieval.
```

### Family B3: Environment-Stable Transition JEPA

Hypothesis:

```text
Average transition loss lets the model exploit batch-specific shortcuts.
```

Mechanism:

```text
Compute transition losses per batch environment and require stable prediction across environments via worst-group loss or IRM-like penalty.
```

Stop/pivot:

```text
Retire if worst-group objective suppresses all transition signal or produces high variance across seeds.
```

### Family B4: Graph/Program Action Prior

Hypothesis:

```text
Perturbation ID embeddings are too weak and encourage memorization. A program/graph action prior improves held-out perturbation generalization.
```

Mechanism:

```text
Map perturbation actions into gene programs or graph neighborhoods, then use those action embeddings in the transition predictor.
```

Stop/pivot:

```text
Retire if action prior improves exact split but not held-out perturbation.
```

### Family B5: Latent OT Population Transition

Hypothesis:

```text
Perturbation effects are distributional maps, not only condition-mean latent shifts.
```

Mechanism:

```text
Add secondary latent distribution matching between predicted perturbed z_bio population and teacher perturbed z_bio population.
```

Stop/pivot:

```text
Retire if OT loss dominates JEPA or collapses diversity.
```

---

## Do-Not-Run List

Do not run:

```text
longer Family A minimal BioAction-JEPA
stronger batch-centroid invariance weights without a new mechanism
exact synth_micro/test-only promotion
condition_key or biological_key one-hot model inputs
test target means or lookup tables as JEPA targets
PLS as the final JEPA representation path
batch labels as encoder inputs for z_bio
any run that lacks batch-probe diagnostics
```

---

## Required Diagnostics For Every Stage 2 Run

Log:

```text
rna_to_image_recall_at_1
rna_to_image_recall_at_5
image_to_rna_recall_at_1
image_to_rna_recall_at_5
transition_source_cosine_improvement
transition_null_comparison
z_bio_batch_probe_balanced_accuracy
z_tech_batch_probe_balanced_accuracy
z_bio_condition_probe_balanced_accuracy
z_tech_condition_probe_balanced_accuracy
majority_baselines_for_all_probes
z_bio_variance
z_tech_variance
z_bio_z_tech_cross_covariance
VICReg_terms
Barlow_terms
unweighted_loss_terms
weighted_loss_terms
aux_to_jepa_loss_ratios
per_batch_environment_transition_losses
worst_batch_environment_loss
collapse_flags
exact_match_fraction
```

Biology-specific metrics when available:

```text
program recovery
direction accuracy
delta cosine
logFC correlation
top50 DE overlap
pseudobulk correlation
NB NLL if count decoder enabled
Poisson NLL if count decoder enabled
technical duplicate ceiling
source-as-target null
global mean null
condition mean table reference
```

---

## Decision Labels

Use exact labels:

```text
PHASE2_AUDIT_COMPLETE_REOPEN
PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN
TIER1_KEEP_CONTROLLED_DISENTANGLEMENT_SIGNAL
TIER1_DISCARD_BATCH_LEAKAGE
TIER1_DISCARD_RETRIEVAL_COLLAPSE
TIER1_DISCARD_NO_TRANSITION_SIGNAL
TIER1_DISCARD_LOSS_DOMINATION
TIER2_PASS_CLEAN
TIER2_PASS_HIGH_RISK_DO_NOT_PROMOTE_YET
TIER2_FAIL_BATCH_LEAKAGE
TIER2_FAIL_SEED_INSTABILITY
TIER2_FAIL_RETRIEVAL_OR_TRANSITION_REGRESSION
TIER3_PASS_NEW_MODEL_OF_RECORD
TIER3_FAIL_USEFUL_FAILURE
SEARCH_CLOSED_NO_NEW_BASELINE
```

---

## Stop Conditions

Stop immediately and write `final_report.md` if any of these happen:

```text
Stage 1 audit shows no valid cross-batch or substitute invariant anchors.
Two consecutive Stage 2 variants have z_bio batch probe > majority + 0.15.
Two consecutive Stage 2 variants collapse RNA->image recall@1 to <= 0.02.
Any candidate uses condition_key/biological_key one-hot inputs.
Any candidate trains on eval/test target rows.
Loss audit shows non-JEPA auxiliary losses dominate the objective and the architecture relies on them.
All B1-B5 families have one controlled failure each with the same batch-leakage pattern.
Compute budget is exhausted.
```

When a stop condition fires, do not launch another experiment. Write the final report and stop.

---

## Launch Sequence

1. Create branch:

```bash
git checkout -b autoresearch/biotech-jepa-phase2
```

2. Create output directory and logs.

3. Run Stage 1 audit:

```bash
python scripts/run_biotech_batch_audit.py \
  --datasets synth_micro synth_heldout_perturbation_lite synth_dose_extrapolation_lite synth_batch_confound_lite \
  --eval-splits test test_heldout_perturbation test_heldout_dose test \
  --seeds 0 1 2 \
  --device cpu \
  --output-root outputs/autoresearch_biotech_jepa_phase2/batch_disentanglement_audit
```

4. Write `REOPENING_DECISION.md`.

5. Only if reopening passes, implement BioTech-JEPA.

6. Run smoke tests:

```bash
pytest \
  tests/test_biotech_batch_audit.py \
  tests/test_biotech_jepa_model.py \
  tests/test_biotech_losses.py \
  tests/test_biotech_condition_pairs.py \
  tests/test_biotech_eval_split.py
```

7. Run Tier 1 only.

8. Apply decision label and update all logs.

---

## Final Instruction

This Phase 2 is not a generic "make it better" search.

The research question is:

> Can a real cross-modal, action-conditioned JEPA learn a biological latent that predicts perturbation transitions while explicitly separating technical batch state into a different latent branch?

If the answer is no on the current synthetic benchmark, stop and recommend dataset/split redesign or real-data pretraining rather than continuing architecture variants.
