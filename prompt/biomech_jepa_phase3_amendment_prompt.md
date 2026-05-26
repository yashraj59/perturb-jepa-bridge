# Phase 3 Codex Amendment: BioMechanistic-JEPA

## Purpose

Build the next *real JEPA* candidate after BioTech-JEPA Phase 2. Do not train the previous model longer by default. The latest evidence says the implementation is a real JEPA and the `z_bio`/`z_tech` factorization is visible, but the transition gain is still small, image-to-RNA retrieval is dead on the synthetic genetic-anchor run, and Norman is RNA-only with no exposed batch metadata. The next step is not a bigger version of the same model. It is a targeted architectural amendment focused on **delta prediction, action/program structure, and population-level transition targets**.

## Active model of record

The protected model of record remains the rank-3 train-split-only PLS raw-linear readout.

No BioAction-JEPA or BioTech-JEPA candidate is promoted.

PLS may be used only as:

1. protected baseline;
2. audit reference;
3. short annealed bootstrap teacher with weight decayed to zero.

PLS must not be the final representation path and must not be described as JEPA.

## Latest known state

### BioAction-JEPA Phase 1

BioAction-JEPA passed real-JEPA identity checks but stopped because technical batch leakage dominated the latent state. Direct batch-centroid invariance weights `0.5` and `5.0` did not fix the failure mode. Do not reopen that exact mitigation.

### BioTech-JEPA Phase 2

BioTech-JEPA added separate `z_bio` and `z_tech` branches, online/context encoders, EMA target encoders, stop-gradient teacher latents, query-based JEPA predictors, cross-modal RNA/image latent prediction, and action-conditioned control-to-perturbed transition prediction.

The genetic-anchor audit reopened architecture search for a synthetic genetic setting because the corrected genetic perturbation framing has a valid substitute teacher structure:

```text
minimum exact held-out cross-batch train-anchor fraction: 0.0000
minimum train biological-key cross-batch anchor fraction: 1.0000
minimum held-out eval biological-key cross-batch replicate fraction: 1.0000
valid substitute for cross-batch teacher: true
max split-half RNA->image same-bio recall@1: 0.6389
max raw/protected batch-probe excess over majority: 0.6667
```

### BTJ001 synthetic genetic anchor

```text
RNA->image recall@1: 0.1875
image->RNA recall@1: 0.0000
transition-to-target recall@1: 0.4375
transition source cosine improvement: +0.0161
joint z_bio batch-probe accuracy: 0.1875
joint z_tech batch-probe accuracy: 0.4375
batch allocation gap: +0.2500
joint z_bio effective rank: 7.5103
decision: TIER1_DIAGNOSTIC_NO_PROMOTION
```

Interpretation: the factorization mechanism is real, because `z_tech` carries more technical signal than `z_bio`. However, the action-transition improvement is too small, and one cross-modal retrieval direction is completely broken.

### BTJ002 Norman RNA-only

```text
RNA-only diagnostic: 1.0
transition-to-target recall@1: 0.0625
transition-to-target median rank: 8.5
transition source cosine improvement: +0.0313
target z_bio effective rank: 7.4066
target z_bio std mean: 0.0183
target z_tech std mean: 0.1228
batch probe: unavailable
```

Norman is useful as a real genetic perturbation sanity check, but it is not a promotion benchmark because the processed h5ad is RNA-only, A549-only, has no exposed batch/acquisition metadata, and has no paired imaging.

## Research thesis for Phase 3

Current BioTech-JEPA predicts **absolute target state** too weakly and does not sufficiently exploit the structure of genetic perturbation actions. Phase 3 should make the core JEPA target be the **biological transition delta** induced by a genetic action:

```text
context = control biological state
action = perturbed gene or gene-pair descriptor
teacher target = stop-gradient biological state of perturbed condition
main prediction = target biological delta and/or target biological prototype set
```

The novel candidate is:

```text
BioMechanistic-JEPA = BioTech-JEPA + program/action tokens + delta-state JEPA + population prototype transition + cross-modal repair
```

Do not implement a generic autoencoder or contrastive-only model. The main identity must remain JEPA:

- online/context encoders;
- EMA target encoders;
- stop-gradient latent targets;
- query-based predictors;
- latent-space prediction rather than raw reconstruction as the main objective;
- action-conditioned control-to-perturbed prediction;
- cross-modal RNA/image latent prediction when imaging exists;
- no exact condition-key memorization.

## Cross-field inspirations to record in `papers_consulted.md`

Add or update literature notes before coding. Record what idea is extracted and how it maps to a concrete mechanism.

Use at least these anchors:

1. **I-JEPA**: target-query latent prediction instead of pixel reconstruction. Mapping: RNA program target queries and image region target queries.
2. **V-JEPA / V-JEPA 2**: latent predictive world model and action-conditioned prediction. Mapping: `control z_bio + perturbation action -> perturbed z_bio`.
3. **BYOL / DINO-style EMA teachers**: online network predicts slowly moving target representations. Mapping: stop-gradient RNA/image teacher encoders.
4. **VICReg / Barlow Twins**: anti-collapse and decorrelation. Mapping: variance/covariance/orthogonality diagnostics on `z_bio`, `z_tech`, action deltas, and predicted transition tokens.
5. **GEARS**: genetic perturbation action structure and combinatorial perturbations. Mapping: gene/gene-pair action descriptors and graph/program action encoder.
6. **CPA**: compositional perturbation effects. Mapping: factor action into gene effect, pair interaction, cell context, and optional dose only for chemical screens.
7. **CellOT / neural optimal transport**: population-level perturbation maps. Mapping: set/prototype transition loss between predicted target population and teacher target population.
8. **scVI/sysVI-style technical variation modeling**: model technical nuisance explicitly instead of hoping invariance removes it. Mapping: keep `z_tech` branch, but restrict retrieval and transition to `z_bio`.
9. **robotics/action world models**: action-token transition operators. Mapping: perturbation is the biological action; cell state is the world state; target condition is the future state.

## Hard constraints

1. Do not use `condition_key`, `biological_key`, exact target-key one-hot features, or test target means as model inputs.
2. Do not train on eval/test target rows.
3. Do not promote anything on exact `synth_micro/test`.
4. Do not use raw-linear PLS as the main representation path.
5. Do not treat Norman `dose_val` as chemical dose. For Norman, dose is fixed guide presence.
6. Do not treat Norman fixed synthetic batch id as real batch metadata.
7. Do not claim batch disentanglement on Norman unless real batch/acquisition metadata is recovered.
8. Do not launch Tier 2 unless every Tier 1 gate below passes.
9. Do not rescue a failed model by simply increasing loss weights or training steps.
10. All experiments must write full documentation and exact decision labels.

## Phase 3 output directory

Create:

```text
outputs/autoresearch_biomech_jepa_phase3/
```

Required files:

```text
outputs/autoresearch_biomech_jepa_phase3/research_journal.md
outputs/autoresearch_biomech_jepa_phase3/architectural_changes_log.md
outputs/autoresearch_biomech_jepa_phase3/results.tsv
outputs/autoresearch_biomech_jepa_phase3/family_allocation.md
outputs/autoresearch_biomech_jepa_phase3/BASELINE_REGISTRY.md
outputs/autoresearch_biomech_jepa_phase3/papers_consulted.md
outputs/autoresearch_biomech_jepa_phase3/identity_violations_considered.md
outputs/autoresearch_biomech_jepa_phase3/final_report.md
```

## Step 0: copy and lock current baselines

Before changing architecture, copy the latest Phase 2 numbers into `BASELINE_REGISTRY.md`.

Minimum registry entries:

```text
BTJ001_synth_genetic_anchor_seed0
BTJ002_norman_rna_only_seed0
protected PLS synthetic model of record
Family N expression-space reference
Family O count-aware reference
split-half RNA->image same-bio ceiling on synth_genetic_anchor_lite
raw/protected batch-probe excess values from the genetic-anchor audit
```

State explicitly that BTJ001 and BTJ002 are not promoted baselines; they are diagnostic references.

## Stage A: diagnostics before architecture changes

Run these diagnostics first. Do not change architecture until these are written.

### A1. Image branch health audit

Goal: explain why BTJ001 has `image->RNA recall@1 = 0.0000`.

Create:

```text
scripts/audit_biotech_image_branch.py
outputs/autoresearch_biomech_jepa_phase3/diagnostics/image_branch_health/REPORT.md
```

Compute on `synth_genetic_anchor_lite/test_heldout_perturbation`:

- image-only intra-JEPA target variance;
- image teacher `z_bio` effective rank;
- image teacher `z_tech` effective rank;
- image-to-RNA nearest-neighbor rank distribution;
- RNA-to-image nearest-neighbor rank distribution;
- image branch gradient norms by module;
- cross-modal loss contribution ratio;
- image branch collapse diagnostics;
- retrieval with frozen random encoders versus trained BTJ001 checkpoint if available;
- retrieval with teacher latents versus online latents.

Decision labels:

```text
IMAGE_BRANCH_AUDIT_HEALTHY
IMAGE_BRANCH_AUDIT_COLLAPSE
IMAGE_BRANCH_AUDIT_LOSS_IMBALANCE
IMAGE_BRANCH_AUDIT_DATA_OR_LOADER_ISSUE
```

If the audit indicates a data/loader issue, patch that first and rerun focused tests before architecture work.

### A2. Transition target audit

Goal: determine whether absolute target prediction is harder than delta target prediction.

Create:

```text
scripts/audit_biotech_transition_targets.py
outputs/autoresearch_biomech_jepa_phase3/diagnostics/transition_target_audit/REPORT.md
```

Compute for train and eval splits:

```text
z_control_teacher_bio
z_target_teacher_bio
delta_teacher = z_target_teacher_bio - z_control_teacher_bio
source-to-target cosine
source-to-delta cosine
nearest-neighbor rank using absolute target
nearest-neighbor rank using delta target
batch predictability of absolute target
batch predictability of delta target
variance/effective rank of delta target
```

Decision labels:

```text
DELTA_TARGET_HAS_HEADROOM
ABSOLUTE_TARGET_PREFERRED
TRANSITION_TARGET_COLLAPSED
TRANSITION_TARGET_BATCH_CONTAMINATED
```

If `DELTA_TARGET_HAS_HEADROOM`, implement delta JEPA. If not, write why and stop Phase 3 unless a loader bug is found.

### A3. Action descriptor audit

Goal: check whether synthetic and Norman actions contain enough information to generalize to held-out perturbations.

Create:

```text
scripts/audit_action_descriptors.py
outputs/autoresearch_biomech_jepa_phase3/diagnostics/action_descriptor_audit/REPORT.md
```

For synthetic genetic anchor, verify that every perturbation, including held-out perturbations, has a non-leaky action descriptor available independently of target expression. If the current synthetic action is only an integer perturbation id, add a non-leaky descriptor generator:

```text
gene multi-hot descriptor
program multi-hot descriptor
optional graph neighborhood descriptor
```

For Norman, reuse the gene multi-hot action descriptor. Do not use condition-key one-hot.

Decision labels:

```text
ACTION_DESCRIPTOR_VALID
ACTION_DESCRIPTOR_MISSING_FOR_HELDOUT
ACTION_DESCRIPTOR_LEAKY
ACTION_DESCRIPTOR_TOO_WEAK_DIAGNOSTIC_ONLY
```

Do not continue to Phase 3 architecture if held-out actions have no valid descriptor.

## Stage B: implement BioMechanistic-JEPA

Implement the smallest architecture that tests the thesis. Prefer extending the current BioTech-JEPA code if that avoids duplication, but keep the new mechanism clearly named.

### New or modified files

Expected files:

```text
perturb_jepa/models/biomech_jepa.py
perturb_jepa/training/biomech_losses.py
perturb_jepa/training/biomech_trainer.py
perturb_jepa/evaluation/biomech_metrics.py
scripts/train_biomech_jepa.py
scripts/evaluate_biomech_jepa.py
tests/test_biomech_jepa_model.py
tests/test_biomech_transition_targets.py
tests/test_action_descriptors.py
```

It is acceptable to share modules with BioTech-JEPA, but the model card must state exactly which implementation path is used.

### B1. ProgramActionEncoder

Add a real action encoder that does not depend on exact condition keys.

Inputs:

```text
gene_multi_hot: [batch, genes]
program_multi_hot: [batch, programs] optional
perturbation_type: genetic_single, genetic_pair, control, chemical optional
cell_context: cell line / cell type embedding when available
chemical_dose: only for chemical/drug datasets, never for Norman genetic guide notation
```

Outputs:

```text
action_global: [batch, action_dim]
action_tokens: [batch, num_action_tokens, action_dim]
action_program_logits: [batch, num_programs]
action_interaction_token: [batch, action_dim] for gene-pair synergy
```

For genetic pairs, include both additive and interaction components:

```text
action = f(gene_a) + f(gene_b) + g(gene_a, gene_b)
```

For single-gene perturbations:

```text
action = f(gene)
```

For control:

```text
action = zero or learned control token, but delta target should be near zero
```

Do not make held-out perturbation ids into learned embedding indices unless a descriptor is also provided and the id embedding is disabled for held-out generalization tests.

### B2. Delta-state JEPA

Add the main Phase 3 loss:

```text
z_control_bio_online = context encoder/control bag -> z_bio
z_target_bio_teacher = EMA target encoder/perturbed bag -> z_bio_teacher.detach()
z_control_bio_teacher = EMA target encoder/control bag -> z_bio_teacher.detach()
delta_teacher = stopgrad(z_target_bio_teacher - z_control_bio_teacher)
delta_pred = DeltaJEPAPredictor(z_control_bio_online, action_tokens)
z_target_pred = z_control_bio_online + delta_pred
```

Losses:

```text
L_delta_jepa = 1 - cosine(delta_pred, delta_teacher)
L_target_jepa = 1 - cosine(z_target_pred, z_target_bio_teacher)
L_delta_mse = smooth_l1(delta_pred, delta_teacher) with low weight
L_control_zero = ||delta_pred_control|| for control actions
```

The delta loss should be the main transition objective. The absolute target loss is auxiliary.

### B3. Population prototype transition JEPA

The model should predict not only one bag embedding but a set of target biological prototypes.

Use the existing multi-prototype aggregation idea. Add:

```text
source_bio_prototypes
teacher_target_bio_prototypes
predicted_target_bio_prototypes
```

Predict target prototypes with cross-attention from action tokens to source prototypes:

```text
predicted_target_prototypes = PopulationTransitionPredictor(
    source_bio_prototypes,
    action_tokens,
    query_tokens
)
```

Losses:

```text
prototype_cosine_jepa
sliced_wasserstein or OT-like set distance
prototype_diversity / repulsion
prototype_delta_consistency
```

This is inspired by population-level perturbation mapping: the model must learn how an action moves a distribution of cells, not just the condition mean.

### B4. Cross-modal repair

Because BTJ001 had `image->RNA recall@1 = 0.0000`, add a targeted cross-modal repair mechanism.

Required changes:

1. Normalize RNA and image teacher targets with the same target normalizer.
2. Add separate `rna_to_image_temperature` and `image_to_rna_temperature` or scaling if logits are used.
3. Add modality-balanced loss accounting so one modality cannot dominate.
4. Add modality dropout:

```text
RNA-only context -> image teacher target
image-only context -> RNA teacher target
joint context -> RNA/image teacher targets
```

5. Add an audit metric:

```text
cross_modal_gradient_ratio = image_branch_gradient_norm / rna_branch_gradient_norm
```

Hard rule: do not use raw image pooled features or raw-linear PLS as the main path to fix image retrieval.

### B5. Bio/tech separation retained

Keep separate `z_bio` and `z_tech`.

Use `z_bio` for:

```text
RNA/image retrieval
cross-modal JEPA
transition JEPA
delta prediction
population prototype prediction
Norman perturbation transition
```

Use `z_tech` for:

```text
batch/acquisition prediction when metadata exists
technical reconstruction or nuisance explanation
technical branch diagnostics
```

Add or keep orthogonality/decorrelation penalties:

```text
orthogonality(z_bio, z_tech)
VICReg variance floor on z_bio
VICReg variance floor on delta_pred
covariance off-diagonal reduction
```

Do not adversarially erase everything: the model should explicitly allocate technical signal into `z_tech`.

## Stage C: training commands and experiment sequence

Run small but meaningful experiments. Use exact decision labels. Do not skip documentation.

### BMJ001: delta-target synthetic only

Goal: test whether delta-state JEPA improves transition signal.

Command shape:

```bash
python scripts/train_biomech_jepa.py \
  --dataset synth_genetic_anchor_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cuda \
  --steps 80 \
  --eval-steps 8 \
  --batch-size 2 \
  --bag-size 4 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --num-condition-prototypes 4 \
  --enable-delta-jepa \
  --output-dir outputs/autoresearch_biomech_jepa_phase3/experiments/BMJ001_delta_synth_seed0 \
  --save-checkpoint
```

Gate:

```text
transition_source_cosine_improvement >= +0.0300
transition_to_target_recall@1 >= 0.4375
z_bio_effective_rank >= 6.0
condition_key_feature_present = 0.0
encoder_path_used = 1.0
```

### BMJ002: add ProgramActionEncoder

Goal: test whether non-leaky action descriptors improve held-out genetic generalization.

Run only if BMJ001 is not a hard failure.

Additional flags:

```text
--enable-program-action-encoder
--disable-perturbation-id-embedding-for-heldout-generalization
```

Gate:

```text
transition_source_cosine_improvement >= BMJ001 + 0.005
transition_to_target_recall@1 >= BMJ001
heldout_action_descriptor_valid = 1.0
```

### BMJ003: add population prototype transition

Goal: test whether set/prototype target prediction improves biological transition without collapsing condition means.

Additional flags:

```text
--enable-population-transition
--prototype-set-loss sliced_wasserstein
```

Gate:

```text
prototype_transition_cosine improves over BMJ002
transition_source_cosine_improvement >= +0.0350
z_bio_effective_rank >= 6.0
prototype_effective_rank >= 3.0
```

### BMJ004: cross-modal repair

Goal: fix `image->RNA recall@1 = 0.0000` without leaking batch into `z_bio`.

Additional flags:

```text
--enable-cross-modal-repair
--modality-dropout 0.25
--modality-balanced-loss
```

Gate:

```text
image_to_rna_recall@1 > 0.0000
rna_to_image_recall@1 >= 0.1875
z_tech_batch_probe_accuracy - z_bio_batch_probe_accuracy >= +0.1500
z_bio_batch_probe_accuracy <= z_tech_batch_probe_accuracy
```

### BMJ005: Norman RNA-only diagnostic

Goal: verify the gene action encoder and delta JEPA run on real Norman RNA-only data.

Command shape:

```bash
python scripts/train_biomech_jepa.py \
  --dataset norman \
  --norman-h5ad data/raw/gears_norman/norman/perturb_processed.h5ad \
  --eval-split test \
  --seed 0 \
  --split-seed 1 \
  --device cuda \
  --steps 40 \
  --eval-steps 8 \
  --batch-size 4 \
  --shared-dim 32 \
  --bio-dim 24 \
  --tech-dim 8 \
  --predictor-dim 64 \
  --gene-count 256 \
  --rna-only \
  --enable-delta-jepa \
  --enable-program-action-encoder \
  --output-dir outputs/autoresearch_biomech_jepa_phase3/experiments/BMJ005_norman_rna_only_seed0 \
  --save-checkpoint
```

Norman gate:

```text
encoder_path_used = 1.0
condition_key_feature_present = 0.0
rna_only_diagnostic = 1.0
transition_source_cosine_improvement >= +0.0313
z_bio_effective_rank >= 6.0
```

Norman cannot promote because it lacks imaging and exposed batch metadata.

### BMJ006: Tier 2 multi-seed only if gates pass

Run seeds `0,1,2` only if BMJ001-BMJ004 produce a controlled synthetic signal and BMJ005 is not a real-data hard failure.

Command shape:

```bash
for seed in 0 1 2; do
  python scripts/train_biomech_jepa.py \
    --dataset synth_genetic_anchor_lite \
    --eval-split test_heldout_perturbation \
    --seed "$seed" \
    --device cuda \
    --steps 160 \
    --eval-steps 16 \
    --batch-size 2 \
    --bag-size 4 \
    --shared-dim 32 \
    --bio-dim 24 \
    --tech-dim 8 \
    --predictor-dim 64 \
    --num-condition-prototypes 4 \
    --enable-delta-jepa \
    --enable-program-action-encoder \
    --enable-population-transition \
    --enable-cross-modal-repair \
    --modality-dropout 0.25 \
    --modality-balanced-loss \
    --output-dir outputs/autoresearch_biomech_jepa_phase3/experiments/BMJ006_synth_tier2_seed${seed} \
    --save-checkpoint
done
```

Tier 2 synthetic pass requires:

```text
mean transition_source_cosine_improvement >= +0.0350
mean transition_to_target_recall@1 >= 0.4375
mean image_to_rna_recall@1 > 0.0000
mean rna_to_image_recall@1 >= 0.1875
mean z_bio_effective_rank >= 6.0
mean batch_allocation_gap >= +0.1500
no seed has condition_key_feature_present > 0.0
no seed has encoder_path_used < 1.0
no seed has z_bio collapse
```

Do not promote after Tier 2. Tier 2 is permission to design Tier 3.

## Metrics to add or preserve

Add these metrics to `biomech_metrics.py` and log them in every eval JSON:

```text
encoder_path_used
pls_raw_linear_main_path_used
condition_key_feature_present
separate_bio_and_tech_latents_present
heldout_action_descriptor_valid
rna_to_image_recall@1
image_to_rna_recall@1
transition_to_target_recall@1
transition_to_target_median_rank
transition_source_cosine_improvement
delta_teacher_effective_rank
delta_pred_effective_rank
delta_cosine
absolute_target_cosine
prototype_transition_cosine
prototype_set_sliced_wasserstein
joint_z_bio_batch_probe_accuracy
joint_z_tech_batch_probe_accuracy
batch_allocation_gap
z_bio_effective_rank
z_tech_effective_rank
cross_modal_gradient_ratio
weighted_loss_to_main_loss_ratios
```

For Norman, set batch metrics to unavailable and explicitly state why.

## Loss contribution discipline

For every auxiliary loss, log:

```text
unweighted_loss
weighted_loss
weighted_to_main_ratio
cap_or_clip_fraction if applicable
```

Main transition loss should be delta JEPA. Reconstruction/count losses are auxiliary only.

Default loss priority:

```text
high: delta_jepa
high: target_transition_jepa
medium: cross_modal_jepa
medium: population_prototype_jepa
medium: VICReg / anti-collapse
low: raw reconstruction
low: count decoder
low: classification probes
```

If an auxiliary loss dominates the main delta JEPA loss, mark the run as:

```text
TIER1_DISCARD_AUXILIARY_DOMINATION
```

## Tests

Add focused tests before training:

```bash
pytest \
  tests/test_biomech_jepa_model.py \
  tests/test_biomech_transition_targets.py \
  tests/test_action_descriptors.py \
  tests/test_norman_biotech_batches.py \
  tests/test_synthetic_biology_lite.py::test_genetic_anchor_config_uses_fixed_dose_and_cross_batch_replicates
```

Required test coverage:

1. no condition-key input feature;
2. action descriptor exists for held-out perturbations;
3. `delta_teacher` is stop-gradient;
4. control action delta is near zero at initialization or after a short smoke fit;
5. `z_bio` and `z_tech` have expected shapes;
6. population prototype predictor returns `[batch, prototypes, bio_dim]`;
7. image branch health audit runs;
8. Norman loader still uses gene multi-hot action descriptor and fixed guide dose.

## Exact decision labels

Use only these labels unless a new label is documented in `results.tsv`:

```text
PHASE3_DIAGNOSTICS_COMPLETE_PROCEED
PHASE3_DIAGNOSTICS_STOP_DATA_OR_LOADER_ISSUE
PHASE3_DIAGNOSTICS_STOP_NO_DELTA_HEADROOM
TIER1_KEEP_DELTA_SIGNAL
TIER1_KEEP_ACTION_SIGNAL
TIER1_KEEP_POPULATION_SIGNAL
TIER1_KEEP_CROSS_MODAL_REPAIR
TIER1_DISCARD_NO_SIGNAL
TIER1_DISCARD_IMAGE_TO_RNA_STILL_ZERO
TIER1_DISCARD_BATCH_ALLOCATION_FAILURE
TIER1_DISCARD_COLLAPSE
TIER1_DISCARD_AUXILIARY_DOMINATION
NORMAN_RNA_ONLY_DIAGNOSTIC_PASS
NORMAN_RNA_ONLY_DIAGNOSTIC_NO_PROMOTION
TIER2_PASS_CONTROLLED_SIGNAL_DO_NOT_PROMOTE_YET
TIER2_FAIL_HIGH_VARIANCE
TIER2_FAIL_BATCH_OR_COLLAPSE
SEARCH_CLOSED_NO_NEW_BASELINE
```

## Stop conditions

Stop Phase 3 and write `final_report.md` if any of the following occurs:

1. image branch audit finds a data/loader bug that cannot be fixed with focused tests;
2. transition target audit shows no delta-target headroom;
3. held-out action descriptors are missing or leaky;
4. BMJ001 transition improvement is `< +0.0200`;
5. BMJ004 image-to-RNA recall remains exactly `0.0000` after the cross-modal repair attempt;
6. `z_bio_effective_rank < 4.0` in any promoted-to-next-stage candidate;
7. `z_bio` carries more batch signal than `z_tech` on synthetic batch-disentanglement evaluation;
8. any run uses condition-key features or test target rows;
9. two consecutive architecture variants fail through the same mechanism.

Stop means stop. Write the final report and do not launch another experiment.

## Final report requirements

When Phase 3 stops or completes, write:

```text
outputs/autoresearch_biomech_jepa_phase3/final_report.md
```

It must include:

- exact model-of-record decision;
- whether BioMechanistic-JEPA is still a real JEPA by identity checks;
- whether delta targets improved over absolute targets;
- whether action descriptors helped held-out perturbation generalization;
- whether population prototype transition helped;
- whether image-to-RNA retrieval recovered from zero;
- whether batch allocation remained correct;
- Norman diagnostic result and limitations;
- exact recommendation: close, amend, Tier 2, or Tier 3 design.

## One-sentence thesis to preserve

BioMechanistic-JEPA should be a biological action world model: it predicts how a genetic perturbation moves a population of cells in latent biological state space, while explicitly separating technical acquisition variation and retaining cross-modal RNA/image predictive structure.
