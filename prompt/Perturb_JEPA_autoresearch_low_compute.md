# Perturb-JEPA Autoresearch: Low-Compute Synthetic-Only Architectural Search

## Purpose

Run an autonomous, **low-compute** research loop to test whether the current Perturb-JEPA concept actually works on controlled synthetic scRNA-seq-like data before using real biological data.

The run must answer:

> Can a JEPA-style model with an scRNA-seq path learn meaningful latent cell-state structure, counterfactual perturbation response, batch robustness, and cross-modal alignment on synthetic data with known ground truth?

The answer may be yes, no, partially, or unclear because the synthetic testbed was flawed. Do not force a positive result.

---

## Non-Negotiable Constraints

### 1. No Real Data

Do not load, download, train on, validate on, or evaluate with real biological data.

Forbidden:

- CELLxGENE
- Tabula Sapiens
- PBMC datasets
- scPerturb
- JUMP Cell Painting
- public AnnData files
- real single-cell atlases
- real expression matrices
- real image manifests
- real marker-gene lists
- real pathway databases
- pretrained biological embeddings
- pretrained image backbones

Only procedurally generated synthetic data is allowed.

### 2. Preserve the Core Concept

Full modification is allowed, but the overall concept must remain:

- JEPA-style representation learning
- scRNA-seq input path
- synthetic perturbation / condition structure
- masked or predictive latent objective
- counterfactual or transition-style prediction

Do not convert the project into only:

- a plain autoencoder
- a classifier
- a VAE-only model
- a diffusion model
- a real-data benchmark
- a supervised cell-type predictor

These may be used as baselines, but not as the main model.

### 3. Protect the GPU

There is another important process using the GPU. This autoresearch run must be GPU-polite.

Default behavior:

- Prefer CPU for smoke tests, generator tests, metric code, and tiny runs.
- Use GPU only for short, bounded training jobs.
- Never launch parallel GPU training jobs.
- Never run multi-seed jobs concurrently on the GPU.
- Never increase dataset size, model width, image size, or training steps to chase better metrics unless explicitly justified in the journal.
- If the GPU is busy, reduce the run to CPU-compatible micro experiments or halt with a compute-budget note.

Before any CUDA run:

```bash
nvidia-smi || true
```

If another process is using substantial GPU memory or sustained GPU utilization, do not start a heavy run. Write:

```text
outputs/autoresearch_synth_lite/GPU_BUSY_SKIPPED_RUN.md
```

and either:

1. run the CPU-safe micro version, or
2. mark the experiment as `TIER*_FAIL_COMPUTE_BUDGET`.

Do not wait in the background. Do not queue long jobs.

---

## Model of Record

The active baseline is:

```text
perturb_jepa.models.bridge.PerturbJEPABridge
```

at the current HEAD commit of `main`.

Before launch:

1. Snapshot the commit hash.
2. Tag it:

```bash
git tag baseline/synthetic-lite-step0
```

3. Write:

```text
outputs/autoresearch_synth_lite/model_of_record.md
```

Include:

- commit hash
- current model files
- current training scripts
- current loss functions
- current config
- known assumptions
- current forward-pass output keys
- what must remain protected
- what is allowed to change

The protected core that must not be removed:

- JEPA training loop
- scRNA-seq path through `RNAEncoder`
- masked / predictive representation objective
- forward pass returning the existing output contract, including `rna_shared`, `rna_state`, `rna_response`, and counterfactual outputs
- compatibility with existing training and evaluation code

Everything else may be modified if the journal explains why.

Only a Tier 3 pass may replace the model of record.

Tier 1 keep does not rebase.

Tier 2 pass does not rebase.

---

## Output Directory

Use:

```text
outputs/autoresearch_synth_lite/
```

Create:

```text
outputs/autoresearch_synth_lite/
  README.md
  model_of_record.md
  compute_budget.md
  results.tsv
  research_journal.md
  architectural_changes_log.md
  synthetic_data_spec.md
  synthetic_generators_log.md
  family_allocation.md
  papers_consulted.md
  external_resources.md
  identity_violations_considered.md
  step0_baselines/
  experiments/
  insights/
  bugs/
  final_report.md
```

No experiment is valid unless it updates:

- `results.tsv`
- `research_journal.md`
- `architectural_changes_log.md`
- `family_allocation.md`

---

## Compute Budget

This run is intentionally small.

### Global Compute Limits

Hard caps:

```text
maximum_experiments: 32
maximum_tier1_wallclock_per_candidate: 15 minutes
maximum_tier2_wallclock_per_candidate: 45 minutes
maximum_tier3_wallclock_per_candidate: 2 hours
maximum_gpu_processes: 1
parallel_gpu_seeds: forbidden
maximum_model_dim_for_search: 128
maximum_latent_dim_for_search: 64
maximum_image_size_for_search: 16x16 or 24x24
maximum_genes_for_search: 512
maximum_cells_per_condition_for_search: 16
```

If the current repository config is larger, create a synthetic-lite config that preserves architecture type but uses smaller dimensions.

Preferred config:

```text
model_dim: 64 or 128
latent_dim: 32 or 64
num_attention_heads: 2 or 4
num_bag_slots: 2 or 4
rna_gene_count: 128 to 512
image_shape: 1 x 16 x 16 or 1 x 24 x 24
cells_per_condition: 8 to 16
batch_size: smallest stable value
precision: bf16 or fp16 only if stable
```

Do not use a large model to hide a weak objective.

### GPU Safety Rules

If CUDA is used:

- set only one training process at a time
- use mixed precision if stable
- use small batch size plus gradient accumulation
- enable early stopping
- save small checkpoints only
- delete failed checkpoints after metrics are extracted
- avoid storing full prediction arrays unless audit-marked

Optional safety code, if compatible:

```python
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)
```

If this causes errors, remove it and document why.

### Early Stopping

All training runs must support early stopping.

Stop early when both are true:

- validation objective has not improved for 20% of the allocated training steps
- collapse diagnostics are stable and not improving

Do not keep training only because more steps are available.

### Step Count Defaults

Use these as upper bounds, not goals:

```text
smoke_test_steps: 2 to 5
micro_debug_steps: 25 to 50
step0_micro_steps: 200
step0_lite_steps: 400 to 600
tier1_steps: 300 to 600
tier2_steps: 600 to 900
tier3_steps: 900 to 1200
```

If a model cannot show any signal on `synth_micro` within this scale, the issue is likely architecture/objective/data-generator design, not lack of compute.

---

## Files To Read Before Coding

Read these completely before changing code:

```text
perturb_jepa/models/bridge.py
perturb_jepa/models/rna_encoder.py
perturb_jepa/models/image_encoder.py
perturb_jepa/models/bag_aggregator.py
perturb_jepa/models/counterfactual.py
perturb_jepa/models/adversary.py
perturb_jepa/losses.py
perturb_jepa/distribution_losses.py
perturb_jepa/training/trainer.py
perturb_jepa/training/objectives.py
perturb_jepa/training/synthetic.py
perturb_jepa/config.py
perturb_jepa/evaluation/retrieval.py
perturb_jepa/evaluation/rna_counterfactual.py
perturb_jepa/evaluation/batch_probe.py
perturb_jepa/baselines/
```

Then read this `autoresearch.md` again.

---

## Infrastructure Verification

Before Step 0:

```bash
python scripts/train_smoke.py --steps 2 --device cpu
```

Then:

```bash
python scripts/train_synthetic.py --steps 5 --device cpu
```

If CPU does not support the exact script, use the smallest available safe device and document it.

All printed loss terms must be finite.

If infrastructure fails, halt and write:

```text
outputs/autoresearch_synth_lite/INFRASTRUCTURE_FAILURE.md
```

Do not begin architectural search until infrastructure works.

---

# Step 0: Synthetic Generator And Baselines

Step 0 is mandatory.

Do not run architectural variants until Step 0 is complete.

The goal is to create a tiny but meaningful synthetic world where the correct answer is knowable.

---

## Step 0.1: Build Synthetic Biology Generator

Create:

```text
perturb_jepa/training/synthetic_biology_lite.py
```

The generator must not read from disk or network.

It must generate synthetic condition bags compatible with the current training pipeline.

### Required Latent Structure

For each cell:

```text
z_bio = baseline[cell_line]
      + dose_curve(dose) * direction[perturbation]
      + interaction[cell_line, perturbation]
      + biological_noise
```

Technical nuisance:

```text
z_tech = batch_offset[batch] + library_size_effect + dropout_effect
```

Observed RNA:

```text
rna = fixed_decoder_rna(z_bio) + fixed_decoder_tech(z_tech) + count_noise
```

Observed image:

```text
image = reshape(fixed_decoder_image(z_bio) + fixed_decoder_image_tech(z_tech))
```

The decoders are fixed seeded functions, not learned.

### Synthetic scRNA-seq Properties

Include:

- count-like non-negative expression
- library-size variation
- dropout / zero inflation
- negative-binomial-like overdispersion if easy to implement
- gene programs
- perturbation affects programs, not arbitrary individual genes
- batch shifts
- control perturbation id 0
- dose response
- held-out perturbation split
- held-out dose split
- held-out batch split

### Ground Truth Must Be Saved

Save:

```text
ground_truth.json
generation_config.json
```

Include:

- latent biological state
- latent technical state
- perturbation direction matrix
- cell-line baseline matrix
- dose curve
- gene program assignment
- batch offsets
- clean RNA before technical corruption
- observed RNA after technical corruption
- split labels

### Unit Tests

Create tests proving:

- no real data is loaded
- same seed gives identical output
- different seed gives different output
- train/val/test cells do not leak
- held-out perturbations are held out
- held-out doses are held out
- held-out batches are held out
- batch-corrupted RNA differs from clean RNA
- synthetic control id exists in every relevant cell-line/batch group

---

## Step 0.2: Low-Compute Synthetic Datasets

Create locked configs under:

```text
outputs/autoresearch_synth_lite/step0_baselines/configs/
```

Use these datasets.

| Name | Role | Size | Purpose |
|---|---|---:|---|
| `synth_micro` | infrastructure + sanity | 4 perturbations, 2 cell lines, 2 doses, 2 batches, 8 cells/condition, 128 genes | should learn quickly |
| `synth_easy_lite` | primary search | 8 perturbations, 3 cell lines, 2 doses, 2 batches, 12 cells/condition, 256 genes | main low-compute signal |
| `synth_medium_lite` | secondary search | 12 perturbations, 3 cell lines, 3 doses, 3 batches, 12 cells/condition, 384 genes | harder but still small |
| `synth_heldout_perturbation_lite` | validator | built from medium, 3 perturbations held out | tests perturbation generalization |
| `synth_batch_confound_lite` | validator | perturbation correlated with batch | tests batch leakage |
| `synth_dose_extrapolation_lite` | validator | train doses `{0,1,3}`, eval doses `{2,5}` | tests dose generalization |

Rules:

- all datasets share the same seeded RNA and image decoders
- all datasets share the same control id
- `synth_easy_lite` and `synth_medium_lite` share baseline and direction matrices where possible
- no dataset has more than 512 genes or 16 cells per condition in this low-compute run
- image size should remain tiny: 16x16 preferred, 24x24 maximum

---

## Step 0.3: Baseline Models

Run these baselines before any architecture search.

### Required Baselines

1. `random_embedding`
2. `dataset_mean`
3. `source_as_target`
4. `metadata_only`
5. `batch_only`
6. `mean_prototype_alignment`
7. plain autoencoder, if already easy to implement
8. unmodified `PerturbJEPABridge`

The autoencoder is optional only if implementing it would consume too much time. If skipped, document why.

### Step 0 Training Budget

Use:

```text
synth_micro:
  seeds: 0
  max_steps: 200
  device: cpu preferred

synth_easy_lite:
  seeds: 0, 1
  max_steps: 400 to 600

synth_medium_lite:
  seeds: 0, 1
  max_steps: 400 to 600

validators:
  seeds: 0
  max_steps: 400 to 600
```

Do not run 3-seed Step 0 unless the GPU is idle and the total wallclock remains modest.

Tier 3 will use stronger validation only for candidates that deserve it.

### Step 0 Required Metrics

Compute:

#### Retrieval

- RNA to image recall@1
- RNA to image recall@5
- image to RNA recall@1
- image to RNA recall@5
- MAP
- median rank

#### Counterfactual

- pseudobulk correlation
- logFC correlation
- direction accuracy
- top-50 synthetic DE overlap
- program-level effect recovery

#### Ground-Truth Synthetic Recovery

- biological latent R² from `rna_shared`
- biological latent R² from `image_shared`
- perturbation direction cosine
- dose-response rank correlation
- cell-line baseline recovery
- program recovery score

#### Batch Robustness

- batch probe balanced accuracy
- batch probe balanced accuracy minus majority baseline
- biological latent R² on held-out batch
- retrieval drop on held-out batch

#### Collapse Diagnostics

- per-dimension std of `rna_shared`
- per-dimension std of `image_shared`
- embedding rank
- covariance spectrum
- student-teacher cosine mean/std
- counterfactual delta norm / state norm
- predictor output variance

#### Negative Controls

- label-shuffle retrieval
- batch-only retrieval
- metadata-only retrieval
- random embedding retrieval

If label-shuffle retrieval is materially above random, the evaluation is broken. Halt and fix it.

---

## Step 0.4: Step 0 Documentation

For every dataset, write:

```text
outputs/autoresearch_synth_lite/step0_baselines/<dataset>_baseline.md
```

Then write:

```text
outputs/autoresearch_synth_lite/step0_baselines/SUMMARY.md
```

The summary must include:

- table of all baselines
- mean ± std where multiple seeds exist
- current Perturb-JEPA performance
- comparison against random, batch-only, source-as-target, and metadata-only baselines
- which synthetic task is too easy
- which synthetic task is too hard
- whether the current model has signal
- exact Tier 1, Tier 2, and Tier 3 thresholds
- recommended first family to test

If `synth_micro` fails to beat random and batch-only baselines after infrastructure is verified, halt and write:

```text
outputs/autoresearch_synth_lite/BASELINE_NO_SIGNAL.md
```

Do not burn GPU trying larger datasets.

---

# What The Model Must Prove

The model must show evidence of all four:

1. It learns non-collapsed representations.
2. It learns biological latent structure better than batch-only shortcuts.
3. It predicts synthetic perturbation/dose effects better than source-as-target.
4. JEPA-style prediction helps beyond simple reconstruction or metadata baselines.

A model that only improves MSE is not enough.

A model that performs well because it learns batch is not enough.

A model that improves retrieval but fails counterfactual direction is not enough.

---

# Architectural Families

Run hypothesis-driven experiments only. Do not randomly change code.

Each experiment belongs to one family.

Because this is a low-compute run, prefer the cheapest variants first.

---

## Family A: JEPA Anti-Collapse And Target Stabilization

### Motivation

JEPA-style models can collapse or learn trivial student-teacher agreement.

### Hypothesis

Small representation-geometry constraints and better target normalization can prevent collapse without making the model heavy.

### Low-Compute Experiments

1. VICReg-lite variance loss on `rna_shared` and `image_shared`.
2. Covariance decorrelation on a random subset of dimensions only.
3. EMA teacher momentum schedule instead of fixed EMA.
4. Teacher output centering and normalization.
5. Predictor bottleneck: smaller predictor hidden size than encoder.
6. Stop-gradient placement ablation.
7. Normalized projection head before JEPA loss.
8. Multi-target JEPA: predict both cell-level and bag-level embeddings, but only on `synth_micro` first.

### Constraints

- No extra module may increase total parameters by more than 15%.
- New regularizer contribution must stay below 25% of total loss.
- If embedding rank drops, discard.
- Init must preserve baseline behavior when new loss weights are zero.

---

## Family B: Lightweight Bag Aggregator Replacements

### Motivation

Attention slots may collapse or overfit condition bags.

### Hypothesis

Simpler permutation-invariant aggregators may be more stable and cheaper.

### Low-Compute Experiments

1. Mean pooling only.
2. Mean + max pooling.
3. Mean + std pooling.
4. Mean + max + std concatenation followed by small MLP.
5. Deep Sets: per-cell MLP, mean, bag MLP.
6. Lightweight attention pooling with one query.
7. Slot orthogonality penalty.
8. Slot dropout during training.

### Constraints

- Must preserve current aggregator input/output shape.
- Parameter count must be ≤ baseline aggregator unless explicitly justified.
- Log slot utilization and attention entropy for attention-based variants.
- Any dead-slot variant is discarded unless retrieval and counterfactual metrics improve strongly without collapse.

---

## Family C: Low-Cost Cross-Modal Alignment

### Motivation

RNA/image alignment can be too weak or can overfit trivial condition metadata.

### Hypothesis

Multi-positive contrastive alignment and better temperature control can improve shared representation without heavy OT.

### Low-Compute Experiments

1. Multi-positive InfoNCE using condition id and perturbation id.
2. Learnable temperature clamped to `[0.03, 0.5]`.
3. Symmetric vs asymmetric RNA-image alignment weights.
4. Hard negatives: same perturbation, different cell line.
5. Alignment on bag embeddings only.
6. Alignment on cell embeddings only.
7. Remove alignment for first N warmup steps, then ramp.
8. Tiny Sinkhorn / OT alignment only after at least one Tier 2 pass from this family.

### Constraints

- No Sinkhorn or OT in Tier 1 unless dataset is `synth_micro`.
- Alignment loss contribution must not exceed 50% of total loss.
- If alignment improves retrieval but worsens biological latent R², discard.

---

## Family D: Counterfactual Delta Parameterization

### Motivation

The current counterfactual head may be too weak, too strong, or poorly matched to perturbation/dose structure.

### Hypothesis

Low-rank and dose-aware delta heads should recover the synthetic rank-structured perturbation effect.

### Low-Compute Experiments

1. Low-rank delta with rank 2.
2. Low-rank delta with rank 4.
3. Dose basis encoding using `[dose, log1p(dose), dose^2]`.
4. FiLM-style state conditioning from perturbation embedding.
5. Additive-only delta head.
6. Gate-only delta head.
7. Cell-line-conditioned delta gate.
8. Small residual MLP delta with spectral normalization.

### Constraints

- Delta must initialize near zero but not be a dead path.
- Do not zero-init both the gate and raw output.
- Log `counterfactual_delta_norm / state_norm`.
- If the delta head remains < 0.01 ratio for > 50% of steps, discard.
- Tier 1 must beat source-as-target on transition/counterfactual metrics.

---

## Family E: Batch-Effect Handling Without Killing Signal

### Motivation

Batch adversaries can remove useful biology when batch is confounded with perturbation.

### Hypothesis

Gentle batch-invariance mechanisms should reduce technical leakage while preserving perturbation and latent-state signal.

### Low-Compute Experiments

1. Batch-adversary ramp schedule.
2. Batch dropout: randomly mask batch labels when used.
3. Feature dropout on nuisance-sensitive dimensions.
4. MMD across batches on `rna_shared`.
5. Linear removal of batch direction from shared embedding.
6. Separate nuisance head and biology head.
7. Diagnostic-only positive batch classifier.
8. Conditional layer norm with tiny embedding.

### Constraints

- Batch leakage must decrease or remain stable.
- Retrieval must not drop by more than 10%.
- Biological latent R² must not drop by more than 10%.
- If batch probe improves only because representation collapses, discard.

---

## Family F: Synthetic Curriculum And Objective Scheduling

### Motivation

A model may fail not because the architecture is wrong, but because the synthetic task is introduced too abruptly or loss terms fight early.

### Hypothesis

A tiny curriculum can reveal whether the concept has signal without increasing compute.

### Low-Compute Experiments

1. Noise curriculum: start with low noise, ramp to target noise.
2. Mask-ratio curriculum: start low mask, ramp up.
3. Loss warmup: JEPA first, counterfactual later.
4. Loss warmup: RNA path first, then cross-modal alignment.
5. Freeze image encoder for first 20% of steps, then unfreeze.
6. Counterfactual-only warmup on `synth_micro`, then full model.
7. Batch-adversary delayed start.
8. Early stopping based on synthetic latent R² instead of total loss.

### Constraints

- This is not a hyperparameter sweep. Each schedule must test a specific failure mode observed in Step 0.
- No schedule may increase total steps beyond the tier budget.
- A curriculum must improve at least one target metric without weakening collapse or batch diagnostics.
- If the same architecture only works with a fragile schedule, mark high-risk.

---

## Family Allocation

Low-compute allocation:

```text
minimum_per_family: 3 experiments
maximum_per_family: 6 experiments
maximum_total_experiments: 32
```

Priority order after Step 0:

1. Family targeting the weakest Step 0 failure mode.
2. Family with cheapest next experiment.
3. Family with fewest experiments so far.
4. Family with highest current Tier reached.

Cross-family combinations are allowed only if both families have:

- at least 3 independent experiments
- at least one Tier 2 pass each

Combinations count against both family budgets.

---

# Tiered Evaluation

## Tier 1: Fast Single-Seed Filter

Purpose:

Reject bad ideas quickly.

Run:

```text
datasets: synth_micro, synth_easy_lite
seeds: 0
steps: 300 to 600 max
device: CPU if feasible, otherwise GPU only if idle
```

Evaluate:

- retrieval recall@1
- counterfactual direction accuracy
- biological latent R²
- batch leakage
- collapse diagnostics
- source-as-target comparison
- dataset-mean comparison

### Tier 1 Keep Criteria

All must hold:

1. At least one primary metric improves by ≥ 5% relative over Step 0 baseline.
2. No primary metric regresses by > 15%.
3. Model beats random embedding.
4. Transition/counterfactual metric beats source-as-target where applicable.
5. No representation collapse.
6. No JEPA collapse.
7. No dead delta head.
8. New loss term is not cap-bound.
9. Init-preserves-baseline smoke test passes.

### Tier 1 Labels

Use exactly one:

```text
TIER1_DISCARD_NO_SIGNAL
TIER1_DISCARD_MODE_COLLAPSE
TIER1_DISCARD_BATCH_LEAKAGE
TIER1_DISCARD_DEAD_DELTA
TIER1_DISCARD_CAP_BOUND
TIER1_DISCARD_METRIC_REGRESSION
TIER1_DISCARD_INIT_BROKEN
TIER1_DISCARD_COMPUTE_BUDGET
TIER1_KEEP_CONTROLLED_SIGNAL
```

Tier 1 keep does not rebase.

---

## Tier 2: Low-Compute Reproducibility Check

Purpose:

Check that Tier 1 was not seed noise.

Run only for `TIER1_KEEP_CONTROLLED_SIGNAL`.

```text
datasets: synth_easy_lite, synth_medium_lite
seeds: 0, 1
steps: 600 to 900 max
device: GPU only if idle, otherwise CPU-compatible reduced run
```

### Tier 2 Pass Criteria

All must hold:

1. Mean improvement ≥ 5% relative over Step 0 baseline.
2. Both seeds beat Step 0 baseline on the claimed primary metric.
3. No seed collapses.
4. No seed shows dead delta head.
5. No seed is cap-bound.
6. Batch leakage does not worsen by more than 0.05 balanced accuracy over majority-adjusted baseline.
7. Runtime stays within budget.

### Tier 2 Labels

Use exactly one:

```text
TIER2_PASS_CLEAN
TIER2_PASS_HIGH_RISK_DO_NOT_PROMOTE_YET
TIER2_FAIL_SIGNAL_NOT_REPRODUCIBLE
TIER2_FAIL_MODE_COLLAPSE
TIER2_FAIL_BATCH_LEAKAGE
TIER2_FAIL_DEAD_DELTA
TIER2_FAIL_CAP_BOUND
TIER2_FAIL_COMPUTE_BUDGET
```

Tier 2 pass does not rebase.

---

## Tier 3: Small No-Regression Gate

Purpose:

Promote only candidates that generalize across synthetic validators.

Run only for:

```text
TIER2_PASS_CLEAN
TIER2_PASS_HIGH_RISK_DO_NOT_PROMOTE_YET
```

Before Tier 3, check GPU status.

If GPU is busy, do not start Tier 3. Write:

```text
outputs/autoresearch_synth_lite/TIER3_SKIPPED_GPU_BUSY.md
```

and halt for user review.

Tier 3 run:

```text
datasets:
  synth_easy_lite
  synth_medium_lite
  synth_heldout_perturbation_lite
  synth_batch_confound_lite
  synth_dose_extrapolation_lite

seeds: 0, 1, 2
steps: 900 to 1200 max
runs: sequential only
```

Optional `synth_hard_lite` only if Tier 3 finishes under budget:

```text
synth_hard_lite:
  16 perturbations
  4 cell lines
  3 doses
  3 batches
  12 cells/condition
  512 genes max
  nonlinear decoder
  higher noise
```

### Tier 3 Pass Criteria

All must hold:

1. Tier 2 improvement preserved.
2. `synth_heldout_perturbation_lite` counterfactual direction accuracy ≥ Step 0 baseline × 0.95.
3. `synth_batch_confound_lite` batch leakage ≤ Step 0 baseline + 0.05.
4. `synth_batch_confound_lite` retrieval recall@1 ≥ Step 0 baseline × 0.90.
5. `synth_dose_extrapolation_lite` logFC correlation ≥ Step 0 baseline × 0.90.
6. Biological latent R² does not regress by > 10%.
7. No collapse on any seed.
8. No dead delta head.
9. No cap-bound loss.
10. Runtime stays within budget.

### Tier 3 Labels

Use exactly one:

```text
TIER3_PASS_NEW_BASELINE
TIER3_FAIL_USEFUL_FAILURE
TIER3_FAIL_GENERALIZATION
TIER3_FAIL_MODE_COLLAPSE
TIER3_FAIL_BATCH_LEAKAGE
TIER3_FAIL_DEAD_DELTA
TIER3_FAIL_COMPUTE_BUDGET
```

Only `TIER3_PASS_NEW_BASELINE` replaces the model of record.

If promoted:

```bash
git tag baseline/synthetic-lite-v2-<short_commit>
```

Then recompute the small Step 0 summary for the new baseline before continuing.

---

# Hard Fail Conditions

Disqualify a candidate regardless of metric improvement:

```text
representation collapse:
  min per-dim std of rna_shared or image_shared < 0.01

JEPA collapse:
  student-teacher cosine mean > 0.999 and std < 0.001

dead delta:
  counterfactual_delta_norm / state_norm < 0.01 for > 50% of non-control steps

eval broken:
  label-shuffle retrieval materially above random

cap-bound:
  new loss contribution at configured cap for > 50% of steps

universal regression:
  candidate regresses on >= 4 synthetic datasets

batch shortcut:
  retrieval improves but batch-only baseline explains most of the improvement

init broken:
  new module changes baseline outputs when disabled or identity-initialized

compute violation:
  run exceeds tier wallclock or launches heavy GPU work while GPU is busy
```

---

# Required Ablations For Tier 2 Or Tier 3 Candidates

Any candidate reaching Tier 2 must run small ablations on `synth_micro` or `synth_easy_lite`:

1. remove new component
2. shuffle perturbation labels
3. shuffle dose labels
4. shuffle batch labels
5. random target embeddings
6. source-as-target comparison
7. dataset-mean comparison

A good model should fail when the relevant structure is shuffled.

If performance does not change after shuffling the claimed signal, the candidate is using shortcuts.

---

# Results Table

`results.tsv` columns:

```text
commit
experiment_num
family
candidate_name
tier_reached
decision_label
device_used
wallclock_minutes
max_gpu_memory_gb
synth_micro_recall1
synth_easy_recall1
synth_medium_recall1
synth_easy_cf_dir_acc
synth_medium_cf_dir_acc
heldout_pert_cf_dir_acc
batch_confound_batch_leakage
batch_confound_recall1
dose_extrap_logfc_corr
bio_latent_r2
representation_rank
delta_norm_ratio
cap_bound
collapse_flag
architecture_change
description
```

---

# Research Journal Format

Every experiment entry:

```markdown
## Experiment <NNN>: <short title>

### Hypothesis

### Family

### Compute budget

### Device decision
- GPU status:
- Device used:
- Wallclock:
- Max GPU memory:

### Code changes

### Init-preserves-baseline test

### Synthetic datasets used

### Training setup

### Metrics vs Step 0

### Collapse diagnostics

### Batch diagnostics

### Counterfactual / transition diagnostics

### Program-level diagnostics

### Ablations, if applicable

### What improved

### What regressed

### Decision label

### Decision rationale

### Next action

### audit_relevant
```

---

# Architectural Change Log Format

Every change:

```markdown
## Change <NNN>: <short title>

### Family

### Files modified

### Motivation

### Exact mechanism

### Why this preserves JEPA

### Why this remains scRNA-seq relevant

### Initialization strategy

### Parameter count change

### Compute cost estimate

### Expected effect

### Observed effect

### Verdict
```

---

# Papers Consulted

This run is synthetic-only, but methodological papers may be consulted for ideas.

Allowed search areas:

- JEPA, I-JEPA, V-JEPA
- BYOL, SimSiam, DINO
- VICReg, Barlow Twins
- Deep Sets, Set Transformer
- CLIP-style contrastive learning
- multi-positive contrastive learning
- domain adversarial learning
- MMD batch alignment
- counterfactual representation learning
- low-rank adaptation
- neural ODEs
- FiLM conditioning
- synthetic benchmarks for representation learning

Do not use papers to add real biological priors.

Do not download datasets.

Do not add real gene sets.

For each paper:

```markdown
- Citation:
- Idea extracted:
- Family alignment:
- Low-compute adaptation:
- Used in experiment:
- Outcome:
```

---

# Insight Briefs

Every 8 experiments, write:

```text
outputs/autoresearch_synth_lite/insights/INSIGHT_BRIEF_008.md
outputs/autoresearch_synth_lite/insights/INSIGHT_BRIEF_016.md
outputs/autoresearch_synth_lite/insights/INSIGHT_BRIEF_024.md
outputs/autoresearch_synth_lite/insights/INSIGHT_BRIEF_032.md
```

Each brief must include:

1. Current best result.
2. Whether the JEPA concept still looks viable.
3. Which synthetic task is most informative.
4. Which family is producing signal.
5. Which family is wasting compute.
6. Collapse or shortcut patterns.
7. GPU/compute usage summary.
8. Next best low-compute experiments.
9. Whether to continue, tighten gates, or stop.

---

# Stop Conditions

Stop immediately when any condition fires.

Finish the current experiment, write the journal entry, write `final_report.md`, then halt.

Do not start a new experiment.

```text
1. hard cap: 32 experiments completed
2. success: 2 Tier 3 wins achieved
3. saturation: 8 consecutive Tier 1 discards
4. family exhaustion: all families hit max without Tier 3 win
5. structural failure: 6 experiments fail the same way across 3+ families
6. baseline no signal: unmodified Perturb-JEPA fails synth_micro sanity
7. compute pressure: GPU busy prevents Tier 2/Tier 3 and CPU fallback is too slow
8. repeated collapse: 4 collapse failures across 2+ families
9. user-directed stop
```

Do not ask "should I continue?" after a stop trigger.

Stop means stop.

---

# Mid-Session Amendments

If evidence requires changing the protocol, write:

```text
outputs/autoresearch_synth_lite/AMENDMENT_NNN.md
```

Required structure:

1. Direct instruction.
2. Active baseline.
3. Fixed Step 0 reference numbers.
4. What changed.
5. Family status.
6. Updated gates.
7. Updated compute budget.
8. Do-not-run list.
9. Immediate next experiment.
10. Stop/user-review trigger.

Use amendments when:

- Tier 1 keeps repeatedly fail Tier 2.
- one family is clearly exhausted.
- a metric is reward-hacked.
- synthetic generator is too easy or too hard.
- compute budget is being exceeded.
- GPU is repeatedly busy.

---

# Checkpoint And Disk Policy

Retain:

- markdown reports
- JSON/CSV/TSV metrics
- compact logs
- final configs
- Tier 3 winning checkpoints
- current model of record

Delete:

- Tier 1 failed checkpoints
- Tier 2 failed checkpoints unless audit-marked
- temporary prediction arrays
- large debug tensors
- duplicate checkpoints

Before deleting a near-miss checkpoint, mark whether it is audit-relevant in the journal.

Do not fill disk with failed runs.

---

# What This Run Is Not

This run is not:

- a real biology study
- a paper result
- a hyperparameter sweep
- a real-data benchmark
- a GPU-burning brute force search
- a foundation model pretraining run
- a proof that the model works on real scRNA-seq
- a reason to skip biological validation later

This is a synthetic proof-of-mechanism study under strict compute limits.

---

# Final Report

When stopping, write:

```text
outputs/autoresearch_synth_lite/final_report.md
```

Use this structure:

## 1. Executive Summary

Answer clearly:

```text
YES: synthetic Perturb-JEPA concept works
NO: synthetic Perturb-JEPA concept does not work
PARTIAL: some pieces work, but not enough
UNCLEAR: synthetic testbed or infrastructure failed
```

## 2. Compute Summary

Include:

- total experiments
- total GPU runs
- total CPU runs
- approximate wallclock
- max GPU memory used
- skipped runs due to GPU pressure

## 3. Active Model Of Record

Original baseline or Tier 3 winner.

Include commit hash.

## 4. Synthetic Generator Validity

Did the generator test:

- count noise
- dropout
- library size
- perturbation programs
- dose response
- batch effects
- held-out perturbations
- held-out doses
- cross-modal latent matching

## 5. Baseline Comparison

Compare final candidate against:

- random embedding
- dataset mean
- source-as-target
- metadata-only
- batch-only
- mean prototype
- autoencoder if implemented
- original Perturb-JEPA

## 6. Evidence For JEPA

State whether JEPA helped with:

- non-collapsed representation
- RNA/image alignment
- biological latent recovery
- counterfactual direction
- dose response
- batch robustness

## 7. Evidence Against JEPA

List failures:

- collapse
- dead delta
- batch shortcut
- no improvement over simple baselines
- poor seed stability
- compute too high
- generator too easy/hard

## 8. Family Findings

For each family:

- strongest experiment
- best metric
- failure mode
- whether to continue later

## 9. Recommendation

Choose one:

```text
PROCEED_TO_SMALL_REAL_DATA_PILOT
DO_NOT_USE_REAL_DATA_YET
REDESIGN_SYNTHETIC_GENERATOR_FIRST
JEPA_CONCEPT_WEAK_RECONSIDER_DIRECTION
PROMISING_BUT_NEEDS_MORE_SYNTHETIC_VALIDATION
```

## 10. Next Phase

Give the smallest safe next step.

If recommending real data later, specify:

- tiny dataset only
- no large training
- no claims
- same no-collapse/no-shortcut gates
- biological validation required

---

# Launch Instruction

When the user says:

```text
begin autoresearch
```

do the following:

1. Create branch:

```bash
git checkout -b autoresearch/synthetic-lite-v1
```

2. Tag current baseline:

```bash
git tag baseline/synthetic-lite-step0
```

3. Create `outputs/autoresearch_synth_lite/`.
4. Write `model_of_record.md`.
5. Write `compute_budget.md`.
6. Read all required files.
7. Run CPU smoke tests.
8. Build the synthetic generator.
9. Generate `synth_micro`.
10. Run Step 0 baselines on `synth_micro`.
11. If `synth_micro` has no signal, stop.
12. Generate the remaining lite datasets.
13. Complete Step 0 summary.
14. Pick the first family based on the weakest Step 0 failure mode.
15. Run Tier 1 experiments only within compute budget.
16. Escalate to Tier 2 only for clean Tier 1 keeps.
17. Escalate to Tier 3 only if GPU is idle and Tier 2 is strong.
18. Stop when any stop condition fires.

When in doubt, choose the cheaper experiment, the smaller dataset, the tighter gate, and the more conservative conclusion.
