# Feature-Level Perturbation Forecaster Validation Plan

This document describes how to turn the current `ProgramBootstrapJEPA` path into
a scientifically defensible product proof for a feature-level perturbation
forecaster:

```text
control biology + perturbation descriptor -> predicted treated phenotype feature
```

The target product claim is intentionally narrow:

```text
The model can rank perturbations or perturbation-dose conditions by their
predicted feature-level phenotype, helping scientists choose which wet-lab
experiments to run next.
```

Do not claim raw gene-count generation, raw image generation, clinical utility,
or autonomous biological discovery from the current model. Those require raw
decoders, uncertainty calibration, and separate external validation.

## Current Model Contract

The working real-data path is the F096/F082 family:

```text
ProgramBootstrapJEPA
+ non-exact perturbation/action descriptors
+ train-only delta calibration
+ leakage-safe baselines and floor comparison
```

Current inputs are condition-level features:

```text
control_expression: source/control RNA or expression feature vector
target_expression: treated RNA or expression feature vector
target_image_flat: treated image/morphology feature vector
source_pca_target: train-only source image/protein/morphology latent
pca_target: train-only target image/protein/morphology latent
action: compound, guide, dose, or perturbation descriptor
metadata: split, treatment, control, batch, replicate, and grouping fields
```

Current outputs are latent predictions and metrics:

```text
predicted treated latent transition
transition improvement
delta cosine
recall@1 / retrieval
bidirectional cross-modal retrieval where available
delta rank and magnitude ratio
identity and leakage flags
```

The model is useful when the product proof is phrased as feature-level
prioritization, not as raw profile generation.

## Dataset Priority

### 1. scGeneScope: development proof, not fresh promotion

Use scGeneScope to reproduce that the current bridge can run on real paired
scRNA plus imaging features. It is the closest strict match to the original
scientific goal, but it already guided F093/F094/F095 repairs, so F096 is
non-promoting evidence.

Use it for:

```text
pipeline sanity checks
feature-table contract checks
train-only calibration behavior
comparison against the known F096 pass
```

Do not use it as:

```text
fresh external confirmation for F096
final product proof
```

### 2. cpg0003 Rosetta: product-style auxiliary benchmark

Use cpg0003 Rosetta to test the practical product question: can the model rank
compound-dose perturbations from feature-level transcriptomic and morphology
profiles?

Why it is useful:

```text
Cell Painting morphology profiles
L1000 expression profiles
shared compound-dose conditions
public preprocessed feature files
small enough CDRPBIO-BBBC036-Bray subset for fast iteration
```

Important caveat: cpg0003 is L1000 plus Cell Painting, not scRNA plus imaging.
The registered F097/F098/F100/F101 Rosetta runs did not promote the model. Treat
Rosetta as an auxiliary product benchmark and stress test unless a new,
predeclared product-ranking endpoint passes.

Recommended product endpoint for Rosetta:

```text
Given train compounds and their L1000/Cell Painting profiles, rank held-out
compound-dose conditions by predicted morphology or transcriptomic effect.
```

### 3. Full Rosetta or LINCS/Cell Painting overlap: stronger product demo

After the small CDRPBIO subset is understood, expand to a larger overlap such as
full cpg0003 or LINCS/Cell Painting resources. This is the best direction for a
commercial screening-assistant demo because the product value is compound
prioritization.

Required before using a larger resource:

```text
verified compound/dose/time/cell-line alignment
explicit controls
replicate metadata
license and commercial-use review
sealed held-out compound split
batch/plate leakage audit
```

### 4. PerturbMulti or another strict paired scRNA plus imaging dataset

For the strongest scientific claim, use a fresh strict paired dataset. The F102
artifact identifies `xingjiepan/PerturbMulti` as the current best public
candidate: public Hugging Face access, CC-BY-4.0 according to the local F102
report, RNA H5AD, protein-intensity H5AD, perturbation metadata, spatial
coordinates, and per-cell image tar archives keyed by cell ID.

Required before any model run:

```text
RNA H5AD backed/obs-only inspection
protein H5AD backed/obs-only inspection
image tar member metadata sampling
cell-ID overlap proof across RNA, protein, and image files
sealed split design
quota/RAM/GPU check
```

If cell-ID or condition-ID overlap fails, document the block and do not train.

## Validation Hypotheses

Pre-register hypotheses before running a dataset:

```text
H1: The model predicts held-out treated phenotype features better than
    source-as-target.

H2: The model beats a protected train-only ridge/full-ridge audit floor on
    transition improvement, delta cosine, and recall@1.

H3: The model enriches true high-effect perturbations in the top K ranked
    predictions compared with random ranking and chemical-nearest-neighbor
    baselines.

H4: Predictions remain valid under held-out compound or held-out guide splits,
    not only replicate holdout.
```

Do not change gates after seeing evaluation results.

## Split Design

Use at least two split modes.

### Compound or guide holdout

This is the main scientific split.

```text
train: compounds/guides A
validation: compounds/guides B
test: compounds/guides C
```

The same compound or guide must not appear in train and test, even under a
different replicate, dose, well, or plate, unless the split is explicitly a dose
generalization test.

### Replicate or same-condition holdout

This is useful for debugging and product reliability, but it is weaker.

```text
train: some replicates of a condition
test: other replicates of the same condition
```

This split can prove implementation quality and replicate consistency. It cannot
alone prove held-out perturbation generalization.

### Batch or plate holdout

Use when metadata allows it.

```text
train: plates/batches A
test: plates/batches B
```

This catches batch leakage and source-state shortcuts.

## Metrics

Report all model metrics by split and by seed.

### Core transition metrics

```text
transition_improvement = cosine(predicted_target, real_target)
                         - cosine(source, real_target)

delta_cosine = cosine(predicted_target - source,
                      real_target - source)

recall@1 = fraction of predictions whose nearest target is the true condition
```

### Product ranking metrics

For the screening-assistant product claim, add:

```text
precision@K for high-effect perturbations
recall@K for high-effect perturbations
hit enrichment@K over random
mean reciprocal rank for matching treated phenotype
top-K overlap with experimentally strongest perturbations
```

Define high-effect perturbations from held-out observed data only after the
prediction scoring rule is frozen. For example:

```text
high_effect = top 10% by replicated treated-vs-control feature distance
```

### Reliability metrics

```text
bootstrap confidence intervals by perturbation
seed mean and worst seed
replicate consistency
wrong-action negative gap
delta effective rank
calibration curve for predicted effect magnitude
abstention coverage versus error
```

## Baselines

Always compare to:

```text
source_as_target
protected train-only ridge/full-ridge floor
raw uncalibrated JEPA
chemical nearest-neighbor baseline
train mean or control-only baseline
random ranking
```

The protected floor is an audit threshold. It is not a fallback candidate output.

## Pass Criteria

A product-style pass requires all of:

```text
identity_violation == 0
leakage_flag == 0
mean transition improvement > source_as_target on every eval split
mean delta cosine > source_as_target on every eval split
recall@1 >= protected floor on every registered eval split
hit enrichment@K > random and chemical-nearest-neighbor baselines
bootstrap 95% CI lower bound above zero for the primary product metric
```

A strict scientific promotion pass additionally requires:

```text
fresh external dataset not used in repair decisions
held-out perturbation split
no pooled train+test statistics
train-only PCA/scaling/calibration
successful audit against source_as_target and protected floor
written non-promotion decision if any registered gate fails
```

## Leakage Rules

Forbidden as model inputs:

```text
condition_key
biological_key
exact treatment one-hot for held-out targets
held-out target means
pooled train+test statistics
batch/plate/well/site fields unless explicitly tested as non-biological nuisance inputs
floor predictions as candidate model inputs
```

Allowed for grouping and scoring only:

```text
treatment labels
compound IDs
guide IDs
condition IDs
replicate IDs
plate/well/site IDs
```

Fit these on train only:

```text
feature scaling
PCA
control/source centroids
delta calibration
baseline models
selection gates
abstention thresholds
```

## Implementation Steps

### Step 0: Verify repository and hardware

```bash
git checkout dev
python -m pip install -e ".[data,dev]"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest \
  tests/test_total_autonomy_council.py \
  tests/test_scgenescope_adapter.py \
  tests/test_synthetic_biology_lite.py \
  tests/test_program_bootstrap_jepa.py \
  tests/test_bioguard_wm_calibration.py -q

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits || true
df -h /content
free -h
```

### Step 1: Reproduce the known scGeneScope real-data path

Use this to confirm the model still runs.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/run_f082_scgenescope_external_validation.py \
  --raw-dir data/raw/scgenescope \
  --output-dir outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_f096_rerun \
  --report-path outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_f096_rerun/report.md \
  --device cuda \
  --seeds 37 38 39 \
  --steps 120 \
  --no-download-missing \
  --descriptor-mode pubchem_fingerprint \
  --gate-mode calibrated \
  --reuse-condition-cache \
  --condition-cache-dir outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_f082_scgenescope
```

Expected use: development reproducibility only. Do not promote from this run.

### Step 2: Run the Rosetta auxiliary product benchmark

Download the small CDRPBIO Rosetta files if absent:

```bash
mkdir -p data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray/CellPainting \
  data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray/L1000

curl -L --fail --retry 3 --retry-delay 2 \
  -o data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz \
  https://cellpainting-gallery.s3.amazonaws.com/cpg0003-rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz

curl -L --fail --retry 3 --retry-delay 2 \
  -o data/raw/cpg0003_rosetta/CDRPBIO-BBBC036-Bray/L1000/replicate_level_l1k.csv.gz \
  https://cellpainting-gallery.s3.amazonaws.com/cpg0003-rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/L1000/replicate_level_l1k.csv.gz
```

Reproduce the same-condition replicate run:

```bash
HF_HOME=/content/hf_cache python scripts/run_cpg0003_rosetta_external_confirmation.py \
  --experiment-id F098 \
  --split-mode same_condition_replicate \
  --device cuda \
  --seeds 37 38 39 \
  --steps 120 \
  --output-dir outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F098_cpg0003_rosetta_replicate_holdout \
  --report-path outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F098_cpg0003_rosetta_replicate_holdout/F098_CPG0003_ROSETTA_REPLICATE_HOLDOUT.md \
  --download-missing
```

Reproduce the small-scale calibration run:

```bash
HF_HOME=/content/hf_cache python scripts/run_cpg0003_rosetta_external_confirmation.py \
  --experiment-id F101 \
  --split-mode same_condition_replicate \
  --source-state control_centroid \
  --delta-calibration-mode train_small_scale \
  --device cuda \
  --seeds 37 38 39 \
  --steps 120 \
  --output-dir outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F101_cpg0003_rosetta_small_scale_calibration \
  --report-path outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F101_cpg0003_rosetta_small_scale_calibration/F101_CPG0003_ROSETTA_SMALL_SCALE_CALIBRATION.md \
  --download-missing
```

Interpretation:

```text
If F101 still fails transition improvement, do not change the architecture in
place. Record the failure as a source-state or validator-mismatch result and use
the dataset to design a product-ranking endpoint, not a promotion claim.
```

### Step 3: Run strict paired dataset preflight

Use PerturbMulti or a similar strict paired dataset only after metadata proves
the contract. The local F102 report defines the minimum safe sequence:

```text
download RNA H5AD to ignored storage
inspect obs with backed/HDF5 access only
confirm perturbation, coordinates, cell IDs, and split-usable metadata
sample image tar member names only
prove RNA/protein/image cell-ID or condition-ID overlap
design sealed split
then train on GPU
```

No model run is allowed before the preflight proves pairing.

### Step 4: Add product-ranking tables

For each dataset, write these artifacts:

```text
condition_features.tsv or .npz
condition_metadata.tsv
sealed_split_metadata.tsv
baseline_metrics.tsv
model_seed_split_metrics.tsv
product_ranking_metrics.tsv
external_summary_metrics.tsv
validation_report.md
```

The product-ranking table should include:

```text
split
seed
condition_id
treatment
dose
observed_effect_size
predicted_effect_size
predicted_target_similarity
true_target_rank
is_high_effect
rank_by_model
rank_by_baseline
```

## Adapting New Brightfield Paired Data

For a user-provided brightfield dataset, first make it feature-level.

Required metadata:

```text
sample_id
condition_id
modality
treatment
dose
timepoint
cell_line or donor
batch
plate
well
site
replicate
is_control
split
image_path
rna_path or expression_sample_id
```

Feature extraction:

```text
brightfield image -> DINOv2/BioMedCLIP/ResNet/CellProfiler feature vector
expression data -> scVI/PCA/pseudobulk feature vector
perturbation -> compound fingerprint, SMILES hash, guide target, or dose vector
```

Minimum condition-level table:

```text
condition_id
split
treatment
dose
is_control
expression_feature_vector
image_feature_vector
action_descriptor
technical metadata for audit only
```

Only after this feature-level bridge passes held-out validation should raw
gene-expression or raw-image generators be built.

## Scientific Report Template

Every validation report should answer:

```text
What dataset was used?
What exact files and versions were used?
What was the predeclared split?
What was the model input contract?
Which metadata fields were forbidden as inputs?
Which train-only transforms were fit?
Which baselines were compared?
What were the registered pass/fail gates?
What failed, if anything?
Is the result promotion-eligible or non-promoting?
What exact command reproduces the run?
```

The decision language should be one of:

```text
PASS_PRODUCT_RANKING_NON_PROMOTING
PASS_FRESH_EXTERNAL_TIER3
FAIL_PRODUCT_RANKING_NO_PROMOTION
FAIL_EXTERNAL_NO_PROMOTION
FAIL_PREFLIGHT_NO_MODEL_RUN
```

## External Sources To Verify Before Publication

- cpg0003 Rosetta paper and data: https://www.nature.com/articles/s41592-022-01667-0
- cpg0003 Rosetta code/data notes: https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods
- Cell Painting Gallery registry: https://broadinstitute.github.io/cellpainting-gallery/complete_datasets.html
- LINCS Cell Painting repository: https://github.com/broadinstitute/lincs-cell-painting
- scGeneScope report/page: https://openreview.net/forum?id=918POZbZ50
- PerturbMulti dataset card: https://huggingface.co/datasets/xingjiepan/PerturbMulti
- PerturbMulti PubMed record: https://pubmed.ncbi.nlm.nih.gov/40513557/

Before commercial use, verify license terms for every raw and processed file,
not only the paper or code repository.

## Definition Of Done

The product proof is scientifically defensible when:

```text
1. A fresh dataset is selected before model changes.
2. Pairing and split metadata are verified without loading full payloads into RAM.
3. The model beats source_as_target and protected train-only baselines.
4. Product ranking metrics show top-K hit enrichment on held-out perturbations.
5. Bootstrap confidence intervals support the primary metric.
6. Leakage and identity checks are zero.
7. Failures are reported as failures, not reframed after the fact.
8. Exact commands, data paths, code version, and reports are committed.
```

Until then, the honest product status is:

```text
working feature-level perturbation-forecasting prototype, not yet a validated
biological discovery engine.
```
