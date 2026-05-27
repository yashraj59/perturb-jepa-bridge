# F082 External Validation Start Instructions

Use this prompt when starting Codex on another cluster for external validation.

## Current Actual Resume Point

Start from the pushed branch:

```bash
git checkout dev
```

The older setup said to start from `synthetic_pass`; that branch remains the
synthetic-pass baseline. The current external-validation handoff is `dev`.

Current status as of the latest handoff:

```text
F082 original scGeneScope external validation failed.
F094 split-safe calibration gate failed.
F095 PubChem fingerprint split-safe gate nearly passed but failed by recall.
F096 froze PubChem fingerprint descriptors plus train-only delta calibration and passed scGeneScope.
F096 is NON-PROMOTING because scGeneScope guided the repair loop.
Fresh external Tier 3 confirmation is still required before promotion.
Protected rank-3 train-split-only PLS raw-linear readout remains model of record.
```

The actual next step is F097:

```text
F097_CPG0003_ROSETTA_FRESH_CONFIRMATION_PREFLIGHT
```

Latest execution status:

```text
F097 cpg0003 Rosetta compound-holdout run failed external confirmation.
F098 cpg0003 Rosetta same-condition replicate-holdout run also failed external confirmation.
F098 had clean identity/leakage checks and positive transition/delta floor gaps,
but it missed the test recall floor and retained negative absolute transition
improvement. No promotion.
```

Use cpg0003 Rosetta CDRPBIO-BBBC036-Bray as the first fresh external
confirmation candidate:

```text
assays = Cell Painting morphology + L1000 expression
cell line = U2OS
shared exact compound+dose pairs = 1469
controls = DMSO/negcon in both modalities
Cell Painting replicates per shared pair = min 4, median 8, max 16
L1000 replicates per shared pair = min 1, median 2, max 2
SMILES missing among shared pairs = 0
```

This is a fresh perturbational transcriptomics+morphology validator, but it is
not scRNA. A pass must be reported with that caveat, and the promotion decision
must explicitly state whether strict paired scRNA+imaging confirmation is still
required.

If the ignored Rosetta files are absent, download only these small files:

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

Then implement/run a fresh confirmation runner that reuses the frozen F096 path,
not a redesigned architecture:

```text
ProgramBootstrapJEPA
non-exact chemical action descriptors from PubChem/SMILES plus dose
train-only split statistics, train-only image/CP PCA, train-only calibration
source-as-target, protected full-ridge floor, and no-residual baselines
transition improvement, delta cosine, recall@1, bidirectional retrieval, rank,
identity, and leakage checks
```

The runner now exists:

```bash
python scripts/run_cpg0003_rosetta_external_confirmation.py --help
```

Reproduce the latest F098 run:

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

Current post-F098 status:

```text
F099 artifact-only Rosetta source geometry audit:
  source-to-target cosine ~= 0.949 and repeated source effective rank ~= 1.
  Diagnosis = source-state contract failure plus validator mismatch.

F100 zero-signature source-state run:
  failed; negative transition improvement and unstable magnitude ratios.

F101 train-only small-scale delta calibration:
  selected residual scale 0.01, repaired delta cosine and recall floor gaps,
  but transition improvement stayed slightly negative on all splits.
```

Reproduce F101:

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

Do not redesign the model or promote anything from F097/F098/F100/F101. Resume
from `F102_STRICT_SCRNA_IMAGING_FRESH_DATASET_PREFLIGHT`: find or recover a
strict paired scRNA plus imaging fresh validation protocol, run only
metadata/manifest/obs-only/backed checks first, and document validation blocked
if no strict fresh paired protocol is available.

Use GPU for model work:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
```

Only use CPU if GPU is unavailable or already occupied.

## Original F082 Prompt

```text
Start from branch `synthetic_pass`.

Read these first:
- CODEX_CHAT_SUMMARY.md
- CODEX_RESUME_HANDOFF.md
- SYNTHETIC_PASS_SUMMARY.md
- REPORTS_INDEX.md
- outputs/autoresearch_total_autonomy_bioguard_wm_jepa/HARD_ESCALATION_REPORT.md
- outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md

Model-wise, validate F082, not F086/F087.

F082 is the actual best model candidate:
`F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`

Use the ProgramBootstrapJEPA / BioGuard-WM path:
- perturb_jepa/models/program_bootstrap_jepa.py
- perturb_jepa/models/bioguard_wm_jepa.py
- perturb_jepa/models/action_adaln_predictor.py
- perturb_jepa/models/jepawm_predictor.py
- perturb_jepa/training/bioguard_wm_calibration.py
- scripts/run_bioguard_wm_total_autonomy.py

F086/F087 are not models. They are validation-readiness / external-ingest
milestones.

Goal:
Run external paired scRNA + imaging validation for the F082 model path,
preferably scGeneScope, without redesigning the model first.

Important:
- Do not train a new architecture before external validation preflight.
- Do not promote anything unless a real external Tier 3 pass happens.
- Protected model of record remains rank-3 train-split-only PLS raw-linear
  readout.
- Use PLS only as protected baseline/audit floor, not as the JEPA representation
  path.
- Do not use condition_key, biological_key, exact target-key one-hot, held-out
  target means, pooled train+test stats, or leakage shortcuts.
- Keep raw data/checkpoints outside git.
- Use backed/obs-only checks first; do not load huge h5ad matrices into RAM.
- Put HF tokens only in environment variables, never in reports or committed
  files.

First steps:
1. Install and run focused tests.
2. Confirm storage/RAM/GPU availability.
3. Run scGeneScope metadata/obs-only/backed contract checks from the recovery
   plan.
4. Validate split mapping and RNA-image pairing.
5. Only then run F082 external validation.
6. Compare against source-as-target, protected full-ridge floor, and no-residual
   baselines.
7. Report transition improvement, delta cosine, recall@1, RNA->image/image->RNA
   retrieval, rank, leakage, and identity checks.
8. Write a clear external_validation_report.md with pass/fail.
```

Short version:

```text
Validate F082 externally. Treat F086/F087 as the data-validation frontier, not
the model.
```
