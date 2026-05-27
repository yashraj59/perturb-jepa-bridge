# Codex Resume Handoff: BioGuard-WM-JEPA Synthetic Pass

## Purpose
This file is the portable handoff for resuming the work on another cluster.

Start from branch:

```bash
git checkout dev
```

`synthetic_pass` is the prior branch. The current external-validation packaging
branch is `dev`.

Then read these files first:

```text
CODEX_RESUME_HANDOFF.md
SYNTHETIC_PASS_SUMMARY.md
REPORTS_INDEX.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/HARD_ESCALATION_REPORT.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_report.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/MODEL_OF_RECORD.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/phase_reports/phase_closure_report_125.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F096_frozen_fingerprint_calibrated_candidate/F096_SCGENESCOPE_EXTERNAL_VALIDATION.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F097_cpg0003_rosetta_fresh_preflight/F097_CPG0003_ROSETTA_FRESH_PREFLIGHT.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/research_journal.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/results.tsv
```

## Current State
The synthetic/current-registry JEPA line works well enough to justify Tier 3 design, but it is not promoted.

Best synthetic candidate:

```text
F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN
```

Key F082 metrics:

```text
mean transition improvement = 0.207816
min transition improvement = 0.124914
mean delta cosine = 0.934403
min delta cosine = 0.904737
recall@1 = 1.000000
mean delta rank = 7.023948
RNA -> image recall@1 = 0.771605
image -> RNA recall@1 = 0.878601
identity violation = 0
leakage flag = 0
```

F086 converted this into a current-registry Tier 3 design:

```text
F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED
synthetic_current_registry_gates_pass = True
cross_modal_gate_pass = True
local_paired_validator_available = False
future_paired_validator_candidate_found = True
```

The active model of record is still:

```text
Protected rank-3 train-split-only PLS raw-linear readout
```

F096 passed the scGeneScope external floor comparison but is explicitly
non-promoting because it occurred after scGeneScope-guided repair. A fresh
external Tier 3 confirmation is still required before any promotion.

## Recovered Hard Stop

Earlier, F092 attempted the smallest scGeneScope paired feature dry run. The
files downloaded, but writing a status artifact failed with:

```text
OSError: [Errno 122] Disk quota exceeded
```

That state is preserved in:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/HARD_ESCALATION_REPORT.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
```

The current workspace later reran F091 and F092 successfully with backed
obs-only checks, then ran F082/F094/F095/F096 scGeneScope external validation.
Raw scGeneScope H5ADs are local under ignored `data/raw/scgenescope/`, but they
must not be committed or pushed.

## Current Resume Point

Do not promote F096. The current next stage is:

```text
F097_CPG0003_ROSETTA_FRESH_CONFIRMATION_PREFLIGHT
```

Manual preflight already found a viable fresh Rosetta candidate:

```text
dataset = cpg0003-rosetta CDRPBIO-BBBC036-Bray
assays = Cell Painting morphology + L1000 expression
cell line = U2OS
shared exact compound+dose pairs = 1469
controls = DMSO/negcon present in both modalities
Cell Painting replicates per shared pair = min 4, median 8, max 16
L1000 replicates per shared pair = min 1, median 2, max 2
SMILES missing among shared pairs = 0
```

Important caveat: cpg0003 Rosetta is a fresh external perturbational
transcriptomics+morphology validator, but L1000 is not scRNA. A pass can support
the frozen F096 path as a fresh external confirmation/stress test, but the final
promotion decision must explicitly decide whether the project still requires a
strict paired scRNA+imaging fresh validation.

F097/F098 execution update:

```text
F097_CPG0003_ROSETTA_COMPOUND_HOLDOUT = FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
F098_CPG0003_ROSETTA_SAME_CONDITION_REPLICATE_HOLDOUT = FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
```

F097 used compound-holdout splits and was stricter than the scGeneScope
replicate-heldout contract. F098 used same-condition replicate holdout, which is
closer to scGeneScope. F098 had positive floor gaps for transition and delta
cosine, but missed the recall floor on `test` and retained negative absolute
transition improvement. No model was promoted.

The next resume step should be an artifact-only F099 diagnostic before any new
model changes: inspect Rosetta source-state/control geometry, source-vs-target
cosines, floor behavior, and whether the L1000/Cell Painting profiles are
already centered so `source_as_target` is a stronger or differently defined
baseline than on scGeneScope.

## How To Resume On Another Cluster
1. Clone/check out `dev`.
2. Install dependencies:

```bash
python -m pip install -e ".[data,dev]"
```

3. Run local verification:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest \
  tests/test_total_autonomy_council.py \
  tests/test_scgenescope_adapter.py \
  tests/test_synthetic_biology_lite.py \
  tests/test_program_bootstrap_jepa.py \
  tests/test_bioguard_wm_calibration.py -q
```

Expected latest local result:

```text
77 passed
```

4. Confirm GPU, RAM, and disk before any model run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
df -h /content
free -h
```

5. If resuming F097 and the ignored Rosetta files are absent, download only the
   small CDRPBIO files:

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

6. Implement/run the F097 fresh confirmation runner. It should reuse the frozen
   F096 model path and avoid architecture redesign:

```text
ProgramBootstrapJEPA
PubChem/SMILES-derived non-exact action descriptors plus train-only delta calibration
source-as-target baseline
protected full-ridge audit floor
no-residual baseline
transition improvement, delta cosine, recall@1, RNA/L1000->image/CP and image/CP->RNA/L1000 retrieval
rank, identity, and leakage checks
```

Already implemented runner:

```text
scripts/run_cpg0003_rosetta_external_confirmation.py
```

Reproduce F098:

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

7. F099-F101 completed after F098:

```text
F099_ROSETTA_SOURCE_GEOMETRY_AUDIT:
  source-to-target cosine ~= 0.949, repeated source rank ~= 1, no leakage/identity flags.
  Diagnosis = source-state contract failure plus validator mismatch; no promotion.

F100_CPG0003_ROSETTA_ZERO_SIGNATURE_SOURCE:
  zero-signature source-state contract failed; no promotion.

F101_CPG0003_ROSETTA_SMALL_SCALE_CALIBRATION:
  train-only small-scale calibration selected scale 0.01, fixed delta cosine and recall floor gaps,
  but transition improvement stayed slightly negative on all external splits; no promotion.
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

8. Stop the Rosetta promotion loop. cpg0003 Rosetta is useful as an auxiliary
   L1000 plus Cell Painting stress test, but it is not strict paired scRNA plus
   imaging validation and did not pass the registered gate.

9. Resume at `F102_STRICT_SCRNA_IMAGING_FRESH_DATASET_PREFLIGHT`: search for or
   recover a strict paired scRNA plus imaging fresh validation protocol,
   preferably scGeneScope with an unused sealed split or another public paired
   dataset. Start with metadata/manifest/obs-only/backed checks only.

10. Before trying any new scGeneScope payload protocol, satisfy the recovery plan in:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
```

11. For any model run, use `--device cuda` unless the GPU is unavailable or
   already occupied. Do not fall back to CPU silently.

## Code Paths For The Synthetic-Passing Candidate
Core architecture and predictor:

```text
perturb_jepa/models/program_bootstrap_jepa.py
perturb_jepa/models/action_adaln_predictor.py
perturb_jepa/models/jepawm_predictor.py
perturb_jepa/models/jepawm_rope.py
perturb_jepa/models/bioguard_wm_jepa.py
```

Synthetic benchmark and action descriptors:

```text
perturb_jepa/training/synthetic_biology_lite.py
perturb_jepa/training/action_descriptors.py
```

Calibration, losses, and rollout support:

```text
perturb_jepa/training/bioguard_wm_calibration.py
perturb_jepa/training/bioguard_wm_losses.py
perturb_jepa/training/bioguard_wm_rollouts.py
perturb_jepa/training/bioguard_wm_status.py
```

Autonomous runner containing F079-F092:

```text
scripts/run_bioguard_wm_total_autonomy.py
```

Real-data adapter:

```text
perturb_jepa/data/scgenescope.py
```

Focused tests:

```text
tests/test_program_bootstrap_jepa.py
tests/test_bioguard_wm_calibration.py
tests/test_total_autonomy_council.py
tests/test_scgenescope_adapter.py
tests/test_synthetic_biology_lite.py
```

External validation scripts added in the current stage:

```text
scripts/run_f082_scgenescope_external_validation.py
scripts/audit_f082_scgenescope_failure_modes.py
```

## Do Not Do
- Do not promote F082 or F096 without a fresh external confirmation.
- Do not treat Norman RNA-only as cross-modal validation.
- Do not treat cpg0003 as scRNA; it is L1000 expression plus Cell Painting.
- Do not use condition_key, biological_key, exact target-key one-hot features, eval target means, or pooled train+test statistics.
- Do not rerun scGeneScope full payload access without quota-safe recovery evidence.
- Do not train on a new architecture before finishing the fresh external
  validation preflight for the frozen F096/F082 path.
- Do not commit or push raw H5AD, CSV.GZ, checkpoint, or condition-cache `.npz`
  payloads.
