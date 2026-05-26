# Codex Resume Handoff: BioGuard-WM-JEPA Synthetic Pass

## Purpose
This file is the portable handoff for resuming the work on another cluster.

Start from branch:

```bash
git checkout synthetic_pass
```

Then read these files first:

```text
CODEX_RESUME_HANDOFF.md
SYNTHETIC_PASS_SUMMARY.md
REPORTS_INDEX.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/HARD_ESCALATION_REPORT.md
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
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

No JEPA candidate has passed Tier 3 or been promoted.

## Hard Stop
The current loop is halted by:

```text
HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED
```

F092 attempted the smallest scGeneScope paired feature dry run. The files downloaded, but writing a status artifact failed with:

```text
OSError: [Errno 122] Disk quota exceeded
```

The generated payload directory was removed:

```text
data/raw/scgenescope
```

Do not rerun F092 blindly. The runner now blocks F092 while `HARD_ESCALATION_REPORT.md` exists unless `--acknowledge-hard-escalation-retry` is explicitly supplied.

## How To Resume On Another Cluster
1. Clone/check out `synthetic_pass`.
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

4. Before trying real scGeneScope payloads, satisfy the recovery plan in:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
```

5. First real-data resume step must remain metadata-only or obs-only/backed. Do not train on scGeneScope until the feature `obs` contract, split mapping, and pair table are proven on the target cluster.

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

## Do Not Do
- Do not promote F082; it is Tier 2 style only.
- Do not treat Norman RNA-only as cross-modal validation.
- Do not use condition_key, biological_key, exact target-key one-hot features, eval target means, or pooled train+test statistics.
- Do not rerun scGeneScope full payload access without quota-safe recovery evidence.
- Do not train on real scGeneScope until backed `obs` contract and split/pairing audits pass.
