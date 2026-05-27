# Codex Chat Summary

This is a compressed operational summary of the Codex research sessions so a new
Codex instance can resume without the original chat transcript.

## User Direction That Controls The Current State

The user repeatedly instructed the agent to run autonomous low-compute
autoresearch on JEPA-style perturbation models for scRNA-seq plus imaging, to
avoid real-data leakage, to keep the protected PLS baseline as model of record
until a real Tier 3 pass, and to document every pivot. Later instructions
enabled continuous autonomy where ordinary failed candidates became documented
pivot events, not stopping events.

The latest packaging request is to push the current external-validation stage to
a new branch named `dev`, without pushing raw data files, and to make the resume
point explicit in `CODEX_CHAT_SUMMARY.md`, `CODEX_RESUME_HANDOFF.md`, and
`F082_EXTERNAL_VALIDATION_INSTRUCTIONS.md`.

## Research Path In One Timeline

1. Initial low-compute protocol established Step 0 baselines, synthetic data,
   leakage gates, and protected PLS/raw-linear readout controls.
2. BioAction-JEPA implemented a real JEPA path, but early candidates were not
   promoted because batch leakage, transition weakness, or floor preservation
   failures dominated.
3. BioTech/BioMechanistic/BioFlow/BioOperator/BioSpectral/BioGuard phases
   diagnosed the same core problem: residual or neural transition heads could
   fit train structure but fell below the protected train-only transition floor
   on held-out perturbation scoring.
4. The 2512.24497v3 paper informed the JEPA-WM/action-AdaLN/RoPE predictor
   branch. That branch produced useful primitives but still needed strict
   floor-preserving calibration.
5. The 2602.02093v1 Cell-JEPA paper informed the RNA representation warmstart
   branch. The conclusion was that Cell-JEPA-style representation learning is
   useful as a representation diagnostic/warmstart, but not enough by itself
   for protected delta/effect-size transition promotion.
6. The successful synthetic path came after the benchmark was redesigned around
   descriptor-aligned program actions and after the model preserved source
   state, delta direction, and latent rank before adding JEPA wrapping and
   train-only delta calibration.
7. F082 passed the synthetic/current-registry gates strongly enough for Tier 3
   design, but it was not promoted.
8. scGeneScope was validated through backed obs-only preflight, then used for
   the F082 external validation path. The original scalar-descriptor F082 run
   failed, F094 split-safe calibration restored delta safety but missed
   transition floors, and F095 PubChem fingerprints nearly passed but the
   predeclared gate still failed.
9. F096 froze the useful repaired candidate:
   PubChem fingerprint descriptors plus train-only delta-calibrated
   ProgramBootstrapJEPA. It passed the scGeneScope external floor comparison,
   but is explicitly non-promoting because scGeneScope had already guided the
   repair loop.
10. The current next stage is fresh external confirmation. The first fresh
    candidate being preflighted is cpg0003 Rosetta CDRPBIO-BBBC036-Bray:
    Cell Painting morphology profiles plus L1000 expression profiles with
    1,469 exact shared compound+dose pairs and DMSO controls. This is a fresh
    perturbational transcriptomics plus morphology confirmation target, not a
    strict scRNA replacement.

## Final Synthetic Candidate

Current best synthetic/current-registry candidate:

```text
F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN
```

Architecture/code path:

```text
perturb_jepa/models/program_bootstrap_jepa.py
perturb_jepa/models/action_adaln_predictor.py
perturb_jepa/models/jepawm_predictor.py
perturb_jepa/models/jepawm_rope.py
perturb_jepa/models/bioguard_wm_jepa.py
perturb_jepa/training/bioguard_wm_calibration.py
perturb_jepa/training/synthetic_biology_lite.py
scripts/run_bioguard_wm_total_autonomy.py
```

Synthetic result:

```text
mean calibrated transition improvement = 0.207816
min calibrated transition improvement = 0.124914
mean calibrated delta cosine = 0.934403
min calibrated delta cosine = 0.904737
recall@1 = 1.000000
mean delta rank = 7.023948
RNA -> image recall@1 = 0.771605
image -> RNA recall@1 = 0.878601
identity violation = 0
leakage flag = 0
```

Status:

```text
Synthetic/current-registry Tier 3 design ready.
F096 scGeneScope repair-loop pass is non-promoting.
Fresh external confirmation is still required before any promotion.
Protected rank-3 train-split-only PLS raw-linear readout remains model of record.
```

## Current External Validation State

F096 result:

```text
candidate = F082_delta_calibrated
descriptor mode = pubchem_fingerprint
gate mode = calibrated
dataset = scGeneScope
device = cuda
decision = PASS_EXTERNAL_TIER3_NON_PROMOTING
promotion = no
```

Key F096 split means:

```text
alternate_test transition = 0.494588, delta cosine = 0.185750, recall@1 = 0.127273
test transition = 0.646406, delta cosine = 0.331194, recall@1 = 0.428571
validation transition = 0.400167, delta cosine = 0.277198, recall@1 = 0.370370
all floor gaps for transition, delta cosine, and recall were >= 0
identity violation = 0
leakage flag = 0
```

Why it does not promote:

```text
The scGeneScope validator informed F093/F094/F095 repair decisions.
F096 therefore confirms the repaired path on scGeneScope but is not a fresh
external Tier 3 confirmation.
```

Current fresh-confirmation preflight:

```text
candidate dataset = cpg0003-rosetta CDRPBIO-BBBC036-Bray
assays = Cell Painting morphology + L1000 expression
shared exact compound+dose pairs = 1469
controls = DMSO/negcon present in both modalities
Cell Painting replicates per shared pair = min 4, median 8, max 16
L1000 replicates per shared pair = min 1, median 2, max 2
SMILES missing among shared pairs = 0
status = runner implementation not yet complete
promotion = still blocked until a fresh external confirmation actually passes
```

Current fresh-confirmation run status:

```text
F097 compound-holdout cpg0003 Rosetta = FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
F098 same-condition replicate-holdout cpg0003 Rosetta = FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
F099 artifact-only Rosetta geometry audit = SOURCE_STATE_CONTRACT_FAILURE_AND_VALIDATOR_MISMATCH_NO_PROMOTION
F100 zero-signature source-state run = FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
F101 small-scale train-only delta calibration = FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
F101 fixed delta cosine and recall floor gaps, but absolute transition
improvement stayed slightly negative on all splits.
No model promoted.
Next needed step = stop the Rosetta promotion loop and resume strict paired
scRNA plus imaging fresh-dataset preflight, preferably an unused scGeneScope
sealed split or another public paired scRNA/imaging dataset.
```

## How Synthetic Finally Worked

The decisive sequence was:

```text
descriptor-aligned synthetic benchmark
-> source/program image-teacher repair
-> source/delta/rank repair
-> real ProgramBootstrapJEPA wrapper
-> train-only delta calibration
-> fresh-seed Tier 2 style validation
```

The important lesson is that the transition head had to preserve source-state
geometry and delta rank before learning residual structure. Training longer,
increasing invariance, or adding unconstrained residual capacity repeatedly
improved train fit while hurting held-out transition metrics.

## How Many Things Were Tried

The current total-autonomy registry contains:

```text
128 experiment/result rows
128 unique experiment IDs
121 debate-council continuation cycles
```

Major families:

```text
Representation repair: 33 rows
Metric and data redesign: 23 rows
Descriptor-aligned action contract: 17 rows
Program and graph action priors: 10 rows
External validator: 6 rows
Tier 3 design: 4 rows
```

## Reports To Read First In A New Codex Session

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
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/results.tsv
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/research_journal.md
```

## Next Safe Resume Step

Start from branch `dev`. Do not promote F096/F097/F098/F100/F101. cpg0003
Rosetta should now be treated as an auxiliary L1000 plus Cell Painting stress
test, not a strict promotion validator. Resume at
`F102_STRICT_SCRNA_IMAGING_FRESH_DATASET_PREFLIGHT`: find or recover a strict
paired scRNA plus imaging fresh validation protocol, run metadata/obs-only/backed
checks first, and only then run the frozen F082/F096 ProgramBootstrapJEPA path on
GPU unless the GPU is unavailable or occupied.
