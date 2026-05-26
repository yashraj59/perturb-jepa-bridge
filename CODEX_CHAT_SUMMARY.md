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

The final packaging request was to save the chat state in the repo, summarize
all reports, push the best synthetic architecture to branch `synthetic_pass`,
and explain how synthetic finally worked and how many things were tried.

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
   design, but it was not promoted because no real paired scRNA plus imaging
   Tier 3 validator has run.
8. The next external validator was scGeneScope. Remote and Croissant discovery
   succeeded, feature preflight identified a smallest paired feature pair, but
   F092 hit disk quota during status artifact writing. The payload was removed
   and a hard escalation report plus quota-safe recovery plan were written.

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
No JEPA promoted.
Protected rank-3 train-split-only PLS raw-linear readout remains model of record.
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
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/results.tsv
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/research_journal.md
```

## Next Safe Resume Step

On the next cluster, do not start by training. First run tests, then run
scGeneScope metadata-only or backed obs-only contract checks under the quota-safe
recovery plan. Only train on real paired data after split mapping, pairing, and
feature contracts are proven without loading multi-GB matrices into the working
tree.
