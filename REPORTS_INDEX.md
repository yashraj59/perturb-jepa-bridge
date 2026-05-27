# Reports Index

This file summarizes the reports needed to resume the project without the original Codex chat UI.

## Top-Level Handoff
```text
F082_EXTERNAL_VALIDATION_INSTRUCTIONS.md
CODEX_CHAT_SUMMARY.md
CODEX_RESUME_HANDOFF.md
SYNTHETIC_PASS_SUMMARY.md
REPORTS_INDEX.md
```

## Current Autoresearch Directory
```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/
```

Core logs:

```text
results.tsv
research_journal.md
AUTONOMY_LEDGER.md
autoresearch.md
family_allocation.md
test_results.md
```

Baseline/model-control files:

```text
MODEL_OF_RECORD.md
BASELINE_REGISTRY.md
BASELINE_REGISTRY_MULTISEED_V1.md
LOCKED_PRIOR_FACTS.md
LEAKAGE_RULES.md
NO_STOP_POLICY.md
identity_violations_considered.md
```

Literature and resources:

```text
papers_consulted.md
external_resources.md
```

Hard-stop and recovery:

```text
HARD_ESCALATION_REPORT.md
SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md
```

Insights:

```text
insights/INSIGHT_BRIEF_001.md
insights/INSIGHT_BRIEF_002.md
```

Debate council and amendment history:

```text
councils/debate_council_001.md ... councils/debate_council_126.md
amendments/SESSION_AMENDMENT_001.md ... amendments/SESSION_AMENDMENT_126.md
phase_reports/phase_closure_report_001.md ... phase_reports/phase_closure_report_127.md
```

## Key Experiment Reports
Synthetic pass path:

```text
experiments/F026_descriptor_aligned_synthetic_benchmark_audit/
experiments/F028_train_only_pca_distilled_rna_encoder/
experiments/F029_pca_bootstrap_cross_modal_action_jepa/
experiments/F030_delta_direction_cross_modal_action_jepa/
experiments/F031_delta_direction_tier2_validation/
experiments/F079_source_delta_rank_repair/
experiments/F080_source_delta_rank_jepa_wrapper/
experiments/F081_delta_calibrated_jepa_wrapper/
experiments/F082_delta_calibrated_tier2_validation/
experiments/F085_current_latent_floor_registry/
experiments/F086_current_registry_tier3_design/
```

External validator path:

```text
experiments/F087_scgenescope_adapter_preflight/
experiments/F088_scgenescope_remote_discovery/
experiments/F089_scgenescope_supplement_harvest/
experiments/F090_scgenescope_croissant_contract/
experiments/F091_scgenescope_feature_preflight/
experiments/F092_scgenescope_obs_only_dry_run/
external_validation_f082_scgenescope/
experiments/F096_frozen_fingerprint_calibrated_candidate/
experiments/F097_cpg0003_rosetta_external_confirmation/
experiments/F098_cpg0003_rosetta_replicate_holdout/
experiments/F099_rosetta_source_geometry_audit/
experiments/F100_cpg0003_rosetta_zero_signature_source/
experiments/F101_cpg0003_rosetta_small_scale_calibration/
experiments/F102_strict_scrna_imaging_fresh_dataset_preflight/
```

## Most Important Decisions
```text
F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN
F085_CURRENT_LATENT_FLOOR_REGISTRY_SUPPORTS_TIER3_WITH_UPDATED_RANK_GATE
F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED
F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE
F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN
F096_PASS_EXTERNAL_TIER3_NON_PROMOTING, not promoted because it is repair-loop validation
F099_SOURCE_STATE_CONTRACT_FAILURE_AND_VALIDATOR_MISMATCH_NO_PROMOTION
F101_FAIL_FRESH_EXTERNAL_CONFIRMATION_NO_PROMOTION
F102_STRICT_SCRNA_IMAGING_CANDIDATE_FOUND_RNA_OBS_PREFLIGHT_PENDING
F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT is the next safe resume step
```

## Test Evidence
Latest focused verification is recorded in:

```text
outputs/autoresearch_total_autonomy_bioguard_wm_jepa/test_results.md
```

Current expected focused result:

```text
77 passed
```

## Data And Payload Notes
Raw payloads are intentionally not committed. `.gitignore` excludes:

```text
data/raw/
*.h5ad
*.pt
*.zip
```

The failed scGeneScope payload directory was removed:

```text
data/raw/scgenescope
```

Norman RNA-only data may exist locally under `data/raw/gears_norman`, but it is ignored by Git and cannot validate cross-modal RNA/image JEPA.
