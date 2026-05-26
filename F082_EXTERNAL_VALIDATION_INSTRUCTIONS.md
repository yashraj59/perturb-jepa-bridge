# F082 External Validation Start Instructions

Use this prompt when starting Codex on another cluster for external validation.

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
