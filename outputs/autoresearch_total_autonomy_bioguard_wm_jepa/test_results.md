# Focused Test Results

Latest verification after F092 hard-escalation recovery-plan documentation and
`synthetic_pass` handoff packaging:

```text
pytest tests/test_total_autonomy_council.py tests/test_scgenescope_adapter.py tests/test_synthetic_biology_lite.py tests/test_program_bootstrap_jepa.py tests/test_bioguard_wm_calibration.py -q
77 passed in 8.18s
```

F092 retry guard check:

```text
python scripts/run_bioguard_wm_total_autonomy.py --f092-scgenescope-obs-only-dry-run-only
{"latest_experiment": "F092_SCGENESCOPE_OBS_ONLY_DRY_RUN", "report": "outputs/autoresearch_total_autonomy_bioguard_wm_jepa/HARD_ESCALATION_REPORT.md", "status": "HARD_ESCALATION_ACTIVE_RETRY_BLOCKED"}
exit_code=2
```

Additional syntax check:

```text
python -m py_compile scripts/run_bioguard_wm_total_autonomy.py perturb_jepa/data/scgenescope.py
passed
```
