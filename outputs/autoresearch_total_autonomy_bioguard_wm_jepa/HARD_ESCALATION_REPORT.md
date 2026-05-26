# Hard Escalation Report

- label: `HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED`
- trigger experiment: `F092_SCGENESCOPE_OBS_ONLY_DRY_RUN`
- date: `2026-05-26`
- reason: F092 attempted the F091-approved smallest scGeneScope paired feature dry run. The two feature payloads downloaded, but writing the status artifact failed with `OSError: [Errno 122] Disk quota exceeded`.
- downloaded payloads before cleanup:
  - `data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad` (`2565764148` bytes)
  - `data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad` (`11023900503` bytes)
- cleanup performed: removed `data/raw/scgenescope`, which was created by F092, to recover workspace quota.
- model/training status: no model was trained, no whitening/calibration/target means were fit, and no model was promoted.
- protected model of record: protected rank-3 train-split-only PLS raw-linear readout remains active.
- transition floor: protected full-ridge transition floor remains active.
- required next step before retry: obtain a smaller manifest-backed feature subset or a workspace quota explicitly large enough for the paired feature payload plus output artifacts; do not retry full paired feature download automatically.
- post-escalation safeguard: `scripts/run_bioguard_wm_total_autonomy.py --f092-scgenescope-obs-only-dry-run-only` now refuses to retry while this report exists unless `--acknowledge-hard-escalation-retry` is provided. F092 also reserves a small status-write buffer before future payload attempts.
- recovery plan: `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/SCGENESCOPE_QUOTA_SAFE_RECOVERY_PLAN.md`
