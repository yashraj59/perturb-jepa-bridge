# F092 scGeneScope Obs-Only Backed Dry Run

## Decision
`F092_SCGENESCOPE_OBS_ONLY_BACKED_DRY_RUN_PASS`

## Purpose
F091 approved a smallest paired feature dry run under storage/RAM/backed-IO gates. F092 downloads only those two approved feature files if needed, opens them in backed mode, and inspects obs metadata only. It does not train, fit, whiten, calibrate, or promote.

## Key Results
- F091 preflight approved: `True`
- Expected paired bytes: `13589664651`
- Low-compute cap bytes: `17179869184`
- Payload cap violation: `False`
- Status-write reserve OK: `True`
- Payload download attempted: `True`
- Download/access error: `False`
- All expected files present: `True`
- Backed open error: `False`
- Obs contract pass: `True`
- Shared obs columns: `['Group', 'Replicate', 'Sample_ID', 'Seen', 'Treatment', 'batch', 'source']`

## File Status
```tsv
modality	repo_path	local_path	expected_bytes	actual_bytes	present	size_within_5pct	download_error
rna	features/rnaseq/scvi/n200/round_2.h5ad	data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad	2565764148	2565764148	True	True	
image	features/imaging/imagenet/vit-l/round_2.h5ad	data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	11023900503	True	True	

```

## Backed Obs Summary
```tsv
modality	path	n_obs	n_vars	required_obs_columns_present	treatment_unique_count	replicate_unique_count	batch_unique_count	group_unique_count	error
rna	data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad	221292	2000	True	29	2	4	2	
image	data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad	267475	5120	True	29	2	4	2	

```

## Summary Row
```tsv
source_experiments	f091_preflight_approved	expected_paired_bytes	low_compute_pair_cap_bytes	payload_cap_violation	status_write_reserve_ok	status_write_reserve_error	payload_download_attempted	download_error	download_errors	all_expected_files_present	backed_open_error	obs_contract_pass	shared_obs_columns	model_promoted	leakage_controls	diagnostic_label
F091	True	13589664651	17179869184	False	True		True	False	{}	True	False	True	['Group', 'Replicate', 'Sample_ID', 'Seen', 'Treatment', 'batch', 'source']	False	F092 downloads only the smallest public feature files selected by F091 and opens H5AD obs metadata in backed mode; no encoders, whitening, target means, residual gates, or model selection are fit.	F092_SCGENESCOPE_OBS_ONLY_BACKED_DRY_RUN_PASS

```

## Decision Use
No model is promoted. A pass only permits the next shape/split audit over backed feature metadata; it does not permit fitting JEPA encoders, transition heads, residual gates, whitening, or target statistics on scGeneScope.
