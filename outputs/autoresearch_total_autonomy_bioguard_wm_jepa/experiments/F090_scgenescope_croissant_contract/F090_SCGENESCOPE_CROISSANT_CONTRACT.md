# F090 scGeneScope Croissant Contract

## Decision
`F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE`

## Purpose
F089 recovered official Croissant metadata and a replicate split contract. F090 validates the local scGeneScope adapter against that metadata using a dry-run manifest only.

## Key Results
- Required Croissant fields present: `True`
- Replicate split mapping valid: `True`
- Adapter contract valid: `True`
- Dry-run manifest rows: `40`
- Dry-run condition pairs: `16/16`
- Splits represented: `4`
- Payload download attempted: `False`

## Split Pair Summary
```tsv
split	manifest_rows	sum	count
alternate_test	16	4	4
test	8	4	4
train	8	4	4
validation	8	4	4

```

## Summary Row
```tsv
required_fields_present	missing_fields	replicate_split_mapping_valid	mapping_mismatches	dose_fixed_or_manifest_supplied	target_column	control_column	batch_column	adapter_contract_valid	manifest_row_count	condition_pair_count	ready_condition_pair_count	split_count	all_dry_run_condition_pairs_ready	train_pair_count	validation_pair_count	test_pair_count	alternate_test_pair_count	payload_download_attempted	model_promoted	diagnostic_label
True	[]	True	{}	True	Treatment	Group	batch	True	40	16	16	4	True	4	4	4	4	False	False	F090_SCGENESCOPE_CROISSANT_CONTRACT_VALIDATED_READY_FOR_FEATURE_SIZE_GATE

```

## Decision Use
No model is promoted. The scGeneScope adapter contract is ready for a storage-gated feature-level preflight, but not for a Tier 3 run until local feature files or a manifest-backed feature subset exists.
