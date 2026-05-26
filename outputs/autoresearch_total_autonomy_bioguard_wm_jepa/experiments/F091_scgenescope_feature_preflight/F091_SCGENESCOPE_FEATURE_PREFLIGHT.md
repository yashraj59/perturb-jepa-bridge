# F091 scGeneScope Feature Preflight

## Decision
`F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN`

## Purpose
F090 validated the Croissant split and adapter contract. F091 checks whether a smallest paired RNA/image feature dry run is safe under storage, RAM, backed-IO, and low-compute gates before any payload download.

## Key Results
- F090 contract valid: `True`
- Feature pair candidates: `56`
- Smallest paired feature footprint: `13.590` GB
- Smallest RNA feature: `features/rnaseq/scvi/n200/round_2.h5ad` (`2565764148` bytes)
- Smallest image feature: `features/imaging/imagenet/vit-l/round_2.h5ad` (`11023900503` bytes)
- Available disk: `201571.21` GiB
- Available memory: `415.32` GiB
- Required disk gate: `53687091200` bytes
- Required backed-IO RAM gate: `22047801006` bytes
- `anndata` available: `True`
- `h5py` available: `True`
- Storage gate pass: `True`
- Backed-IO gate pass: `True`
- Low-compute payload gate pass: `True`
- Dataset payload download attempted: `False`

## Smallest Candidate Pairs
```tsv
round	rna_path	rna_size_bytes	image_path	image_size_bytes	paired_size_bytes
round_2	features/rnaseq/scvi/n200/round_2.h5ad	2565764148	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	13589664651
round_2	features/rnaseq/scvi/n200/round_2.h5ad	2565764148	features/imaging/imagenet/vit-h/round_2.h5ad	13762897751	16328661899
round_2	features/rnaseq/geneformer/round_2.h5ad	9195433072	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	20219333575
round_2	features/rnaseq/scvi/scvi_1/round_2.h5ad	10176230692	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	21200131195
round_2	features/rnaseq/scvi/scvi_2/round_2.h5ad	10176230692	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	21200131195
round_2	features/rnaseq/scgpt/round_2.h5ad	10408144396	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	21432044899
round_2	features/rnaseq/UCE/4layer/round_2.h5ad	10788867496	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	21812767999
round_1	features/rnaseq/scvi/n200/round_1.h5ad	4366952116	features/imaging/imagenet/vit-l/round_1.h5ad	18506448104	22873400220
round_2	features/rnaseq/geneformer/round_2.h5ad	9195433072	features/imaging/imagenet/vit-h/round_2.h5ad	13762897751	22958330823
round_2	features/rnaseq/scvi/scvi_1/round_2.h5ad	10176230692	features/imaging/imagenet/vit-h/round_2.h5ad	13762897751	23939128443
round_2	features/rnaseq/scvi/scvi_2/round_2.h5ad	10176230692	features/imaging/imagenet/vit-h/round_2.h5ad	13762897751	23939128443
round_2	features/rnaseq/scgpt/round_2.h5ad	10408144396	features/imaging/imagenet/vit-h/round_2.h5ad	13762897751	24171042147
round_2	features/rnaseq/scvi/n200/round_2.h5ad	2565764148	features/imaging/imagenet/resnet152/round_2.h5ad	21979889495	24545653643
round_2	features/rnaseq/scvi/n200/round_2.h5ad	2565764148	features/imaging/imagenet/resnet50/round_2.h5ad	21979889495	24545653643
round_2	features/rnaseq/UCE/4layer/round_2.h5ad	10788867496	features/imaging/imagenet/vit-h/round_2.h5ad	13762897751	24551765247
round_1	features/rnaseq/scvi/n200/round_1.h5ad	4366952116	features/imaging/imagenet/vit-h/round_1.h5ad	23107251432	27474203548
round_2	features/rnaseq/geneformer/round_2.h5ad	9195433072	features/imaging/imagenet/resnet152/round_2.h5ad	21979889495	31175322567
round_2	features/rnaseq/geneformer/round_2.h5ad	9195433072	features/imaging/imagenet/resnet50/round_2.h5ad	21979889495	31175322567
round_2	features/rnaseq/scvi/scvi_1/round_2.h5ad	10176230692	features/imaging/imagenet/resnet152/round_2.h5ad	21979889495	32156120187
round_2	features/rnaseq/scvi/scvi_1/round_2.h5ad	10176230692	features/imaging/imagenet/resnet50/round_2.h5ad	21979889495	32156120187

```

## Summary Row
```tsv
source_experiments	f090_contract_valid	remote_summary_path	remote_summary_present	remote_feature_record_count	feature_pair_candidate_count	smallest_pair_round	smallest_rna_feature_path	smallest_rna_feature_bytes	smallest_image_feature_path	smallest_image_feature_bytes	smallest_paired_feature_bytes	smallest_paired_feature_gb	available_disk_bytes	available_disk_gib	available_memory_bytes	available_memory_gib	required_disk_bytes	required_backed_ram_bytes	estimated_full_in_memory_bytes	low_compute_pair_cap_bytes	anndata_available	h5py_available	scanpy_available	storage_gate_pass	backed_io_gate_pass	low_compute_payload_gate_pass	local_root_exists	local_manifest_valid	payload_download_attempted	model_promoted	next_safe_action	diagnostic_label
F088,F090	True	outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F088_scgenescope_remote_discovery/hf_tree_summary.tsv	True	22	56	round_2	features/rnaseq/scvi/n200/round_2.h5ad	2565764148	features/imaging/imagenet/vit-l/round_2.h5ad	11023900503	13589664651	13.589664651	216435434389504	201571.20599365234	445950296064	415.3235778808594	53687091200	22047801006	54358658604	17179869184	True	True	True	True	True	True	False	False	False	False	If approved by the autonomy loop, run an obs-only/backed H5AD open dry run on the smallest paired feature files with strict byte accounting; do not train or promote.	F091_SCGENESCOPE_FEATURE_PREFLIGHT_APPROVES_BACKED_OBS_ONLY_DRY_RUN

```

## Decision Use
No model is promoted and no payload was downloaded. If this preflight approves, the next step is still an obs-only/backed H5AD open dry run with byte accounting, not training. If any gate fails, the loop should request or build a smaller manifest-backed feature subset instead of loading full payloads.
