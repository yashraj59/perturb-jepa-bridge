# F088 scGeneScope Remote Discovery

## Decision
`F088_SCGENESCOPE_REMOTE_FEATURES_FOUND_BUT_TOO_LARGE_FOR_LOW_COMPUTE`

## Purpose
F087 created a local adapter contract but found no local scGeneScope root. F088 queries only Hugging Face file-tree metadata to find small manifest/split files or low-compute feature-level dry-run targets. It does not download dataset payloads.

## Key Results
- Remote API accessible: `True`
- Remote entries inspected: `47`
- Remote errors: `0`
- Light manifest candidates found: `0`
- Large payload files observed: `26`
- Smallest RNA feature file: `2565764148` bytes
- Smallest image feature file: `11023900503` bytes
- Smallest paired feature footprint: `13.590` GB
- Dataset payload download attempted: `False`

## Remote Tree Preview
```tsv
path	type	size_bytes	modality	representation	round	is_light_metadata_candidate	is_large_payload	is_feature_h5ad
features	directory	0	unknown	unknown	unknown	False	False	False
measured	directory	0	unknown	unknown	unknown	False	False	False
.gitattributes	file	4841	unknown	unknown	unknown	False	False	False
README.md	file	33	unknown	unknown	unknown	False	False	False
features/imaging	directory	0	image	feature	unknown	False	False	False
features/rnaseq	directory	0	rna	feature	unknown	False	False	False
features/rnaseq/UCE	directory	0	rna	feature	unknown	False	False	False
features/rnaseq/geneformer	directory	0	rna	feature	unknown	False	False	False
features/rnaseq/pca	directory	0	rna	feature	unknown	False	False	False
features/rnaseq/scgpt	directory	0	rna	feature	unknown	False	False	False
features/rnaseq/scvi	directory	0	rna	feature	unknown	False	False	False
features/rnaseq/pca/n2000/round_1.h5ad	file	43347445080	rna	feature	round_1	False	True	True
features/rnaseq/pca/n2000/round_2.h5ad	file	26608721472	rna	feature	round_2	False	True	True
features/rnaseq/scvi/n200/round_1.h5ad	file	4366952116	rna	feature	round_1	False	True	True
features/rnaseq/scvi/n200/round_2.h5ad	file	2565764148	rna	feature	round_2	False	True	True
features/rnaseq/scvi/scvi_1/round_1.h5ad	file	16478887380	rna	feature	round_1	False	True	True
features/rnaseq/scvi/scvi_1/round_2.h5ad	file	10176230692	rna	feature	round_2	False	True	True
features/rnaseq/scvi/scvi_2/round_1.h5ad	file	16478887380	rna	feature	round_1	False	True	True
features/rnaseq/scvi/scvi_2/round_2.h5ad	file	10176230692	rna	feature	round_2	False	True	True
features/rnaseq/UCE/4layer/round_1.h5ad	file	18663899464	rna	feature	round_1	False	True	True
features/rnaseq/UCE/4layer/round_2.h5ad	file	10788867496	rna	feature	round_2	False	True	True
features/rnaseq/geneformer/round_1.h5ad	file	15593619472	rna	feature	round_1	False	True	True
features/rnaseq/geneformer/round_2.h5ad	file	9195433072	rna	feature	round_2	False	True	True
features/rnaseq/scgpt/round_1.h5ad	file	16904806844	rna	feature	round_1	False	True	True
features/rnaseq/scgpt/round_2.h5ad	file	10408144396	rna	feature	round_2	False	True	True
features/imaging/imagenet	directory	0	image	feature	unknown	False	False	False
features/imaging/imagenet/resnet152	directory	0	image	feature	unknown	False	False	False
features/imaging/imagenet/resnet50	directory	0	image	feature	unknown	False	False	False
features/imaging/imagenet/vit-h	directory	0	image	feature	unknown	False	False	False
features/imaging/imagenet/vit-l	directory	0	image	feature	unknown	False	False	False
features/imaging/imagenet/resnet50/round_1.h5ad	file	36909661416	image	feature	round_1	False	True	True
features/imaging/imagenet/resnet50/round_2.h5ad	file	21979889495	image	feature	round_2	False	True	True
features/imaging/imagenet/resnet152/round_1.h5ad	file	36909661416	image	feature	round_1	False	True	True
features/imaging/imagenet/resnet152/round_2.h5ad	file	21979889495	image	feature	round_2	False	True	True
features/imaging/imagenet/vit-l/round_1.h5ad	file	18506448104	image	feature	round_1	False	True	True
features/imaging/imagenet/vit-l/round_2.h5ad	file	11023900503	image	feature	round_2	False	True	True
features/imaging/imagenet/vit-h/round_1.h5ad	file	23107251432	image	feature	round_1	False	True	True
features/imaging/imagenet/vit-h/round_2.h5ad	file	13762897751	image	feature	round_2	False	True	True
measured/imaging	directory	0	image	measured	unknown	False	False	False
measured/rnaseq	directory	0	rna	measured	unknown	False	False	False
measured/rnaseq/round_1.h5ad	file	48101886092	rna	measured	round_1	False	True	False
measured/rnaseq/round_2.h5ad	file	29797702796	rna	measured	round_2	False	True	False
measured/imaging/uint8	directory	0	image	measured	unknown	False	False	False
measured/imaging/round_1.h5	file	2046741347	image	measured	round_1	False	True	False
measured/imaging/round_2.h5	file	1419408564	image	measured	round_2	False	True	False
measured/imaging/uint8/round_1	directory	0	image	measured	round_1	False	False	False
measured/imaging/uint8/round_2	directory	0	image	measured	round_2	False	False	False

```

## Remote Errors
```tsv
probe_path	error

```

## Summary Row
```tsv
remote_api_accessible	probe_path_count	remote_entry_count	remote_error_count	remote_file_count	remote_directory_count	light_manifest_candidate_count	light_manifest_candidate_found	large_payload_count	smallest_rna_feature_bytes	smallest_image_feature_bytes	smallest_paired_feature_bytes	smallest_paired_feature_gb	paired_feature_payloads_found	paired_feature_family_pair_count	download_attempted	model_promoted	diagnostic_label
True	20	47	0	28	19	0	False	26	2565764148	11023900503	13589664651	13.589664651	True	1	False	False	F088_SCGENESCOPE_REMOTE_FEATURES_FOUND_BUT_TOO_LARGE_FOR_LOW_COMPUTE

```

## Decision Use
No model is promoted. If no light manifest files are exposed, the next step should search official supplementary/code metadata or build an adapter that can use a user-supplied manifest. The full H5AD/image payloads should not be downloaded blindly under low-compute/storage constraints.
