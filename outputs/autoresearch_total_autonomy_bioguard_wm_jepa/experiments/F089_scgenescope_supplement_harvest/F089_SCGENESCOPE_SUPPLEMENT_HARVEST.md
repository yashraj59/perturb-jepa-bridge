# F089 scGeneScope Supplement Harvest

## Decision
`F089_SCGENESCOPE_SUPPLEMENT_METADATA_RECOVERED_ADAPTER_UPDATED`

## Purpose
F088 found only large Hugging Face payload files and no small remote manifest. F089 downloads the small official NeurIPS supplemental archive and extracts metadata needed for a safe validator adapter.

## Key Results
- Supplement downloaded: `True`
- Supplement bytes: `2056507`
- Zip entries: `281`
- Croissant metadata found: `True`
- Code README found: `True`
- Replicate split contract found: `True`
- Treatment target field found: `True`
- GitHub repo from README: `https://github.com/altoslabs/scGeneScope.git`
- Full imaging download size from README: `186.0` GB
- scRNA download size from README: `80.0` GB
- Dataset payload download attempted: `False`

## Croissant Fields
```tsv
record_set	field	description	data_type
treatment_classification	cell_id	Unique identifier for each sample.	string
treatment_classification	Treatment	Chemical perturbation applied to cells. This is the target label for the treatment classification task.	string
treatment_classification	Replicate	Replicate number (1-5), used for train/val/test splits. Split assignments are: 3 = train, 5 = validation, 4 = test, and 1+2 = alternate test set.	string
treatment_classification	batch	Batch identifier for the sample. Currently unused, but could be used for batch effects correction.	string
treatment_classification	Group	Indicates whether the sample is a treatment or DMSO control. Not used for classification.	string

```

## Split Contract
```json
{
  "batch_column": "batch",
  "cell_id_column": "cell_id",
  "control_column": "Group",
  "dose_column": "not exposed in Croissant metadata; adapter treats dose as fixed unless a manifest provides concentration",
  "replicate_to_split": {
    "1": "alternate_test",
    "2": "alternate_test",
    "3": "train",
    "4": "test",
    "5": "validation"
  },
  "source": "scGeneScope Croissant metadata in NeurIPS supplemental archive",
  "target_column": "Treatment"
}
```

## Summary Row
```tsv
supplement_url	supplement_downloaded	download_error	supplement_bytes	zip_entry_count	croissant_found	code_readme_found	croissant_field_count	replicate_split_contract_found	treatment_target_field_found	github_repo_from_readme	full_imaging_download_gb_from_readme	scrna_download_gb_from_readme	single_plate_download_mb_from_readme	adapter_contract_updated_for_fixed_dose	dataset_payload_download_attempted	model_promoted	diagnostic_label
https://proceedings.neurips.cc/paper_files/paper/2025/file/ce02df43d66626bb7087ec699e20c7ea-Supplemental-Datasets_and_Benchmarks_Track.zip	True		2056507	281	True	True	5	True	True	https://github.com/altoslabs/scGeneScope.git	186.0	80.0	173.0	True	False	False	F089_SCGENESCOPE_SUPPLEMENT_METADATA_RECOVERED_ADAPTER_UPDATED

```

## Decision Use
No model is promoted. The adapter now has a concrete replicate-based split contract and should treat dose as fixed/implicit unless a manifest supplies concentration. The next step can validate the adapter against supplemental metadata and only then consider a capped feature-level dry run.
