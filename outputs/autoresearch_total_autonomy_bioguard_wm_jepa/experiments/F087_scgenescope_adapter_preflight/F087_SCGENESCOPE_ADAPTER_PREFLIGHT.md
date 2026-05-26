# F087 scGeneScope Adapter Preflight

## Decision
`F087_SCGENESCOPE_ADAPTER_CONTRACT_READY_DATA_NOT_LOCAL`

## Purpose
F086 found a viable future paired validator candidate, but no local paired scRNA plus imaging validator was ready for Tier 3. F087 builds the metadata-only adapter contract before any large dataset download.

## Local Audit
- Root: `data/raw/scgenescope`
- Root exists: `False`
- Manifest path: `None`
- Manifest valid: `False`
- Paired condition count: `0`
- RNA record count: `0`
- Image record count: `0`
- Split count: `0`
- Message: Local scGeneScope root does not exist; no data was downloaded.

## Adapter Contract
```json
{
  "dataset": "scGeneScope",
  "forbidden_model_inputs": [
    "condition_key",
    "biological_key",
    "target_key",
    "exact_target_key"
  ],
  "local_root": "data/raw/scgenescope",
  "low_compute_ingest_order": [
    "metadata manifest only",
    "precomputed RNA/image embeddings or profile-level features if available",
    "small capped image tiles only after embedding/profile path validates"
  ],
  "manifest_template": "outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F087_scgenescope_adapter_preflight/scgenescope_manifest_template.tsv",
  "pairing_columns": [
    "treatment_id",
    "dose",
    "round"
  ],
  "paper_url": "https://papers.nips.cc/paper_files/paper/2025/hash/ce02df43d66626bb7087ec699e20c7ea-Abstract-Datasets_and_Benchmarks_Track.html",
  "required_columns": [
    "modality",
    "treatment_id",
    "dose",
    "round",
    "batch",
    "replicate",
    "split",
    "uri"
  ],
  "source_url": "https://huggingface.co/datasets/altoslabs/scGeneScope",
  "split_contract": "Use released train/validation/test or held-out-experiment splits; never fit encoders, calibrators, whitening, target means, or residual gates on eval target rows.",
  "tier3_role": "paired scRNA plus Cell Painting condition-paired external validator after local manifest validation"
}
```

## Summary Row
```tsv
root	root_exists	manifest_path	manifest_valid	paired_condition_count	rna_record_count	image_record_count	split_count	message	adapter_contract_written	manifest_template_written	model_promoted	diagnostic_label
data/raw/scgenescope	False		False	0	0	0	0	Local scGeneScope root does not exist; no data was downloaded.	True	True	False	F087_SCGENESCOPE_ADAPTER_CONTRACT_READY_DATA_NOT_LOCAL

```

## Decision Use
No model is promoted. If a local scGeneScope manifest is supplied later, the next low-compute step is a Tier 3 dry run over profile/embedding-level features only. If no manifest is local, continue with metadata/split adapter work and do not download full image payloads blindly.
