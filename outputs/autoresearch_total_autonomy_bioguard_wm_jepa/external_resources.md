# External Resources

No new dataset was downloaded by this total-autonomy run.

Internet literature pages were inspected for paper/source context only and are recorded in `papers_consulted.md`.


## F086 Validator Discovery

No external dataset was downloaded. F086 wrote a validator shortlist at `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F086_current_registry_tier3_design/external_validator_shortlist.tsv` from local state plus public source pages inspected during the run:

- scGeneScope NeurIPS page: condition-paired scRNA-seq plus Cell Painting dataset and benchmark.
- JUMP Cell Painting Consortium page: public imaging perturbation resource.
- scPerturb page: harmonized single-cell perturbation molecular readouts.
- RxRx19a page: public fluorescent microscopy perturbation images and embeddings.
- cpg0003-rosetta reference from public Cell Painting Gallery/Rosetta materials: morphology plus L1000 condition-paired profiles, not scRNA-seq.

No license-gated or large dataset ingestion was attempted in F086.

## F088/F089 scGeneScope Resource Audit

No scGeneScope dataset payload was downloaded.

F088 queried Hugging Face file-tree metadata only and wrote:

- `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F088_scgenescope_remote_discovery/hf_tree_summary.tsv`
- `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F088_scgenescope_remote_discovery/F088_SCGENESCOPE_REMOTE_DISCOVERY.md`

Result: the Hugging Face API was accessible, but no light manifest/split file was visible. The smallest visible paired RNA/image feature footprint was about `13.59 GB`, so no feature H5AD was downloaded.

F089 downloaded only the official NeurIPS supplemental archive under a strict byte cap:

- URL: `https://proceedings.neurips.cc/paper_files/paper/2025/file/ce02df43d66626bb7087ec699e20c7ea-Supplemental-Datasets_and_Benchmarks_Track.zip`
- bytes: `2056507`
- artifact: `outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F089_scgenescope_supplement_harvest/scgenescope_supplemental.zip`

This archive supplied Croissant metadata and benchmark code only; it is not a dataset payload.


## F091 scGeneScope Feature Preflight

No scGeneScope dataset payload was downloaded in F091. The run used the saved Hugging Face file-tree summary from F088 and the Croissant contract from F090.

- smallest paired feature footprint: `13.590 GB`
- smallest RNA feature: `features/rnaseq/scvi/n200/round_2.h5ad`
- smallest image feature: `features/imaging/imagenet/vit-l/round_2.h5ad`
- storage gate pass: `True`
- backed-IO gate pass: `True`
- low-compute payload gate pass: `True`
- payload download attempted: `False`

## F092 scGeneScope Obs-Only Dry Run

F092 attempted only the F091-approved smallest paired feature payloads. The files downloaded, but writing the status artifact failed with `OSError: [Errno 122] Disk quota exceeded`, so this became `HARD_ESCALATE_COMPUTE_OR_STORAGE_EXHAUSTED`.

Payloads before cleanup:

```tsv
modality	path	bytes
rna	data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad	2565764148
image	data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad	11023900503
```

Cleanup performed: removed `data/raw/scgenescope`. No model was trained or promoted.
