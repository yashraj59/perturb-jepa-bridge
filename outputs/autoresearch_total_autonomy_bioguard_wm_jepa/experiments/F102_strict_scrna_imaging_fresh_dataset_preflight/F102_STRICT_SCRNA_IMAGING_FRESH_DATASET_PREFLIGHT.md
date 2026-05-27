# F102 Strict scRNA Imaging Fresh Dataset Preflight

## Decision
`F102_STRICT_SCRNA_IMAGING_CANDIDATE_FOUND_RNA_OBS_PREFLIGHT_PENDING`

No model is promoted. No model was trained. The protected rank-3
train-split-only PLS raw-linear readout remains the model of record.

## Scope
- objective: find a strict paired scRNA plus imaging fresh validation protocol
- model path under protection: frozen F082/F096 ProgramBootstrapJEPA path
- checks run: public source discovery, manifest-level file inspection, backed
  obs/schema check on the small protein-intensity H5AD only
- checks not run: large RNA H5AD backed obs, image tar cell-ID sampling, paired
  RNA-image overlap, model validation

## Candidate 1: PerturbMulti
PerturbMulti is the best actionable candidate found in this pass. The public
Hugging Face dataset `xingjiepan/PerturbMulti` is not gated, is licensed
CC-BY-4.0, and contains 54 tracked files at repository SHA
`8aac954eb631b68f6e11171a8313db61cc16c38c`.

Relevant manifest:

```tsv
candidate	repository_or_source	access	state	relevant_files	size_gib	contract_notes
PerturbMulti	xingjiepan/PerturbMulti	public_huggingface_cc_by_4_0	manifest_pass_obs_partial	RNA_scaled_crispr_screen_20240615.h5ad; protein_intensities_crispr_screen_20240615.h5ad; crispr_screen_20240615_chunk_*.tar	426.194	Public, non-gated. CRISPR screen has RNA h5ad, protein-intensity h5ad, perturbation metadata, spatial coordinates, and per-cell image tar archives keyed by cell ID.
```

The dataset card says the CRISPR screen RNA H5AD contains RNA expression,
spatial coordinates, and perturbation information for individual cells; the
protein-intensity H5AD is available for cells with targeting or control guide
RNAs; and each image file in the tar archives is named by cell ID.

Backed protein-intensity obs/schema probe:

```tsv
file	local_path	n_obs	n_vars	required_columns_present	condition_unique	perturbation_unique	dataset_unique	var_names_head
protein_intensities_crispr_screen_20240615.h5ad	/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad	99294	18	True	1	508	11	Alb,polyT,rRNA,M6PR,CathB,Perilipin,Sqstm1,LC3b,TOMM20,Calreticulin
```

Required columns observed in the protein H5AD: `cell_id`, `cell_name`,
`condition`, `perturbation`, `fov`, `x`, `y`, `z`, `dataset`.

## Candidate 2: HYPED
HYPED is modality-relevant on paper: long-read single-cell RNA sequencing from
approximately 20,000 cells and live-cell imaging across four perturbation
conditions. It is not actionable yet because this pass found the OpenReview
paper page and PDF but no public data manifest or stable download URL.

## Current Blockers
- PerturbMulti CRISPR RNA H5AD is 13.226 GiB. It was not downloaded in this
  pass because the machine had only about 2 GiB available RAM, and the next
  check should be a careful backed/HDF5 obs-only probe.
- PerturbMulti image archives are large. The full repository is 426.194 GiB, so
  the next check must sample only the minimum tar member metadata needed to prove
  cell-ID overlap.
- PerturbMulti is tissue/in vivo Perturb-seq plus imaging, not scGeneScope. It
  may be strict enough for paired scRNA plus imaging validation if cell-ID
  overlap is confirmed, but it requires RNA obs and image-key contract checks
  before any model run.

## Next Step
Run `F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT`:

1. Download only `RNA_scaled_crispr_screen_20240615.h5ad` to ignored
   `/content/hf_cache` or `data/raw/`.
2. Inspect RNA H5AD with backed or HDF5-level obs-only access; do not load `.X`.
3. Confirm RNA obs has `cell_name`/cell ID, perturbation, coordinates, and
   split-usable batch fields.
4. Check overlap between RNA cell IDs, protein H5AD cell IDs, and tar member
   image IDs using metadata only.
5. Only if pairing passes, design a small sealed split and run the frozen F082
   path on GPU.

## Sources
- Hugging Face dataset card: `https://huggingface.co/datasets/xingjiepan/PerturbMulti`
- PubMed record: `https://pubmed.ncbi.nlm.nih.gov/40513557/`
- Nature Genetics highlight: `https://www.nature.com/articles/s41588-025-02279-y.pdf`
- HYPED OpenReview page: `https://openreview.net/forum?id=HsJJDxjJ4R&noteId=xDsIpsq2JS`
