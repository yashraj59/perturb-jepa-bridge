# F086 Current-Registry Tier 3 Design

## Decision
`F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED`

## Purpose
F085 made the current image-teacher latent floor explicit. F086 converts that registry into a Tier 3 preflight design and checks whether a paired scRNA plus imaging perturbation validator is ready locally.

## Corrected Current-Registry Gates
- Current-registry transition floor: `0.209255`
- Current-registry delta-cosine floor: `0.937581`
- Current-registry recall@1 floor: `1.000000`
- Current-registry true-delta rank ceiling: `7.599185`
- Candidate transition: `0.207816`
- Candidate delta cosine: `0.934403`
- Candidate recall@1: `1.000000`
- Candidate delta rank: `7.023948`
- Synthetic current-registry gates pass: `True`
- Cross-modal gate pass: `True`

## Validator Readiness
- Local paired scRNA plus imaging validator available now: `False`
- Future paired validator candidate found: `True`

## External Validator Shortlist
```tsv
candidate	source_url	local_available	condition_paired_rna_image	rna_single_cell	cell_imaging	perturbation_controls	license_or_access_note	tier3_role	locally_usable_tier3_validator_now	future_paired_validator_candidate
Norman2019 processed h5ad	local:data/raw/gears_norman/norman/perturb_processed.h5ad	True	False	True	False	True	Local RNA-only processed h5ad; no imaging modality.	RNA-only diagnostic/no-regression only; cannot validate cross-modal JEPA.	False	False
scGeneScope	https://papers.nips.cc/paper_files/paper/2025/hash/ce02df43d66626bb7087ec699e20c7ea-Abstract-Datasets_and_Benchmarks_Track.html	False	True	True	True	True	Dataset reported as condition-paired scRNA-seq plus Cell Painting; Hugging Face listing reports cc-by-nc-4.0 and about 595 GB.	Best future paired validator candidate; metadata/feature-level ingest needed before Tier 3.	False	True
cpg0003-rosetta Cell Painting + L1000	s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/preprocessed_data/	False	True	False	True	True	Condition-paired morphology plus L1000/bulk-like expression profiles, not scRNA-seq.	Cross-modal condition-pair stress test only; not sufficient for scRNA JEPA promotion.	False	False
JUMP Cell Painting	https://jump-cellpainting.broadinstitute.org/	False	False	False	True	True	Large public imaging perturbation resource; no matched scRNA readout in this source.	Imaging-only no-regression or pretraining source, not paired validation.	False	False
scPerturb	https://www.sanderlab.org/scPerturb/	False	False	True	False	True	Harmonized single-cell perturbation molecular readouts; no Cell Painting modality in this resource.	RNA-only perturbation generalization/no-regression, not cross-modal validation.	False	False
RxRx19a	https://www.rxrx.ai/rxrx19a	False	False	False	True	True	Public fluorescent microscopy perturbation dataset; site reports CC-BY for RxRx19a and image/embedding downloads.	Imaging-only perturbation stress test, not scRNA+image validation.	False	False

```

## Summary Row
```tsv
source_experiments	current_registry_transition_floor	current_registry_delta_cosine_floor	current_registry_recall_floor	current_registry_true_delta_rank_ceiling	candidate_transition	candidate_delta_cosine	candidate_recall_at_1	candidate_delta_rank	min_candidate_minus_current_floor_transition	min_candidate_minus_current_floor_delta_cosine	min_candidate_minus_current_floor_recall	mean_rna_to_image_recall_at_1	mean_image_to_rna_recall_at_1	min_calibrated_transition_improvement	min_calibrated_delta_cosine	max_identity_violation	max_leakage_flag	candidate_preserves_current_floor	current_registry_rank_supported	locked_rank_floor_incomparable	local_paired_validator_available	future_paired_validator_candidate_found	model_promoted	identity_or_leakage_fail	cross_modal_gate_pass	synthetic_current_registry_gates_pass	diagnostic_label
F082,F085	0.20925520650173876	0.9375813779442321	1.0	7.59918459232027	0.20781648143800754	0.9344029333900138	1.0	7.0239482850671004	-0.008224412759849031	-0.013539194093556417	0.0	0.7716049382716049	0.8786008230452675	0.12491412882339287	0.9047365628809659	0.0	0.0	True	True	True	False	True	False	False	True	True	F086_CURRENT_REGISTRY_TIER3_READY_EXTERNAL_INGEST_NEEDED

```

## Tier 3 Decision
No model is promoted. A locked Tier 3 run is not launched from F086 because no paired scRNA plus Cell Painting perturbation validator is locally ingested. The next low-compute step should build a metadata/adapter preflight for the best external paired validator candidate, while keeping Norman RNA-only as a non-promoting RNA diagnostic.
