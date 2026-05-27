# scGeneScope External Validation Report

## Current Decision
`F096_PASS_EXTERNAL_TIER3_NON_PROMOTING`

F096 fixed the F082 external-validation failure as a non-promoting result: PubChem fingerprint action descriptors plus train-only delta-calibrated ProgramBootstrapJEPA cleared transition, delta-cosine, and recall floor gaps on validation, test, and alternate_test with zero identity or leakage flags. No model is promoted from this repair loop. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record unless a later fresh external Tier 3 confirmation explicitly supersedes it.

## F096 Fixed Candidate
- candidate family: `F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`
- fixed candidate method: `F082_delta_calibrated`
- descriptor mode: `pubchem_fingerprint`
- action descriptor: PubChem numeric molecular properties, PubChem 2D structure fingerprint bits, found/control flags; not exact treatment one-hot
- action dim: `932`
- missing non-control descriptor rows: `0`
- device: `cuda`

```tsv
method	split	mean_transition_improvement	mean_delta_cosine	mean_recall_at_1	mean_delta_rank	mean_rna_to_image_recall_at_1	mean_image_to_rna_recall_at_1	max_identity_violation	max_leakage_flag	floor_gap_transition_improvement	floor_gap_delta_cosine	floor_gap_recall_at_1
F082_delta_calibrated	alternate_test	0.4945876734920182	0.18575027719433235	0.12727272727272726	17.00653062864313	0.03636363636363636	0.03636363636363636	0.0	0.0	0.0006186114088329031	0.002169341817555376	0.0
F082_delta_calibrated	test	0.6464055625625784	0.3311939503906145	0.42857142857142855	17.089363558152993	0.17857142857142858	0.11904761904761903	0.0	0.0	0.0007091654080298992	0.001743470287220994	0.0
F082_delta_calibrated	validation	0.4001668950524826	0.27719805958822025	0.3703703703703704	17.348707910454987	0.12345679012345678	0.1111111111111111	0.0	0.0	0.0009176215168191781	0.0021773529266831404	0.0
```

## Original F082 Decision
`FAIL_EXTERNAL_NO_PROMOTION`

The original scalar-descriptor F082 result below remains preserved for audit lineage.

## Scope
- candidate: `F082_DELTA_CALIBRATED_TIER2_READY_FOR_TIER3_DESIGN`
- model path: `ProgramBootstrapJEPA` with train-only delta calibration
- external validator: scGeneScope feature H5ADs, RNA `scvi_n200`, imaging `imagenet/vit-l`
- action descriptor: PubChem numeric molecular properties plus found/control flags; not exact treatment one-hot
- action dim: `12`
- missing non-control descriptor rows: `10`

## Backed Obs Contract
```tsv
modality	round	path	n_obs	n_vars	required_obs_columns_present	replicate_values	split_values	treatment_unique_count	batch_unique_count	group_unique_count
rna	1	data/raw/scgenescope/features/rnaseq/scvi/n200/round_1.h5ad	406412	2000	True	3,4,5	test,train,validation	29	6	2
rna	2	data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad	221292	2000	True	1,2	alternate_test	29	4	2
image	1	data/raw/scgenescope/features/imaging/imagenet/vit-l/round_1.h5ad	449292	5120	True	3,4,5	test,train,validation	29	6	2
image	2	data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad	267475	5120	True	1,2	alternate_test	29	4	2

```

## Split And Pairing
Replicate mapping is fixed as `3=train`, `5=validation`, `4=test`, and `1/2=alternate_test`.

```tsv
split	paired_conditions	noncontrol_conditions	treatments	min_rna_profiles	min_image_profiles
alternate_test	57	55	29	90	2289
test	29	28	29	1641	2688
train	29	28	29	1494	3829
validation	28	27	28	2281	3661

```

## F082 And Floor Comparison
```tsv
method	split	mean_transition_improvement	mean_delta_cosine	mean_recall_at_1	mean_delta_rank	mean_magnitude_ratio	mean_rna_to_image_recall_at_1	mean_image_to_rna_recall_at_1	max_identity_violation	max_leakage_flag	floor_transition_improvement	floor_delta_cosine	floor_recall_at_1	floor_gap_transition_improvement	floor_gap_delta_cosine	floor_gap_recall_at_1
F082_delta_calibrated	alternate_test	0.5076391743202779	0.2165458954558255	0.12727272727272726	16.14132130953151	0.46079024883791764	0.048484848484848485	0.030303030303030304	0.0	0.0	0.5320606082928845	0.30895263319843574	0.0909090909090909	-0.024421433972606543	-0.09240673774261024	0.03636363636363636
F082_delta_calibrated	test	0.6482381066667665	0.35378500780851035	0.4047619047619048	16.222183440183237	0.6085822611763523	0.10714285714285714	0.09523809523809523	0.0	0.0	0.6123170931117412	0.396581728618621	0.21428571428571427	0.035921013555025305	-0.042796720810110656	0.19047619047619055
F082_delta_calibrated	validation	0.3777334133583768	0.279498786694004	0.2839506172839506	16.937475680453442	0.6799876942920443	0.1111111111111111	0.08641975308641975	0.0	0.0	0.32076760476355326	0.29575080567596823	0.14814814814814814	0.05696580859482353	-0.01625201898196421	0.13580246913580246
F082_no_delta_calibration	alternate_test	0.44449910613832894	0.32973208372917745	0.09696969696969697	14.461537777057103	0.26753662244515186					0.5320606082928845	0.30895263319843574	0.0909090909090909	-0.08756150215455555	0.020779450530741705	0.006060606060606072
F082_no_delta_calibration	test	0.5853030730325055	0.45874377395419663	0.2976190476190476	14.5102409948068	0.36213313078868975					0.6123170931117412	0.396581728618621	0.21428571428571427	-0.0270140200792357	0.06216204533557562	0.08333333333333334
F082_no_delta_calibration	validation	0.32579725090519557	0.34560641911537066	0.2839506172839506	14.289523584637116	0.42016077904627336					0.32076760476355326	0.29575080567596823	0.14814814814814814	0.005029646141642308	0.049855613439402424	0.13580246913580246
protected_full_ridge_floor	alternate_test	0.5320606082928845	0.30895263319843574	0.0909090909090909	8.281860527779063	0.3539842311778361					0.5320606082928845	0.30895263319843574	0.0909090909090909	0.0	0.0	0.0
protected_full_ridge_floor	test	0.6123170931117412	0.396581728618621	0.21428571428571427	8.284315081027978	0.47974003691409556					0.6123170931117412	0.396581728618621	0.21428571428571427	0.0	0.0	0.0
protected_full_ridge_floor	validation	0.32076760476355326	0.29575080567596823	0.14814814814814814	8.412934234879748	0.5441236100407503					0.32076760476355326	0.29575080567596823	0.14814814814814814	0.0	0.0	0.0
source_as_target	alternate_test	0.0	0.0	0.01818181818181818	0.0	0.0					0.5320606082928845	0.30895263319843574	0.0909090909090909	-0.5320606082928845	-0.30895263319843574	-0.07272727272727272
source_as_target	test	0.0	0.0	0.03571428571428571	0.0	0.0					0.6123170931117412	0.396581728618621	0.21428571428571427	-0.6123170931117412	-0.396581728618621	-0.17857142857142855
source_as_target	validation	0.0	0.0	0.037037037037037035	0.0	0.0					0.32076760476355326	0.29575080567596823	0.14814814814814814	-0.32076760476355326	-0.29575080567596823	-0.1111111111111111

```

## Baselines
```tsv
method	split	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio
F082_no_delta_calibration	alternate_test	0.44449910613832894	0.32973208372917745	0.09696969696969697	14.461537777057103	0.26753662244515186
F082_no_delta_calibration	test	0.5853030730325055	0.45874377395419663	0.2976190476190476	14.5102409948068	0.36213313078868975
F082_no_delta_calibration	validation	0.32579725090519557	0.34560641911537066	0.2839506172839506	14.289523584637116	0.42016077904627336
protected_full_ridge_floor	alternate_test	0.5320606082928845	0.30895263319843574	0.0909090909090909	8.281860527779063	0.3539842311778361
protected_full_ridge_floor	test	0.6123170931117412	0.396581728618621	0.21428571428571427	8.284315081027978	0.47974003691409556
protected_full_ridge_floor	validation	0.32076760476355326	0.29575080567596823	0.14814814814814814	8.412934234879748	0.5441236100407503
source_as_target	alternate_test	0.0	0.0	0.01818181818181818	0.0	0.0
source_as_target	test	0.0	0.0	0.03571428571428571	0.0	0.0
source_as_target	validation	0.0	0.0	0.037037037037037035	0.0	0.0

```

## Leakage And Identity Checks
- `condition_key`, `biological_key`, exact treatment one-hot, held-out target means, and pooled train+test statistics were not used as model inputs.
- Treatment labels were used for grouping, pairing, and retrieval relevance only.
- Image PCA and delta calibration were fit on train split only.
- PLS/full-ridge outputs are audit floors/baselines only; they are not the JEPA representation path.
- Raw data and cache files remain under ignored `data/raw/` and `outputs/` paths.

## Council Follow-Up
Council 122 and the F093 artifact-only audit diagnose this as a calibration-transfer and weak-action-descriptor failure, not a validation plumbing failure. The next permitted repair is `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`: a JEPA-only train/internal-replicate gate that can abstain from harmful calibration or choose a JEPA blend. The protected PLS/full-ridge floor remains an audit threshold only, not a candidate representation path or fallback output.

## Later Repair Status
F094 restored delta-cosine floor safety by abstaining to raw JEPA, but it still missed transition floor gaps on harder external splits. F095 added non-exact PubChem fingerprint descriptors and produced a near pass: the official split-safe gate missed `alternate_test` recall by `-0.006061`, while the calibrated fingerprint row cleared all floor gaps. No model is promoted because the calibrated row was not the predeclared selected candidate and scGeneScope has now informed multiple repair decisions.

## Fresh Confirmation Status
F096 remains a non-promoting scGeneScope repair-loop pass. A fresh external
confirmation is still required for promotion.

cpg0003 Rosetta was tested as an auxiliary fresh perturbational transcriptomics
plus morphology validator, but it is L1000 plus Cell Painting, not strict scRNA
plus imaging. F097 compound-holdout, F098 same-condition replicate-holdout, F100
zero-signature source-state, and F101 small-scale train-only calibration all
failed the registered fresh-confirmation gate. F099 diagnosed the Rosetta failure
as a source-state contract and validator mismatch. No model is promoted; the
protected rank-3 train-split-only PLS raw-linear readout remains the model of
record.

F102 found PerturbMulti as a public strict paired single-cell RNA plus imaging
candidate. Only manifest checks and a backed obs/schema probe of the small
protein-intensity H5AD were run; the large RNA H5AD obs and image-key pairing
checks are pending. No model was trained and no model is promoted from F102.
