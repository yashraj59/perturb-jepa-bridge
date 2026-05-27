# F093 F082 scGeneScope Failure Audit

## Decision
`F093_CALIBRATION_AND_DESCRIPTOR_REPAIR_REQUIRED`

No model is promoted. This audit reads only the existing F082 scGeneScope TSV/JSON artifacts and does not open raw H5AD matrices, fit a model, refit PCA, refit calibration, or use held-out target statistics for selection.

## Main Finding
F082 is not blocked by validation plumbing. The backed contracts, pairing, identity checks, and leakage checks passed. The failure is a representation-output repair problem: train-only delta calibration improves transition and recall in several held-out splits, but it consistently damages delta-cosine relative to the protected floor, while the uncalibrated JEPA output preserves delta-cosine better but lacks floor-safe transition on the external replicate and round shift.

## Split Failure Audit
```tsv
split	calibrated_transition	calibrated_delta_cosine	calibrated_recall_at_1	calibrated_delta_rank	rna_to_image_recall_at_1	image_to_rna_recall_at_1	floor_gap_transition	floor_gap_delta_cosine	floor_gap_recall_at_1	raw_minus_calibrated_transition	raw_minus_calibrated_delta_cosine	raw_minus_calibrated_recall_at_1	raw_floor_gap_transition	raw_floor_gap_delta_cosine	raw_floor_gap_recall_at_1
alternate_test	0.5076391743202779	0.2165458954558255	0.1272727272727272	16.14132130953151	0.0484848484848484	0.0303030303030303	-0.0244214339726065	-0.0924067377426102	0.0363636363636363	-0.06314006818194906	0.11318618827335189	-0.030303030303030304	-0.0875615021545556	0.020779450530741705	0.0060606060606060025
test	0.6482381066667665	0.3537850078085103	0.4047619047619048	16.222183440183237	0.1071428571428571	0.0952380952380952	0.0359210135550253	-0.0427967208101106	0.1904761904761905	-0.062935033634261	0.10495876614568628	-0.1071428571428572	-0.0270140200792357	0.062162045335575566	0.08333333333333343
validation	0.3777334133583768	0.279498786694004	0.2839506172839506	16.937475680453442	0.1111111111111111	0.0864197530864197	0.0569658085948235	-0.0162520189819642	0.1358024691358024	-0.05193616245318128	0.06610763242136658	0.0	0.005029646141642308	0.049855613439402424	0.1358024691358025

```

## Calibration Transfer Audit
```tsv
split	n_seed_rows	mean_train_calibrated_transition_gain	min_train_calibrated_transition_gain	max_train_calibrated_transition_gain	mean_train_calibrated_delta_cosine_gain	min_train_calibrated_delta_cosine_gain	max_train_calibrated_delta_cosine_gain	mean_train_calibrated_recall_gain	min_train_calibrated_recall_gain	max_train_calibrated_recall_gain	mean_calibrated_transition_gain	min_calibrated_transition_gain	max_calibrated_transition_gain	mean_calibrated_delta_cosine_gain	min_calibrated_delta_cosine_gain	max_calibrated_delta_cosine_gain	mean_calibrated_recall_gain	min_calibrated_recall_gain	max_calibrated_recall_gain
alternate_test	3	0.09149453106449668	0.083881716924473	0.09699744283146405	0.1166196990302782	0.11177156318356907	0.12327873406012513	0.023809523809523836	0.0	0.07142857142857151	0.06314006818194894	0.0335000285611546	0.1157990761202206	-0.11318618827335188	-0.1397021543941173	-0.0724752237876514	0.030303030303030234	0.0181818181818181	0.0545454545454545
test	3	0.09149453106449668	0.083881716924473	0.09699744283146405	0.1166196990302782	0.11177156318356907	0.12327873406012513	0.023809523809523836	0.0	0.07142857142857151	0.06293503363426096	0.0248229063158521	0.1215569270779197	-0.10495876614568622	-0.1401994182579762	-0.0588676871555795	0.1071428571428571	0.0714285714285714	0.1428571428571428
validation	3	0.09149453106449668	0.083881716924473	0.09699744283146405	0.1166196990302782	0.11177156318356907	0.12327873406012513	0.023809523809523836	0.0	0.07142857142857151	0.051936162453181244	0.0081167160839996	0.1180689795887351	-0.0661076324213666	-0.1037339015444554	-0.0102394283921349	0.0	-0.037037037037037	0.074074074074074

```

## Descriptor Coverage Audit
```tsv
total_treatments	pubchem_found_treatments	missing_noncontrol_treatments	missing_noncontrol_treatment_names	descriptor_action_dim	interpretation
29	26	2	PQ401;SKII	12	PubChem scalar properties are leakage-safe but too low-capacity for mechanism-aware perturbation action encoding.

```

## Repair Target
1. Implement `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`.
2. The gate must select among JEPA raw, JEPA calibrated, and train-only JEPA blend outputs using only train/internal replicate criteria.
3. The protected PLS/full-ridge floor remains an audit threshold only, not a candidate representation path or fallback output.
4. Add a descriptor-upgrade branch only after the calibration gate audit is complete, using non-exact public chemical or coarse mechanism descriptors and explicit missingness flags.
