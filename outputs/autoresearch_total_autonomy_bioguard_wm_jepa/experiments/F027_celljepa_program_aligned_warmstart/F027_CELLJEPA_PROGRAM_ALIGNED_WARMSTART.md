# F027 Cell-JEPA Program-Aligned Warmstart

## Decision
`F027_CELLJEPA_BELOW_PCA_FLOOR_USE_REPRESENTATION_REPAIR`

## Purpose
F026 fixed the synthetic action-target contract. This branch tests whether a JEPA-style RNA representation can preserve the new non-exact program-action transition floor before launching a full cross-modal/action JEPA.

## Active Gates
- transition improvement >= `0.003221`
- delta cosine >= `0.394388`
- condition recall@1 >= `0.255644`

## Key Rows
```tsv
representation	baseline	seed_count	mean_transition_improvement	min_transition_improvement	mean_delta_cosine	min_delta_cosine	mean_condition_recall_at_1	min_condition_recall_at_1	mean_delta_rank	mean_magnitude_ratio	mean_same_condition_recall_at_1	mean_effective_rank	mean_perturbation_probe_cv	mean_batch_probe_cv	mean_same_cell_recall_at_1	mean_dropout_robustness_cosine	min_weighted_jepa_to_rec_ratio
true_synthetic_z_bio	source_plus_program_action	5	0.15556172697338594	0.1252167257592387	0.9631851472491324	0.9549995836621555	1.0	1.0	3.5375183149621514	0.9728785289331497							
observed_rna_train_pca	source_plus_program_action	5	0.2807934818144681	0.10372015262959243	0.9013989661117551	0.8900379511214758	0.8814814814814815	0.6296296296296297	7.657517866786312	0.9922856494415846							
celljepa_rna_z	source_plus_program_action	5	7.080593010945172e-05	3.2880112823179344e-05	0.6673737488548827	0.5881244748373923	0.4666666666666667	0.3333333333333333	3.9790056813078962	0.9386546822764107							
celljepa_rna_z		5									0.3037037037037037	8.534979391531126	0.17777777777777776	0.28888888888888886	0.005555555555555555	0.9987188557636486	20.8021183013916

```

## Transition Summary
```tsv
representation	baseline	seed_count	mean_transition_improvement	min_transition_improvement	mean_delta_cosine	min_delta_cosine	mean_condition_recall_at_1	min_condition_recall_at_1	mean_delta_rank	mean_magnitude_ratio
celljepa_rna_z	source_plus_program_action	5	7.080593010945172e-05	3.2880112823179344e-05	0.6673737488548827	0.5881244748373923	0.4666666666666667	0.3333333333333333	3.9790056813078962	0.9386546822764107
celljepa_rna_z	source_only_ridge	5	5.771647668106977e-05	1.730402580870896e-05	0.5753335024637561	0.4622119350274677	0.2518518518518518	0.2222222222222222	3.2472049288409304	0.8044559685603476
celljepa_rna_z	program_action_only	5	2.8694246989642212e-05	1.6537620040428063e-06	0.3921189793527332	0.15991106889985357	0.37037037037037035	0.2222222222222222	1.5692071366129405	0.8116482895187321
celljepa_rna_z	source_as_target	5	0.0	0.0	0.0	0.0	0.26666666666666666	0.18518518518518517	0.0	0.0
observed_rna_train_pca	source_plus_program_action	5	0.2807934818144681	0.10372015262959243	0.9013989661117551	0.8900379511214758	0.8814814814814815	0.6296296296296297	7.657517866786312	0.9922856494415846
observed_rna_train_pca	program_action_only	5	0.23438994675751967	0.07657020593619916	0.7899362691730354	0.7557214054079874	0.837037037037037	0.6666666666666666	1.7429692536626562	0.8681031927019125
observed_rna_train_pca	source_only_ridge	5	0.07829559014850092	0.02927740861265825	0.5599042975660975	0.4876140654466471	0.3333333333333333	0.3333333333333333	7.348208030697805	0.6911800224091256
observed_rna_train_pca	source_as_target	5	0.0	0.0	0.0	0.0	0.3037037037037037	0.25925925925925924	0.0	0.0
true_synthetic_z_bio	source_plus_program_action	5	0.15556172697338594	0.1252167257592387	0.9631851472491324	0.9549995836621555	1.0	1.0	3.5375183149621514	0.9728785289331497
true_synthetic_z_bio	program_action_only	5	0.15546574125344373	0.12507697719503433	0.9635274402366057	0.954271044956526	1.0	1.0	1.978065909983501	0.9693707663603378
true_synthetic_z_bio	source_only_ridge	5	0.02851220795131265	0.020930454773517137	0.4040576135216812	0.34510920705742015	0.3333333333333333	0.3333333333333333	6.638632100329502	0.4688286784860578
true_synthetic_z_bio	source_as_target	5	0.0	0.0	0.0	0.0	0.3333333333333333	0.3333333333333333	0.0	0.0

```

## Representation Summary
```tsv
representation	seed_count	mean_same_condition_recall_at_1	mean_effective_rank	mean_perturbation_probe_cv	mean_batch_probe_cv	mean_same_cell_recall_at_1	mean_dropout_robustness_cosine	min_weighted_jepa_to_rec_ratio
celljepa_rna_z	5	0.3037037037037037	8.534979391531126	0.17777777777777776	0.28888888888888886	0.005555555555555555	0.9987188557636486	20.8021183013916
observed_rna_train_pca	5	0.8518518518518519	7.991993529808421	0.30277777777777776	0.23888888888888887	0.0	0.0	0.0
true_synthetic_z_bio	5	1.0	5.666920781350162	0.5722222222222222	0.12222222222222223	0.0	0.0	0.0

```

## Leakage And Promotion Status
- No model is promoted.
- Cell-JEPA pretraining uses train expression values only.
- Program-action floors are fit on train split only and scored on held-out perturbations.
- No `condition_key`, `biological_key`, held-out target means, pooled train+test statistics, or exact held-out perturbation one-hot features are model inputs.
