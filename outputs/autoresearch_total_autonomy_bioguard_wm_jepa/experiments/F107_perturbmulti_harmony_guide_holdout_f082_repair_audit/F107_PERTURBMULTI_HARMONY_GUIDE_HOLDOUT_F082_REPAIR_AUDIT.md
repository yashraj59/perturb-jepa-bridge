# F107 PerturbMulti F082 External Validation

## Decision
`F107_FAIL_FRESH_EXTERNAL_TIER3_NO_PROMOTION`

- model promoted: `False`
- split mode: `guide_holdout_supported_gene`
- model path: frozen F082/F096 ProgramBootstrapJEPA architecture and train-only delta calibration
- protected audit floor: rank-3 train-split-only PLS/raw-linear model remains protected unless this report records a fresh Tier 3 pass
- raw H5AD/image payloads stayed outside git

## Preflight
```json
{
  "action_summary": {
    "action_descriptor": "Train-scaled gene-symbol morphology, signed character n-gram hashes, and train-control RNA projections; not exact target-key one-hot and no held-out target means.",
    "action_dim": 57,
    "descriptor_columns": [
      "gene_length",
      "uppercase_fraction",
      "digit_fraction",
      "vowel_fraction",
      "ascii_mean",
      "ascii_std",
      "is_control",
      "control_mean_projection_0",
      "control_mean_projection_1",
      "signed_char_ngram_hash_0",
      "signed_char_ngram_hash_1",
      "signed_char_ngram_hash_2",
      "signed_char_ngram_hash_3",
      "signed_char_ngram_hash_4",
      "signed_char_ngram_hash_5",
      "signed_char_ngram_hash_6",
      "signed_char_ngram_hash_7",
      "signed_char_ngram_hash_8",
      "signed_char_ngram_hash_9",
      "signed_char_ngram_hash_10",
      "signed_char_ngram_hash_11",
      "signed_char_ngram_hash_12",
      "signed_char_ngram_hash_13",
      "signed_char_ngram_hash_14",
      "signed_char_ngram_hash_15",
      "signed_char_ngram_hash_16",
      "signed_char_ngram_hash_17",
      "signed_char_ngram_hash_18",
      "signed_char_ngram_hash_19",
      "signed_char_ngram_hash_20",
      "signed_char_ngram_hash_21",
      "signed_char_ngram_hash_22",
      "signed_char_ngram_hash_23",
      "signed_char_ngram_hash_24",
      "signed_char_ngram_hash_25",
      "signed_char_ngram_hash_26",
      "signed_char_ngram_hash_27",
      "signed_char_ngram_hash_28",
      "signed_char_ngram_hash_29",
      "signed_char_ngram_hash_30",
      "signed_char_ngram_hash_31",
      "signed_char_ngram_hash_32",
      "signed_char_ngram_hash_33",
      "signed_char_ngram_hash_34",
      "signed_char_ngram_hash_35",
      "signed_char_ngram_hash_36",
      "signed_char_ngram_hash_37",
      "signed_char_ngram_hash_38",
      "signed_char_ngram_hash_39",
      "signed_char_ngram_hash_40",
      "signed_char_ngram_hash_41",
      "signed_char_ngram_hash_42",
      "signed_char_ngram_hash_43",
      "signed_char_ngram_hash_44",
      "signed_char_ngram_hash_45",
      "signed_char_ngram_hash_46",
      "signed_char_ngram_hash_47"
    ],
    "hash_dim": 48
  },
  "device_status": {
    "cuda_device_count": 1,
    "requested": "cuda",
    "selected": "cuda",
    "torch_cuda_available": true
  },
  "experiment_id": "F107",
  "leakage_controls": {
    "condition_key_eval_label_only": true,
    "no_biological_key": true,
    "no_condition_key_in_model_inputs": true,
    "no_exact_target_key_one_hot": true,
    "no_heldout_target_means": true,
    "raw_payloads_outside_git": true,
    "train_only_scaling_and_pca": true
  },
  "paired_summary": {
    "condition_genes": 202,
    "condition_guides": 417,
    "control_guides": 46,
    "exact_guide_matched_cells": 79048,
    "gene_verified_cells": 79048,
    "image_dim": 18,
    "min_cells_per_guide": 20,
    "protein_cells": 99294,
    "rna_dim": 30,
    "rna_feature_source": "obsm:X_pca_harmony",
    "shared_protein_rna_cells": 93848
  },
  "preflight_pass": true,
  "preflight_reason": "",
  "promotion_eligible": false,
  "protein_h5ad": "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad",
  "rna_feature_source": "obsm:X_pca_harmony",
  "rna_h5ad": "/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad",
  "split_mode": "guide_holdout_supported_gene",
  "split_summary": {
    "all_eval_splits_present": true,
    "rows": [
      {
        "cells": 3089,
        "guides": 17,
        "noncontrol_genes": 17,
        "noncontrol_guides": 17,
        "split": "alternate_test"
      },
      {
        "cells": 2563,
        "guides": 17,
        "noncontrol_genes": 17,
        "noncontrol_guides": 17,
        "split": "test"
      },
      {
        "cells": 69509,
        "guides": 366,
        "noncontrol_genes": 202,
        "noncontrol_guides": 320,
        "split": "train"
      },
      {
        "cells": 3510,
        "guides": 17,
        "noncontrol_genes": 17,
        "noncontrol_guides": 17,
        "split": "validation"
      }
    ]
  }
}
```

## Summary Metrics
method	split	mean_transition_improvement	mean_delta_cosine	mean_recall_at_1	mean_delta_rank	mean_magnitude_ratio	mean_rna_to_image_recall_at_1	mean_image_to_rna_recall_at_1	max_identity_violation	max_leakage_flag	floor_transition_improvement	floor_delta_cosine	floor_recall_at_1	floor_gap_transition_improvement	floor_gap_delta_cosine	floor_gap_recall_at_1
F082_delta_calibrated	alternate_test	-0.09591789151784724	0.07132840507107253	0.0784313725490196	10.086884903527855	0.7681484915375808	0.1372549019607843	0.0784313725490196	0.0	0.0	-0.028378499210749673	0.13936366922438723	0.0588235294117647	-0.06753939230709757	-0.06803526415331469	0.01960784313725491
F082_delta_calibrated	test	0.0471381765381785	0.09128426295475611	0.0784313725490196	10.936327436620305	0.7125089736412417	0.1568627450980392	0.1568627450980392	0.0	0.0	0.1304374170235519	0.205504310973899	0.0588235294117647	-0.0832992404853734	-0.1142200480191429	0.01960784313725491
F082_delta_calibrated	validation	0.1755140116065874	0.17589236493570926	0.19607843137254902	10.214474233585646	0.8104259193624065	0.13725490196078433	0.0588235294117647	0.0	0.0	0.24921410735603822	0.2532384622689893	0.17647058823529413	-0.07370009574945083	-0.07734609733328002	0.019607843137254888
F082_no_delta_calibration	alternate_test	-0.05613043172284774	0.08135193359238514	0.11764705882352942	12.062107105491956	0.7079638774193362					-0.028378499210749673	0.13936366922438723	0.0588235294117647	-0.02775193251209807	-0.058011735632002084	0.058823529411764726
F082_no_delta_calibration	test	0.03989686017447034	0.07110974596350593	0.0588235294117647	12.93448864550485	0.6568662851829604					0.1304374170235519	0.205504310973899	0.0588235294117647	-0.09054055684908158	-0.1343945650103931	0.0
F082_no_delta_calibration	validation	0.17223828813338224	0.15261223544666555	0.1372549019607843	12.239155852572862	0.710632063310466					0.24921410735603822	0.2532384622689893	0.17647058823529413	-0.07697581922265598	-0.10062622682232372	-0.03921568627450983
no_residual_source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0					-0.028378499210749673	0.13936366922438723	0.0588235294117647	0.028378499210749673	-0.13936366922438723	0.0
no_residual_source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0					0.1304374170235519	0.205504310973899	0.0588235294117647	-0.1304374170235519	-0.205504310973899	0.0
no_residual_source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0					0.24921410735603822	0.2532384622689893	0.17647058823529413	-0.24921410735603822	-0.2532384622689893	-0.11764705882352944
protected_full_ridge_floor	alternate_test	-0.028378499210749673	0.13936366922438723	0.0588235294117647	10.820958143204294	0.606092485466373					-0.028378499210749673	0.13936366922438723	0.0588235294117647	0.0	0.0	0.0
protected_full_ridge_floor	test	0.1304374170235519	0.205504310973899	0.0588235294117647	11.450959011986837	0.6306197547833433					0.1304374170235519	0.205504310973899	0.0588235294117647	0.0	0.0	0.0
protected_full_ridge_floor	validation	0.24921410735603822	0.2532384622689893	0.17647058823529413	10.660093609375213	0.6323725868222095					0.24921410735603822	0.2532384622689893	0.17647058823529413	0.0	0.0	0.0
source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0					-0.028378499210749673	0.13936366922438723	0.0588235294117647	0.028378499210749673	-0.13936366922438723	0.0
source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0					0.1304374170235519	0.205504310973899	0.0588235294117647	-0.1304374170235519	-0.205504310973899	0.0
source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0					0.24921410735603822	0.2532384622689893	0.17647058823529413	-0.24921410735603822	-0.2532384622689893	-0.11764705882352944

## Baseline Metrics
method	split	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio
F082_no_delta_calibration	alternate_test	-0.05613043172284774	0.08135193359238514	0.11764705882352942	12.062107105491956	0.7079638774193362
F082_no_delta_calibration	test	0.03989686017447034	0.07110974596350593	0.0588235294117647	12.93448864550485	0.6568662851829604
F082_no_delta_calibration	validation	0.17223828813338224	0.15261223544666555	0.1372549019607843	12.239155852572862	0.710632063310466
no_residual_source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0
no_residual_source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0
no_residual_source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0
protected_full_ridge_floor	alternate_test	-0.028378499210749673	0.13936366922438723	0.0588235294117647	10.820958143204294	0.606092485466373
protected_full_ridge_floor	test	0.1304374170235519	0.205504310973899	0.0588235294117647	11.450959011986837	0.6306197547833433
protected_full_ridge_floor	validation	0.24921410735603822	0.2532384622689893	0.17647058823529413	10.660093609375213	0.6323725868222095
source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0
source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0
source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0

## Leakage And Identity
- condition_key/biological_key/exact one-hot descriptors were not used.
- action descriptors were train-scaled gene-symbol/control-expression descriptors.
- PCA, standardization, and delta calibration used train split only.
- identity_violation and leakage_flag are reported in the summary metrics.
