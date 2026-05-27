# F106 PerturbMulti F082 External Validation

## Decision
`F106_FAIL_FRESH_EXTERNAL_TIER3_NO_PROMOTION`

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
  "experiment_id": "F106",
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
    "rna_dim": 209,
    "shared_protein_rna_cells": 93848
  },
  "preflight_pass": true,
  "preflight_reason": "",
  "promotion_eligible": false,
  "protein_h5ad": "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad",
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
F082_delta_calibrated	alternate_test	-0.09549258779412684	0.05295676606189937	0.17647058823529413	11.499606939418838	0.9290284552329616	0.19607843137254902	0.2352941176470588	0.0	0.0	-0.07100543006251646	0.11687765259991605	0.0588235294117647	-0.02448715773161038	-0.06392088653801667	0.11764705882352944
F082_delta_calibrated	test	0.12931706811262322	0.1465440928663133	0.1372549019607843	10.500833419193372	0.9369275764933684	0.11764705882352942	0.13725490196078433	0.0	0.0	0.11545413618045242	0.1814850216089974	0.17647058823529413	0.013862931932170808	-0.03494092874268409	-0.03921568627450983
F082_delta_calibrated	validation	0.16285495925922147	0.1310945475102747	0.21568627450980393	11.09306306240692	0.973302724470073	0.0784313725490196	0.0588235294117647	0.0	0.0	0.26446252287477384	0.267930013270515	0.1176470588235294	-0.10160756361555237	-0.13683546576024033	0.09803921568627454
F082_no_delta_calibration	alternate_test	-0.07029385990322079	0.05230023088643317	0.19607843137254902	11.762741382083354	0.9479885815515602					-0.07100543006251646	0.11687765259991605	0.0588235294117647	0.0007115701592956686	-0.06457742171348288	0.13725490196078433
F082_no_delta_calibration	test	0.10752835056762693	0.10668816790722697	0.17647058823529413	10.898129134014367	0.9494601972381025					0.11545413618045242	0.1814850216089974	0.17647058823529413	-0.00792578561282549	-0.07479685370177043	0.0
F082_no_delta_calibration	validation	0.15635948392681667	0.10971118820691127	0.19607843137254902	11.351376748128507	0.97665987873107					0.26446252287477384	0.267930013270515	0.1176470588235294	-0.10810303894795717	-0.15821882506360374	0.07843137254901962
no_residual_source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0					-0.07100543006251646	0.11687765259991605	0.0588235294117647	0.07100543006251646	-0.11687765259991605	0.0
no_residual_source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0					0.11545413618045242	0.1814850216089974	0.17647058823529413	-0.11545413618045242	-0.1814850216089974	-0.11764705882352944
no_residual_source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0					0.26446252287477384	0.267930013270515	0.1176470588235294	-0.26446252287477384	-0.267930013270515	-0.0588235294117647
protected_full_ridge_floor	alternate_test	-0.07100543006251646	0.11687765259991605	0.0588235294117647	10.864400401293823	0.6106668324946977					-0.07100543006251646	0.11687765259991605	0.0588235294117647	0.0	0.0	0.0
protected_full_ridge_floor	test	0.11545413618045242	0.1814850216089974	0.17647058823529413	11.354556867652589	0.6579006907274377					0.11545413618045242	0.1814850216089974	0.17647058823529413	0.0	0.0	0.0
protected_full_ridge_floor	validation	0.26446252287477384	0.267930013270515	0.1176470588235294	10.520526171499528	0.635780227284937					0.26446252287477384	0.267930013270515	0.1176470588235294	0.0	0.0	0.0
source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0					-0.07100543006251646	0.11687765259991605	0.0588235294117647	0.07100543006251646	-0.11687765259991605	0.0
source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0					0.11545413618045242	0.1814850216089974	0.17647058823529413	-0.11545413618045242	-0.1814850216089974	-0.11764705882352944
source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0					0.26446252287477384	0.267930013270515	0.1176470588235294	-0.26446252287477384	-0.267930013270515	-0.0588235294117647

## Baseline Metrics
method	split	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio
F082_no_delta_calibration	alternate_test	-0.07029385990322079	0.05230023088643317	0.19607843137254902	11.762741382083354	0.9479885815515602
F082_no_delta_calibration	test	0.10752835056762693	0.10668816790722697	0.17647058823529413	10.898129134014367	0.9494601972381025
F082_no_delta_calibration	validation	0.15635948392681667	0.10971118820691127	0.19607843137254902	11.351376748128507	0.97665987873107
no_residual_source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0
no_residual_source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0
no_residual_source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0
protected_full_ridge_floor	alternate_test	-0.07100543006251646	0.11687765259991605	0.0588235294117647	10.864400401293823	0.6106668324946977
protected_full_ridge_floor	test	0.11545413618045242	0.1814850216089974	0.17647058823529413	11.354556867652589	0.6579006907274377
protected_full_ridge_floor	validation	0.26446252287477384	0.267930013270515	0.1176470588235294	10.520526171499528	0.635780227284937
source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0
source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0
source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0

## Leakage And Identity
- condition_key/biological_key/exact one-hot descriptors were not used.
- action descriptors were train-scaled gene-symbol/control-expression descriptors.
- PCA, standardization, and delta calibration used train split only.
- identity_violation and leakage_flag are reported in the summary metrics.
