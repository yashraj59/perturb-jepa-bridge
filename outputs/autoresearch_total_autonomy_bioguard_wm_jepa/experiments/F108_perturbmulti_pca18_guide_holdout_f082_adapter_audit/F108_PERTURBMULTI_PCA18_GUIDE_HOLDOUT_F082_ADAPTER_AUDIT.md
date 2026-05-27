# F108 PerturbMulti F082 External Validation

## Decision
`F108_FAIL_FRESH_EXTERNAL_TIER3_NO_PROMOTION`

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
  "experiment_id": "F108",
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
    "rna_feature_source": "X",
    "shared_protein_rna_cells": 93848
  },
  "preflight_pass": true,
  "preflight_reason": "",
  "promotion_eligible": false,
  "protein_h5ad": "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad",
  "rna_feature_source": "X",
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
F082_delta_calibrated	alternate_test	-0.03524416774758162	0.1344781334712785	0.1372549019607843	10.359875712377931	0.7357471926136906	0.17647058823529413	0.2549019607843137	0.0	0.0	-0.07100543006251638	0.11687765259991605	0.0588235294117647	0.035761262314934754	0.017600480871362442	0.0784313725490196
F082_delta_calibrated	test	0.03599889910549199	0.11972560590133059	0.13725490196078433	10.420419203743673	0.719473934747383	0.1568627450980392	0.23529411764705885	0.0	0.0	0.11545413618045253	0.18148502160899746	0.17647058823529413	-0.07945523707496054	-0.06175941570766687	-0.0392156862745098
F082_delta_calibrated	validation	0.17032497449233144	0.18110977259015051	0.17647058823529413	10.345167750936978	0.769265983856018	0.11764705882352942	0.19607843137254902	0.0	0.0	0.26446252287477384	0.26793001327051513	0.1176470588235294	-0.0941375483824424	-0.08682024068036462	0.05882352941176473
F082_no_delta_calibration	alternate_test	-0.010255427587097866	0.143125220791214	0.13725490196078433	11.740796441816387	0.6429075612680614					-0.07100543006251638	0.11687765259991605	0.0588235294117647	0.06075000247541851	0.026247568191297962	0.07843137254901963
F082_no_delta_calibration	test	0.03223977917214708	0.10529867475349175	0.11764705882352942	11.74903063774424	0.6191178842887747					0.11545413618045253	0.18148502160899746	0.17647058823529413	-0.08321435700830546	-0.07618634685550571	-0.058823529411764705
F082_no_delta_calibration	validation	0.15925261957446443	0.16913514176628228	0.19607843137254902	11.861889178811131	0.6394547448624573					0.26446252287477384	0.26793001327051513	0.1176470588235294	-0.10520990330030941	-0.09879487150423286	0.07843137254901962
no_residual_source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0					-0.07100543006251638	0.11687765259991605	0.0588235294117647	0.07100543006251638	-0.11687765259991605	0.0
no_residual_source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0					0.11545413618045253	0.18148502160899746	0.17647058823529413	-0.11545413618045253	-0.18148502160899746	-0.11764705882352944
no_residual_source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0					0.26446252287477384	0.26793001327051513	0.1176470588235294	-0.26446252287477384	-0.26793001327051513	-0.0588235294117647
protected_full_ridge_floor	alternate_test	-0.07100543006251638	0.11687765259991605	0.0588235294117647	10.864400401293823	0.6106668324946977					-0.07100543006251638	0.11687765259991605	0.0588235294117647	0.0	0.0	0.0
protected_full_ridge_floor	test	0.11545413618045253	0.18148502160899746	0.17647058823529413	11.354556867652589	0.6579006907274378					0.11545413618045253	0.18148502160899746	0.17647058823529413	0.0	0.0	0.0
protected_full_ridge_floor	validation	0.26446252287477384	0.26793001327051513	0.1176470588235294	10.520526171499533	0.6357802272849371					0.26446252287477384	0.26793001327051513	0.1176470588235294	0.0	0.0	0.0
source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0					-0.07100543006251638	0.11687765259991605	0.0588235294117647	0.07100543006251638	-0.11687765259991605	0.0
source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0					0.11545413618045253	0.18148502160899746	0.17647058823529413	-0.11545413618045253	-0.18148502160899746	-0.11764705882352944
source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0					0.26446252287477384	0.26793001327051513	0.1176470588235294	-0.26446252287477384	-0.26793001327051513	-0.0588235294117647

## Baseline Metrics
method	split	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio
F082_no_delta_calibration	alternate_test	-0.010255427587097866	0.143125220791214	0.13725490196078433	11.740796441816387	0.6429075612680614
F082_no_delta_calibration	test	0.03223977917214708	0.10529867475349175	0.11764705882352942	11.74903063774424	0.6191178842887747
F082_no_delta_calibration	validation	0.15925261957446443	0.16913514176628228	0.19607843137254902	11.861889178811131	0.6394547448624573
no_residual_source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0
no_residual_source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0
no_residual_source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0
protected_full_ridge_floor	alternate_test	-0.07100543006251638	0.11687765259991605	0.0588235294117647	10.864400401293823	0.6106668324946977
protected_full_ridge_floor	test	0.11545413618045253	0.18148502160899746	0.17647058823529413	11.354556867652589	0.6579006907274378
protected_full_ridge_floor	validation	0.26446252287477384	0.26793001327051513	0.1176470588235294	10.520526171499533	0.6357802272849371
source_as_target	alternate_test	0.0	0.0	0.0588235294117647	0.0	0.0
source_as_target	test	0.0	0.0	0.0588235294117647	0.0	0.0
source_as_target	validation	0.0	0.0	0.0588235294117647	0.0	0.0

## Leakage And Identity
- condition_key/biological_key/exact one-hot descriptors were not used.
- action descriptors were train-scaled gene-symbol/control-expression descriptors.
- PCA, standardization, and delta calibration used train split only.
- identity_violation and leakage_flag are reported in the summary metrics.
