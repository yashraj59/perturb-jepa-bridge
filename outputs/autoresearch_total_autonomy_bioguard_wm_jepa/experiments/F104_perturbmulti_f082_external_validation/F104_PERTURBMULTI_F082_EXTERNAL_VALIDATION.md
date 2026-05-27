# F104 PerturbMulti F082 External Validation

## Decision
`F104_FAIL_FRESH_EXTERNAL_TIER3_NO_PROMOTION`

- model promoted: `False`
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
  "protein_h5ad": "/content/hf_cache/hub/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/protein_intensities_crispr_screen_20240615.h5ad",
  "rna_h5ad": "/content/hf_cache/datasets--xingjiepan--PerturbMulti/snapshots/8aac954eb631b68f6e11171a8313db61cc16c38c/RNA_scaled_crispr_screen_20240615.h5ad",
  "split_summary": {
    "all_eval_splits_present": true,
    "rows": [
      {
        "cells": 6836,
        "guides": 34,
        "noncontrol_genes": 20,
        "noncontrol_guides": 34,
        "split": "alternate_test"
      },
      {
        "cells": 4651,
        "guides": 36,
        "noncontrol_genes": 20,
        "noncontrol_guides": 36,
        "split": "test"
      },
      {
        "cells": 57306,
        "guides": 310,
        "noncontrol_genes": 142,
        "noncontrol_guides": 264,
        "split": "train"
      },
      {
        "cells": 9878,
        "guides": 37,
        "noncontrol_genes": 20,
        "noncontrol_guides": 37,
        "split": "validation"
      }
    ]
  }
}
```

## Summary Metrics
method	split	mean_transition_improvement	mean_delta_cosine	mean_recall_at_1	mean_delta_rank	mean_magnitude_ratio	mean_rna_to_image_recall_at_1	mean_image_to_rna_recall_at_1	max_identity_violation	max_leakage_flag	floor_transition_improvement	floor_delta_cosine	floor_recall_at_1	floor_gap_transition_improvement	floor_gap_delta_cosine	floor_gap_recall_at_1
F082_delta_calibrated	alternate_test	0.03357650623114258	0.13104600224090032	0.0196078431372549	10.754879341174671	0.8980985699354881	0.09803921568627451	0.08823529411764706	0.0	0.0	0.03831348152925784	0.11678073023072562	0.0	-0.004736975298115259	0.014265272010174704	0.0196078431372549
F082_delta_calibrated	test	-0.013883321123527631	0.09553785311742259	0.037037037037037035	11.109366786272032	0.8564501906277148	0.0462962962962963	0.06481481481481481	0.0	0.0	-0.027866439732316683	0.06391900084712711	0.08333333333333333	0.013983118608789051	0.03161885227029548	-0.046296296296296294
F082_delta_calibrated	validation	-0.015826248155331585	0.02553079211637924	0.009009009009009009	10.753262807970595	0.9334013603872963	0.12612612612612614	0.0900900900900901	0.0	0.0	-0.021152621427227417	0.03428382831295945	0.0	0.005326373271895831	-0.008753036196580213	0.009009009009009009
F082_no_delta_calibration	alternate_test	0.023887970250591417	0.11087605381696959	0.0392156862745098	13.95819959816942	0.8371132610259387					0.03831348152925784	0.11678073023072562	0.0	-0.014425511278666425	-0.005904676413756027	0.0392156862745098
F082_no_delta_calibration	test	-0.03396269210002009	0.06555171112460785	0.037037037037037035	14.210072190205196	0.7893947381777907					-0.027866439732316683	0.06391900084712711	0.08333333333333333	-0.006096252367703409	0.0016327102774807423	-0.046296296296296294
F082_no_delta_calibration	validation	-0.0001441886935581345	0.04848549178395408	0.018018018018018018	14.015560193765943	0.843535556090536					-0.021152621427227417	0.03428382831295945	0.0	0.021008432733669282	0.014201663470994629	0.018018018018018018
no_residual_source_as_target	alternate_test	0.0	0.0	0.02941176470588235	0.0	0.0					0.03831348152925784	0.11678073023072562	0.0	-0.03831348152925784	-0.11678073023072562	0.02941176470588235
no_residual_source_as_target	test	0.0	0.0	0.027777777777777776	0.0	0.0					-0.027866439732316683	0.06391900084712711	0.08333333333333333	0.027866439732316683	-0.06391900084712711	-0.05555555555555555
no_residual_source_as_target	validation	0.0	0.0	0.02702702702702703	0.0	0.0					-0.021152621427227417	0.03428382831295945	0.0	0.021152621427227417	-0.03428382831295945	0.02702702702702703
protected_full_ridge_floor	alternate_test	0.03831348152925784	0.11678073023072562	0.0	11.623156519336021	0.8788342943745971					0.03831348152925784	0.11678073023072562	0.0	0.0	0.0	0.0
protected_full_ridge_floor	test	-0.027866439732316683	0.06391900084712711	0.08333333333333333	11.251232201077167	0.9685456869394123					-0.027866439732316683	0.06391900084712711	0.08333333333333333	0.0	0.0	0.0
protected_full_ridge_floor	validation	-0.021152621427227417	0.03428382831295945	0.0	11.35549789153282	0.9699642722239744					-0.021152621427227417	0.03428382831295945	0.0	0.0	0.0	0.0
source_as_target	alternate_test	0.0	0.0	0.02941176470588235	0.0	0.0					0.03831348152925784	0.11678073023072562	0.0	-0.03831348152925784	-0.11678073023072562	0.02941176470588235
source_as_target	test	0.0	0.0	0.027777777777777776	0.0	0.0					-0.027866439732316683	0.06391900084712711	0.08333333333333333	0.027866439732316683	-0.06391900084712711	-0.05555555555555555
source_as_target	validation	0.0	0.0	0.02702702702702703	0.0	0.0					-0.021152621427227417	0.03428382831295945	0.0	0.021152621427227417	-0.03428382831295945	0.02702702702702703

## Baseline Metrics
method	split	transition_improvement	delta_cosine	recall_at_1	delta_rank	magnitude_ratio
F082_no_delta_calibration	alternate_test	0.023887970250591417	0.11087605381696959	0.0392156862745098	13.95819959816942	0.8371132610259387
F082_no_delta_calibration	test	-0.03396269210002009	0.06555171112460785	0.037037037037037035	14.210072190205196	0.7893947381777907
F082_no_delta_calibration	validation	-0.0001441886935581345	0.04848549178395408	0.018018018018018018	14.015560193765943	0.843535556090536
no_residual_source_as_target	alternate_test	0.0	0.0	0.02941176470588235	0.0	0.0
no_residual_source_as_target	test	0.0	0.0	0.027777777777777776	0.0	0.0
no_residual_source_as_target	validation	0.0	0.0	0.02702702702702703	0.0	0.0
protected_full_ridge_floor	alternate_test	0.03831348152925784	0.11678073023072562	0.0	11.623156519336021	0.8788342943745971
protected_full_ridge_floor	test	-0.027866439732316683	0.06391900084712711	0.08333333333333333	11.251232201077167	0.9685456869394123
protected_full_ridge_floor	validation	-0.021152621427227417	0.03428382831295945	0.0	11.35549789153282	0.9699642722239744
source_as_target	alternate_test	0.0	0.0	0.02941176470588235	0.0	0.0
source_as_target	test	0.0	0.0	0.027777777777777776	0.0	0.0
source_as_target	validation	0.0	0.0	0.02702702702702703	0.0	0.0

## Leakage And Identity
- condition_key/biological_key/exact one-hot descriptors were not used.
- action descriptors were train-scaled gene-symbol/control-expression descriptors.
- PCA, standardization, and delta calibration used train split only.
- identity_violation and leakage_flag are reported in the summary metrics.
