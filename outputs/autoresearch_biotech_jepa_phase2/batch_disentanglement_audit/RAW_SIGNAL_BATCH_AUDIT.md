# Raw Signal Batch Audit

## Raw / Protected Train-To-Eval Batch Probes

| dataset | seed | eval_split | representation | balanced_accuracy | eval_majority_accuracy | balanced_accuracy_ci_low | balanced_accuracy_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synth_micro | 0 | test | raw_rna_pseudobulk | 0.6875 | 0.5 | 0.5005 | 0.8327 |
| synth_micro | 0 | test | raw_count_pseudobulk | 0.8125 | 0.5 | 0.6766 | 0.9359 |
| synth_micro | 0 | test | raw_image_pooled_pixels | 1 | 0.5 | 1 | 1 |
| synth_micro | 0 | test | protected_pls_rna_latent | 0.4375 | 0.5 | 0.2614 | 0.601 |
| synth_micro | 0 | test | protected_pls_image_latent | 0.5 | 0.5 | 0.3631 | 0.711 |
| synth_micro | 0 | test | random_gaussian_16d | 0.5 | 0.5 | 0.3404 | 0.6773 |
| synth_micro | 0 | test | metadata_only_excluding_batch | 0.5 | 0.5 | 0.5 | 0.5 |
| synth_micro | 0 | test | synthetic_oracle_true_z_bio_audit_only | 0.5 | 0.5 | 0.3276 | 0.6865 |
| synth_micro | 0 | test | synthetic_oracle_true_z_tech_audit_only | 1 | 0.5 | 1 | 1 |
| synth_micro | 1 | test | raw_rna_pseudobulk | 0.6562 | 0.5 | 0.4666 | 0.8341 |
| synth_micro | 1 | test | raw_count_pseudobulk | 0.7188 | 0.5 | 0.5334 | 0.87 |
| synth_micro | 1 | test | raw_image_pooled_pixels | 1 | 0.5 | 1 | 1 |
| synth_micro | 1 | test | protected_pls_rna_latent | 0.4062 | 0.5 | 0.2353 | 0.5888 |
| synth_micro | 1 | test | protected_pls_image_latent | 0.5 | 0.5 | 0.3194 | 0.6701 |
| synth_micro | 1 | test | random_gaussian_16d | 0.4688 | 0.5 | 0.3116 | 0.606 |
| synth_micro | 1 | test | metadata_only_excluding_batch | 0.5 | 0.5 | 0.5 | 0.5 |
| synth_micro | 1 | test | synthetic_oracle_true_z_bio_audit_only | 0.5312 | 0.5 | 0.3426 | 0.721 |
| synth_micro | 1 | test | synthetic_oracle_true_z_tech_audit_only | 0.9062 | 0.5 | 0.7794 | 1 |
| synth_micro | 2 | test | raw_rna_pseudobulk | 0.7188 | 0.5 | 0.6031 | 0.8433 |
| synth_micro | 2 | test | raw_count_pseudobulk | 0.7812 | 0.5 | 0.6335 | 0.9037 |
| synth_micro | 2 | test | raw_image_pooled_pixels | 1 | 0.5 | 1 | 1 |
| synth_micro | 2 | test | protected_pls_rna_latent | 0.5 | 0.5 | 0.3121 | 0.6563 |
| synth_micro | 2 | test | protected_pls_image_latent | 0.625 | 0.5 | 0.4716 | 0.7661 |
| synth_micro | 2 | test | random_gaussian_16d | 0.5312 | 0.5 | 0.375 | 0.6807 |
| synth_micro | 2 | test | metadata_only_excluding_batch | 0.5 | 0.5 | 0.5 | 0.5 |
| synth_micro | 2 | test | synthetic_oracle_true_z_bio_audit_only | 0.5 | 0.5 | 0.3415 | 0.6489 |
| synth_micro | 2 | test | synthetic_oracle_true_z_tech_audit_only | 0.9688 | 0.5 | 0.8691 | 1 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | raw_rna_pseudobulk | 0.7037 | 0.3333 | 0.6037 | 0.8163 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | raw_count_pseudobulk | 0.8889 | 0.3333 | 0.8188 | 0.9459 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | raw_image_pooled_pixels | 0.9012 | 0.3333 | 0.8403 | 0.9563 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | protected_pls_rna_latent | 0.3457 | 0.3333 | 0.2366 | 0.4341 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | protected_pls_image_latent | 0.3333 | 0.3333 | 0.2437 | 0.4218 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | random_gaussian_16d | 0.3086 | 0.3333 | 0.1927 | 0.4119 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | metadata_only_excluding_batch | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | synthetic_oracle_true_z_bio_audit_only | 0.3457 | 0.3333 | 0.2405 | 0.4492 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | synthetic_oracle_true_z_tech_audit_only | 1 | 0.3333 | 1 | 1 |
| synth_heldout_perturbation_lite | 1 | test_heldout_perturbation | raw_rna_pseudobulk | 0.5802 | 0.3333 | 0.4938 | 0.6927 |
| synth_heldout_perturbation_lite | 1 | test_heldout_perturbation | raw_count_pseudobulk | 0.8025 | 0.3333 | 0.7196 | 0.883 |
| synth_heldout_perturbation_lite | 1 | test_heldout_perturbation | raw_image_pooled_pixels | 0.9877 | 0.3333 | 0.9561 | 1 |
| synth_heldout_perturbation_lite | 1 | test_heldout_perturbation | protected_pls_rna_latent | 0.3333 | 0.3333 | 0.2512 | 0.4235 |

## Technical Duplicate / Split-Half Ceiling

| dataset | seed | eval_split | status | bag_count | rna_to_rna_same_bag_recall@1 | image_to_image_same_bag_recall@1 | rna_to_image_same_bio_recall@1 | rna_to_image_same_bio_recall@5 | rna_half_batch_probe_balanced_accuracy | rna_half_batch_probe_majority_accuracy | image_half_batch_probe_balanced_accuracy | image_half_batch_probe_majority_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_micro | 0 | test | ok | 32 | 0.1562 | 0.8438 | 0.4688 | 0.8125 | 0.7188 | 0.5 | 0.9844 | 0.5 |
| synth_micro | 1 | test | ok | 32 | 0.25 | 0.7812 | 0.5312 | 0.875 | 0.5 | 0.5 | 0.7969 | 0.5 |
| synth_micro | 2 | test | ok | 32 | 0.1562 | 0.8125 | 0.5625 | 0.7812 | 0.8281 | 0.5 | 1 | 0.5 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | ok | 324 | 0.1605 | 0.929 | 0.2747 | 0.5154 | 0.6543 | 0.3333 | 1 | 0.3333 |
| synth_heldout_perturbation_lite | 1 | test_heldout_perturbation | ok | 324 | 0.1358 | 0.8827 | 0.1883 | 0.3333 | 0.5633 | 0.3333 | 0.9985 | 0.3333 |
| synth_heldout_perturbation_lite | 2 | test_heldout_perturbation | ok | 324 | 0.1914 | 0.8056 | 0.142 | 0.3179 | 0.6358 | 0.3333 | 0.9938 | 0.3333 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | ok | 540 | 0.1222 | 0.9093 | 0.1704 | 0.3481 | 0.6935 | 0.3333 | 1 | 0.3333 |
| synth_dose_extrapolation_lite | 1 | test_heldout_dose | ok | 540 | 0.1222 | 0.8463 | 0.1093 | 0.2593 | 0.6148 | 0.3333 | 1 | 0.3333 |
| synth_dose_extrapolation_lite | 2 | test_heldout_dose | ok | 540 | 0.1056 | 0.787 | 0.1074 | 0.2407 | 0.6454 | 0.3333 | 0.9991 | 0.3333 |
| synth_batch_confound_lite | 0 | test | ok | 126 | 0.3492 | 0.8413 | 0.3016 | 0.627 | 0.6472 | 0.3571 | 0.9537 | 0.3571 |
| synth_batch_confound_lite | 1 | test | ok | 126 | 0.373 | 0.8254 | 0.1587 | 0.5159 | 0.6852 | 0.3571 | 0.9519 | 0.3571 |
| synth_batch_confound_lite | 2 | test | ok | 126 | 0.4206 | 0.8651 | 0.1667 | 0.5159 | 0.7111 | 0.3571 | 0.9139 | 0.3571 |
