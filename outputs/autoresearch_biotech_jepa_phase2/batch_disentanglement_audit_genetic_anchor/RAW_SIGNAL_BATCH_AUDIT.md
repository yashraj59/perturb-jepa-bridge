# Raw Signal Batch Audit

## Raw / Protected Train-To-Eval Batch Probes

| dataset | seed | eval_split | representation | balanced_accuracy | eval_majority_accuracy | balanced_accuracy_ci_low | balanced_accuracy_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | raw_rna_pseudobulk | 0.7407 | 0.3333 | 0.5868 | 0.8684 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | raw_count_pseudobulk | 0.7037 | 0.3333 | 0.5653 | 0.8014 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | raw_image_pooled_pixels | 1 | 0.3333 | 1 | 1 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | protected_pls_rna_latent | 0.3333 | 0.3333 | 0.1986 | 0.4954 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | protected_pls_image_latent | 0.2963 | 0.3333 | 0.1342 | 0.491 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | random_gaussian_16d | 0.2222 | 0.3333 | 0.06217 | 0.4094 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | metadata_only_excluding_batch | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | synthetic_oracle_true_z_bio_audit_only | 0.3333 | 0.3333 | 0.1916 | 0.4678 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | synthetic_oracle_true_z_tech_audit_only | 1 | 0.3333 | 1 | 1 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | raw_rna_pseudobulk | 0.5185 | 0.3333 | 0.3087 | 0.7058 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | raw_count_pseudobulk | 0.8148 | 0.3333 | 0.6556 | 0.9227 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | raw_image_pooled_pixels | 0.8889 | 0.3333 | 0.7844 | 1 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | protected_pls_rna_latent | 0.3333 | 0.3333 | 0.1969 | 0.507 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | protected_pls_image_latent | 0.3333 | 0.3333 | 0.2086 | 0.5 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | random_gaussian_16d | 0.3704 | 0.3333 | 0.2083 | 0.5436 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | metadata_only_excluding_batch | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | synthetic_oracle_true_z_bio_audit_only | 0.3333 | 0.3333 | 0.2019 | 0.4627 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | synthetic_oracle_true_z_tech_audit_only | 1 | 0.3333 | 1 | 1 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | raw_rna_pseudobulk | 0.5926 | 0.3333 | 0.4256 | 0.7834 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | raw_count_pseudobulk | 0.8519 | 0.3333 | 0.732 | 0.9608 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | raw_image_pooled_pixels | 0.9259 | 0.3333 | 0.807 | 1 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | protected_pls_rna_latent | 0.4074 | 0.3333 | 0.2336 | 0.5493 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | protected_pls_image_latent | 0.3333 | 0.3333 | 0.2035 | 0.4619 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | random_gaussian_16d | 0.3333 | 0.3333 | 0.1521 | 0.4957 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | metadata_only_excluding_batch | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | synthetic_oracle_true_z_bio_audit_only | 0.3333 | 0.3333 | 0.1724 | 0.539 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | synthetic_oracle_true_z_tech_audit_only | 1 | 0.3333 | 1 | 1 |
| synth_batch_confound_lite | 0 | test | raw_rna_pseudobulk | 0.5926 | 0.3571 | 0.4939 | 0.6877 |
| synth_batch_confound_lite | 0 | test | raw_count_pseudobulk | 0.8296 | 0.3571 | 0.7673 | 0.8979 |
| synth_batch_confound_lite | 0 | test | raw_image_pooled_pixels | 1 | 0.3571 | 1 | 1 |
| synth_batch_confound_lite | 0 | test | protected_pls_rna_latent | 0.3963 | 0.3571 | 0.3152 | 0.468 |
| synth_batch_confound_lite | 0 | test | protected_pls_image_latent | 0.437 | 0.3571 | 0.3485 | 0.5115 |
| synth_batch_confound_lite | 0 | test | random_gaussian_16d | 0.3019 | 0.3571 | 0.2214 | 0.3744 |
| synth_batch_confound_lite | 0 | test | metadata_only_excluding_batch | 0.8667 | 0.3571 | 0.8097 | 0.9197 |
| synth_batch_confound_lite | 0 | test | synthetic_oracle_true_z_bio_audit_only | 0.7204 | 0.3571 | 0.6563 | 0.7914 |
| synth_batch_confound_lite | 0 | test | synthetic_oracle_true_z_tech_audit_only | 1 | 0.3571 | 1 | 1 |
| synth_batch_confound_lite | 1 | test | raw_rna_pseudobulk | 0.5778 | 0.3571 | 0.4866 | 0.6588 |
| synth_batch_confound_lite | 1 | test | raw_count_pseudobulk | 0.7167 | 0.3571 | 0.6412 | 0.7803 |
| synth_batch_confound_lite | 1 | test | raw_image_pooled_pixels | 0.9852 | 0.3571 | 0.9683 | 1 |
| synth_batch_confound_lite | 1 | test | protected_pls_rna_latent | 0.4759 | 0.3571 | 0.3844 | 0.5542 |

## Technical Duplicate / Split-Half Ceiling

| dataset | seed | eval_split | status | bag_count | rna_to_rna_same_bag_recall@1 | image_to_image_same_bag_recall@1 | rna_to_image_same_bio_recall@1 | rna_to_image_same_bio_recall@5 | rna_half_batch_probe_balanced_accuracy | rna_half_batch_probe_majority_accuracy | image_half_batch_probe_balanced_accuracy | image_half_batch_probe_majority_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | ok | 108 | 0.3241 | 0.9815 | 0.6389 | 0.8889 | 0.6667 | 0.3333 | 0.9722 | 0.3333 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | ok | 108 | 0.3056 | 0.9352 | 0.4352 | 0.7037 | 0.537 | 0.3333 | 0.8935 | 0.3333 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | ok | 108 | 0.2778 | 0.8981 | 0.5278 | 0.713 | 0.5602 | 0.3333 | 0.7824 | 0.3333 |
| synth_batch_confound_lite | 0 | test | ok | 126 | 0.3492 | 0.8413 | 0.3016 | 0.627 | 0.6472 | 0.3571 | 0.9537 | 0.3571 |
| synth_batch_confound_lite | 1 | test | ok | 126 | 0.373 | 0.8254 | 0.1587 | 0.5159 | 0.6852 | 0.3571 | 0.9519 | 0.3571 |
| synth_batch_confound_lite | 2 | test | ok | 126 | 0.4206 | 0.8651 | 0.1667 | 0.5159 | 0.7111 | 0.3571 | 0.9139 | 0.3571 |
