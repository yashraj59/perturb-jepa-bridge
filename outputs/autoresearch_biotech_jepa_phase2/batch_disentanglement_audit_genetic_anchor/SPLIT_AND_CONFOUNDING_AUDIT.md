# Split And Confounding Audit

## Anchor Summary

| dataset | seed | eval_split | bio_key_count | cross_batch_replicate_count_per_bio_key_mean | bio_keys_with_1_batch_only | bio_keys_with_2plus_batches | bio_keys_with_3plus_batches | train_bio_key_count | train_bio_keys_with_2plus_batches | train_cross_batch_anchor_fraction | eval_bio_key_count | eval_bio_keys_with_2plus_batches | eval_cross_batch_replicate_fraction | eval_target_count | eval_targets_with_cross_batch_train_anchor | eval_targets_with_only_same_batch_anchor | eval_targets_with_no_bio_anchor | fraction_of_eval_targets_with_cross_batch_train_anchor | fraction_of_eval_targets_with_only_same_batch_anchor | fraction_of_eval_targets_with_no_bio_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | 36 | 3 | 0 | 36 | 36 | 24 | 24 | 1 | 9 | 9 | 1 | 27 | 0 | 0 | 27 | 0 | 0 | 1 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | 36 | 3 | 0 | 36 | 36 | 24 | 24 | 1 | 9 | 9 | 1 | 27 | 0 | 0 | 27 | 0 | 0 | 1 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | 36 | 3 | 0 | 36 | 36 | 24 | 24 | 1 | 9 | 9 | 1 | 27 | 0 | 0 | 27 | 0 | 0 | 1 |
| synth_batch_confound_lite | 0 | test | 108 | 1.167 | 99 | 9 | 9 | 99 | 0 | 0 | 99 | 0 | 0 | 99 | 0 | 99 | 0 | 0 | 1 | 0 |
| synth_batch_confound_lite | 1 | test | 108 | 1.167 | 99 | 9 | 9 | 99 | 0 | 0 | 99 | 0 | 0 | 99 | 0 | 99 | 0 | 0 | 1 | 0 |
| synth_batch_confound_lite | 2 | test | 108 | 1.167 | 99 | 9 | 9 | 99 | 0 | 0 | 99 | 0 | 0 | 99 | 0 | 99 | 0 | 0 | 1 | 0 |

## Strongest Batch Associations

| dataset | seed | eval_split | table | n_rows | batch_nunique | other_nunique | cramers_v | normalized_mutual_information | max_cell_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_batch_confound_lite | 2 | test | batch_id_x_condition_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 2 | test | batch_id_x_biological_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 1 | test | batch_id_x_condition_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 1 | test | batch_id_x_biological_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 0 | test | batch_id_x_biological_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 0 | test | batch_id_x_condition_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 0 | test | batch_id_x_perturbation_id | 378 | 3 | 12 | 0.8851 | 0.4907 | 0.07143 |
| synth_batch_confound_lite | 2 | test | batch_id_x_perturbation_id | 378 | 3 | 12 | 0.8851 | 0.4907 | 0.07143 |
| synth_batch_confound_lite | 1 | test | batch_id_x_perturbation_id | 378 | 3 | 12 | 0.8851 | 0.4907 | 0.07143 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | batch_id_x_cell_line_id | 270 | 3 | 3 | 0 | 0 | 0.1111 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | batch_id_x_dose | 270 | 3 | 1 | 0 | 0 | 0.3333 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | batch_id_x_perturbation_id | 270 | 3 | 12 | 0 | 0 | 0.03333 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | batch_id_x_condition_key | 270 | 3 | 36 | 0 | 0 | 0.01111 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | batch_id_x_split | 270 | 3 | 4 | 0 | 0 | 0.1 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | batch_id_x_perturbation_id | 270 | 3 | 12 | 0 | 0 | 0.03333 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | batch_id_x_biological_key | 270 | 3 | 36 | 0 | 0 | 0.01111 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | batch_id_x_cell_line_id | 270 | 3 | 3 | 0 | 0 | 0.1111 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | batch_id_x_condition_key | 270 | 3 | 36 | 0 | 0 | 0.01111 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | batch_id_x_split | 270 | 3 | 4 | 0 | 0 | 0.1 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | batch_id_x_dose | 270 | 3 | 1 | 0 | 0 | 0.3333 |
