# Split And Confounding Audit

## Anchor Summary

| dataset | seed | eval_split | bio_key_count | cross_batch_replicate_count_per_bio_key_mean | bio_keys_with_1_batch_only | bio_keys_with_2plus_batches | bio_keys_with_3plus_batches | eval_target_count | eval_targets_with_cross_batch_train_anchor | eval_targets_with_only_same_batch_anchor | eval_targets_with_no_bio_anchor | fraction_of_eval_targets_with_cross_batch_train_anchor | fraction_of_eval_targets_with_only_same_batch_anchor | fraction_of_eval_targets_with_no_bio_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_micro | 0 | test | 16 | 2 | 0 | 16 | 0 | 24 | 24 | 0 | 0 | 1 | 0 | 0 |
| synth_micro | 1 | test | 16 | 2 | 0 | 16 | 0 | 24 | 24 | 0 | 0 | 1 | 0 | 0 |
| synth_micro | 2 | test | 16 | 2 | 0 | 16 | 0 | 24 | 24 | 0 | 0 | 1 | 0 | 0 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | 108 | 3 | 0 | 108 | 108 | 81 | 0 | 0 | 81 | 0 | 0 | 1 |
| synth_heldout_perturbation_lite | 1 | test_heldout_perturbation | 108 | 3 | 0 | 108 | 108 | 81 | 0 | 0 | 81 | 0 | 0 | 1 |
| synth_heldout_perturbation_lite | 2 | test_heldout_perturbation | 108 | 3 | 0 | 108 | 108 | 81 | 0 | 0 | 81 | 0 | 0 | 1 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | 180 | 3 | 0 | 180 | 180 | 198 | 0 | 0 | 198 | 0 | 0 | 1 |
| synth_dose_extrapolation_lite | 1 | test_heldout_dose | 180 | 3 | 0 | 180 | 180 | 198 | 0 | 0 | 198 | 0 | 0 | 1 |
| synth_dose_extrapolation_lite | 2 | test_heldout_dose | 180 | 3 | 0 | 180 | 180 | 198 | 0 | 0 | 198 | 0 | 0 | 1 |
| synth_batch_confound_lite | 0 | test | 108 | 1.167 | 99 | 9 | 9 | 99 | 0 | 99 | 0 | 0 | 1 | 0 |
| synth_batch_confound_lite | 1 | test | 108 | 1.167 | 99 | 9 | 9 | 99 | 0 | 99 | 0 | 0 | 1 | 0 |
| synth_batch_confound_lite | 2 | test | 108 | 1.167 | 99 | 9 | 9 | 99 | 0 | 99 | 0 | 0 | 1 | 0 |

## Strongest Batch Associations

| dataset | seed | eval_split | table | n_rows | batch_nunique | other_nunique | cramers_v | normalized_mutual_information | max_cell_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_batch_confound_lite | 0 | test | batch_id_x_biological_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 0 | test | batch_id_x_condition_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 1 | test | batch_id_x_biological_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 1 | test | batch_id_x_condition_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 2 | test | batch_id_x_biological_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 2 | test | batch_id_x_condition_key | 378 | 3 | 108 | 0.8851 | 0.3013 | 0.007937 |
| synth_batch_confound_lite | 0 | test | batch_id_x_perturbation_id | 378 | 3 | 12 | 0.8851 | 0.4907 | 0.07143 |
| synth_batch_confound_lite | 2 | test | batch_id_x_perturbation_id | 378 | 3 | 12 | 0.8851 | 0.4907 | 0.07143 |
| synth_batch_confound_lite | 1 | test | batch_id_x_perturbation_id | 378 | 3 | 12 | 0.8851 | 0.4907 | 0.07143 |
| synth_micro | 1 | test | batch_id_x_perturbation_id | 96 | 2 | 4 | 0 | 0 | 0.125 |
| synth_micro | 0 | test | batch_id_x_biological_key | 96 | 2 | 16 | 0 | 0 | 0.03125 |
| synth_micro | 0 | test | batch_id_x_split | 96 | 2 | 3 | 0 | 0 | 0.1667 |
| synth_micro | 0 | test | batch_id_x_condition_key | 96 | 2 | 16 | 0 | 0 | 0.03125 |
| synth_micro | 0 | test | batch_id_x_cell_line_id | 96 | 2 | 2 | 0 | 0 | 0.25 |
| synth_micro | 0 | test | batch_id_x_dose | 96 | 2 | 2 | 0 | 0 | 0.25 |
| synth_micro | 0 | test | batch_id_x_perturbation_id | 96 | 2 | 4 | 0 | 0 | 0.125 |
| synth_micro | 1 | test | batch_id_x_dose | 96 | 2 | 2 | 0 | 0 | 0.25 |
| synth_micro | 1 | test | batch_id_x_cell_line_id | 96 | 2 | 2 | 0 | 0 | 0.25 |
| synth_micro | 1 | test | batch_id_x_split | 96 | 2 | 3 | 0 | 0 | 0.1667 |
| synth_micro | 1 | test | batch_id_x_condition_key | 96 | 2 | 16 | 0 | 0 | 0.03125 |
