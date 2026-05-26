# Representation Audit

## Latent Batch Signal

| dataset | seed | eval_split | reference | state | status | n | dim | batch_probe_balanced_accuracy | batch_probe_majority_accuracy | batch_probe_resub_balanced_accuracy | effective_rank | std_mean | checkpoint | batch_probe_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_dose_extrapolation_lite | 1 | test_heldout_dose | zero_step_bioaction_encoder | transition_predicted_latent | ok | 198 | 32 | 1 | 0.3333 | 1 | 13.51 | 0.1203 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | EXP002_familyA_heldout_dose | image_online_z_shared | ok | 198 | 32 | 1 | 0.3333 | 1 | 4.987 | 0.1414 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 1 | test_heldout_dose | zero_step_bioaction_encoder | image_online_z_shared | ok | 198 | 32 | 1 | 0.3333 | 1 | 4.575 | 0.1677 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 1 | test_heldout_dose | zero_step_bioaction_encoder | rna_online_z_shared | ok | 198 | 32 | 1 | 0.3333 | 1 | 13.02 | 0.188 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | zero_step_bioaction_encoder | rna_online_z_shared | ok | 198 | 32 | 1 | 0.3333 | 1 | 13.21 | 0.1825 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 1 | test_heldout_dose | zero_step_bioaction_encoder | joint_online_z_shared | ok | 198 | 32 | 1 | 0.3333 | 1 | 11.67 | 0.1219 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | zero_step_bioaction_encoder | image_online_z_shared | ok | 198 | 32 | 1 | 0.3333 | 1 | 4.498 | 0.1684 | nan | 0.6667 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | zero_step_bioaction_encoder | transition_predicted_latent | ok | 198 | 32 | 0.9697 | 0.3333 | 1 | 12.86 | 0.134 | nan | 0.6364 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | zero_step_bioaction_encoder | joint_online_z_shared | ok | 198 | 32 | 0.9646 | 0.3333 | 1 | 11.08 | 0.134 | nan | 0.6313 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | EXP002_familyA_heldout_dose | rna_online_z_shared | ok | 198 | 32 | 0.9596 | 0.3333 | 1 | 13.21 | 0.1744 | nan | 0.6263 |
| synth_dose_extrapolation_lite | 2 | test_heldout_dose | zero_step_bioaction_encoder | rna_online_z_shared | ok | 198 | 32 | 0.9596 | 0.3333 | 1 | 13.18 | 0.1579 | nan | 0.6263 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | image_online_z_shared | ok | 99 | 32 | 0.9784 | 0.3636 | 1 | 9.043 | 0.1714 | nan | 0.6148 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | EXP002_familyA_heldout_dose | joint_online_z_shared | ok | 198 | 32 | 0.9242 | 0.3333 | 1 | 10.87 | 0.1334 | nan | 0.5909 |
| synth_dose_extrapolation_lite | 0 | test_heldout_dose | EXP002_familyA_heldout_dose | transition_predicted_latent | ok | 198 | 32 | 0.9192 | 0.3333 | 0.9697 | 11.9 | 0.1333 | nan | 0.5859 |
| synth_dose_extrapolation_lite | 2 | test_heldout_dose | zero_step_bioaction_encoder | joint_online_z_shared | ok | 198 | 32 | 0.8889 | 0.3333 | 0.9444 | 11.26 | 0.1092 | nan | 0.5556 |
| synth_dose_extrapolation_lite | 2 | test_heldout_dose | zero_step_bioaction_encoder | transition_predicted_latent | ok | 198 | 32 | 0.8788 | 0.3333 | 0.9444 | 13.27 | 0.1093 | nan | 0.5455 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 99 | 32 | 0.9043 | 0.3636 | 0.9815 | 24.8 | 0.2247 | nan | 0.5407 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 99 | 32 | 0.8704 | 0.3636 | 0.9815 | 24.06 | 0.1992 | nan | 0.5067 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | EXP001_familyA_heldout_perturb | image_online_z_shared | ok | 81 | 32 | 0.8395 | 0.3333 | 1 | 8.977 | 0.1276 | nan | 0.5062 |
| synth_micro | 0 | test | zero_step_bioaction_encoder | image_online_z_shared | ok | 24 | 32 | 1 | 0.5 | 1 | 4.26 | 0.1436 | nan | 0.5 |
| synth_micro | 2 | test | zero_step_bioaction_encoder | image_online_z_shared | ok | 24 | 32 | 1 | 0.5 | 1 | 4.454 | 0.104 | nan | 0.5 |
| synth_micro | 2 | test | zero_step_bioaction_encoder | joint_online_z_shared | ok | 24 | 32 | 1 | 0.5 | 1 | 6.588 | 0.1133 | nan | 0.5 |
| synth_micro | 2 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 24 | 32 | 1 | 0.5 | 1 | 6.654 | 0.1872 | nan | 0.5 |
| synth_micro | 1 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 24 | 32 | 1 | 0.5 | 1 | 6.688 | 0.1999 | nan | 0.5 |
| synth_micro | 2 | test | zero_step_bioaction_encoder | transition_predicted_latent | ok | 24 | 32 | 1 | 0.5 | 1 | 7.449 | 0.1138 | nan | 0.5 |
| synth_heldout_perturbation_lite | 0 | test_heldout_perturbation | zero_step_bioaction_encoder | image_online_z_shared | ok | 81 | 32 | 0.8272 | 0.3333 | 1 | 7.521 | 0.1697 | nan | 0.4938 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | transition_predicted_latent | ok | 99 | 32 | 0.8457 | 0.3636 | 0.9444 | 21.88 | 0.1387 | nan | 0.482 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 99 | 32 | 0.8395 | 0.3636 | 0.9815 | 24.01 | 0.2209 | nan | 0.4759 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | joint_online_z_shared | ok | 99 | 32 | 0.8333 | 0.3636 | 0.9444 | 21.63 | 0.1403 | nan | 0.4697 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | transition_predicted_latent | ok | 99 | 32 | 0.8302 | 0.3636 | 0.9259 | 21.92 | 0.1464 | nan | 0.4666 |

## Loss Geometry

| experiment | steps_logged | loss_total_last | loss_total_mean | weighted_jepa_sum_last | weighted_aux_sum_last | loss_batch_invariance_last | weighted_batch_invariance_last | jepa_to_aux_ratio_last | vicreg_covariance_last | barlow_last | gradient_norms_logged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP001_familyA_minimal_heldout_perturb_seed0_s32 | 50 | 2.088 | 3.716 | 0.6665 | 0 | 0 | 0 | nan | 7.982 | 11.92 | 0 |
| EXP002_familyA_minimal_heldout_dose_seed0_s32_cuda | 50 | 2.014 | 3.641 | 0.6075 | 0 | 0 | 0 | nan | 8.172 | 12.59 | 0 |
| EXP003_familyA_minimal_synth_micro_seed0_s32_cuda | 50 | 1.586 | 4.077 | 0.3387 | 0 | 0 | 0 | nan | 16.96 | 1 | 0 |
| EXP004_familyF_batch_invariant_synth_micro_seed0_s32_cuda | 50 | 1.612 | 4.093 | 0.3374 | 0.03319 | 0.06638 | 0.03319 | 20.71 | 16.84 | 1 | 0 |
| EXP005_familyF_batch_invariant_strong_synth_micro_seed0_s32_cuda | 50 | 1.8 | 4.217 | 0.3407 | 0.1965 | 0.03929 | 0.1965 | 3.618 | 16.85 | 1 | 0 |
