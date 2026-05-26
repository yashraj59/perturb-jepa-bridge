# Representation Audit

## Latent Batch Signal

| dataset | seed | eval_split | reference | state | status | n | dim | batch_probe_balanced_accuracy | batch_probe_majority_accuracy | batch_probe_resub_balanced_accuracy | effective_rank | std_mean | batch_probe_excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | image_online_z_shared | ok | 99 | 32 | 0.9784 | 0.3636 | 1 | 9.043 | 0.1714 | 0.6148 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | zero_step_bioaction_encoder | rna_online_z_shared | ok | 27 | 32 | 0.8889 | 0.3333 | 1 | 16.38 | 0.1755 | 0.5556 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 99 | 32 | 0.9043 | 0.3636 | 0.9815 | 24.8 | 0.2247 | 0.5407 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 99 | 32 | 0.8704 | 0.3636 | 0.9815 | 24.06 | 0.1992 | 0.5067 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | transition_predicted_latent | ok | 99 | 32 | 0.8457 | 0.3636 | 0.9444 | 21.88 | 0.1387 | 0.482 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | rna_online_z_shared | ok | 99 | 32 | 0.8395 | 0.3636 | 0.9815 | 24.01 | 0.2209 | 0.4759 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | joint_online_z_shared | ok | 99 | 32 | 0.8333 | 0.3636 | 0.9444 | 21.63 | 0.1403 | 0.4697 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | transition_predicted_latent | ok | 99 | 32 | 0.8302 | 0.3636 | 0.9259 | 21.92 | 0.1464 | 0.4666 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | joint_online_z_shared | ok | 99 | 32 | 0.821 | 0.3636 | 0.9167 | 21.72 | 0.1466 | 0.4574 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | zero_step_bioaction_encoder | rna_online_z_shared | ok | 27 | 32 | 0.7778 | 0.3333 | 1 | 17.23 | 0.1954 | 0.4444 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | joint_online_z_shared | ok | 99 | 32 | 0.7778 | 0.3636 | 0.9907 | 21.43 | 0.1278 | 0.4141 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | transition_predicted_latent | ok | 99 | 32 | 0.7593 | 0.3636 | 0.9907 | 21.56 | 0.1276 | 0.3956 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | image_ema_teacher_target | ok | 99 | 32 | 0.75 | 0.3636 | 0.8457 | 16.44 | 0.1885 | 0.3864 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | zero_step_bioaction_encoder | rna_online_z_shared | ok | 27 | 32 | 0.7037 | 0.3333 | 1 | 16.78 | 0.2077 | 0.3704 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | image_ema_teacher_target | ok | 99 | 32 | 0.7191 | 0.3636 | 0.821 | 16.47 | 0.1901 | 0.3555 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | zero_step_bioaction_encoder | transition_predicted_latent | ok | 27 | 32 | 0.6667 | 0.3333 | 1 | 14.51 | 0.1342 | 0.3333 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | zero_step_bioaction_encoder | joint_online_z_shared | ok | 27 | 32 | 0.6667 | 0.3333 | 1 | 14.29 | 0.1134 | 0.3333 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | image_online_z_shared | ok | 99 | 32 | 0.6636 | 0.3636 | 0.9568 | 9.918 | 0.1712 | 0.2999 |
| synth_genetic_anchor_lite | 0 | test_heldout_perturbation | zero_step_bioaction_encoder | joint_online_z_shared | ok | 27 | 32 | 0.6296 | 0.3333 | 1 | 14.46 | 0.1343 | 0.2963 |
| synth_genetic_anchor_lite | 2 | test_heldout_perturbation | zero_step_bioaction_encoder | transition_predicted_latent | ok | 27 | 32 | 0.6296 | 0.3333 | 1 | 14.41 | 0.1138 | 0.2963 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | image_online_z_shared | ok | 99 | 32 | 0.6512 | 0.3636 | 0.9352 | 8.824 | 0.1623 | 0.2876 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | image_ema_teacher_target | ok | 99 | 32 | 0.6296 | 0.3636 | 0.8241 | 16.76 | 0.1967 | 0.266 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | joint_ema_teacher_target | ok | 99 | 32 | 0.5957 | 0.3636 | 0.7654 | 26.77 | 0.1544 | 0.232 |
| synth_batch_confound_lite | 0 | test | zero_step_bioaction_encoder | transition_joint_ema_teacher_target | ok | 99 | 32 | 0.5957 | 0.3636 | 0.7654 | 26.77 | 0.1544 | 0.232 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | transition_joint_ema_teacher_target | ok | 99 | 32 | 0.5741 | 0.3636 | 0.8056 | 26.69 | 0.1404 | 0.2104 |
| synth_batch_confound_lite | 2 | test | zero_step_bioaction_encoder | joint_ema_teacher_target | ok | 99 | 32 | 0.5741 | 0.3636 | 0.8056 | 26.69 | 0.1404 | 0.2104 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | zero_step_bioaction_encoder | joint_online_z_shared | ok | 27 | 32 | 0.4815 | 0.3333 | 0.9259 | 14.91 | 0.1334 | 0.1481 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | joint_ema_teacher_target | ok | 99 | 32 | 0.5031 | 0.3636 | 0.716 | 26.81 | 0.1519 | 0.1395 |
| synth_batch_confound_lite | 1 | test | zero_step_bioaction_encoder | transition_joint_ema_teacher_target | ok | 99 | 32 | 0.5031 | 0.3636 | 0.716 | 26.81 | 0.1519 | 0.1395 |
| synth_genetic_anchor_lite | 1 | test_heldout_perturbation | zero_step_bioaction_encoder | transition_predicted_latent | ok | 27 | 32 | 0.4444 | 0.3333 | 0.9259 | 15.16 | 0.1323 | 0.1111 |

## Loss Geometry

| experiment | steps_logged | loss_total_last | loss_total_mean | weighted_jepa_sum_last | weighted_aux_sum_last | loss_batch_invariance_last | weighted_batch_invariance_last | jepa_to_aux_ratio_last | vicreg_covariance_last | barlow_last | gradient_norms_logged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP001_familyA_minimal_heldout_perturb_seed0_s32 | 50 | 2.088 | 3.716 | 0.6665 | 0 | 0 | 0 | nan | 7.982 | 11.92 | 0 |
| EXP002_familyA_minimal_heldout_dose_seed0_s32_cuda | 50 | 2.014 | 3.641 | 0.6075 | 0 | 0 | 0 | nan | 8.172 | 12.59 | 0 |
| EXP003_familyA_minimal_synth_micro_seed0_s32_cuda | 50 | 1.586 | 4.077 | 0.3387 | 0 | 0 | 0 | nan | 16.96 | 1 | 0 |
| EXP004_familyF_batch_invariant_synth_micro_seed0_s32_cuda | 50 | 1.612 | 4.093 | 0.3374 | 0.03319 | 0.06638 | 0.03319 | 20.71 | 16.84 | 1 | 0 |
| EXP005_familyF_batch_invariant_strong_synth_micro_seed0_s32_cuda | 50 | 1.8 | 4.217 | 0.3407 | 0.1965 | 0.03929 | 0.1965 | 3.618 | 16.85 | 1 | 0 |
