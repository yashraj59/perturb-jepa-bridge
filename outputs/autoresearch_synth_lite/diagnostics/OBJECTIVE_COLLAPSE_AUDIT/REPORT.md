# Objective Collapse Report

## Summary

- Diagnostic runs: 15
- Final collapsed: 0
- Non-collapsed: 15
- Collapsed at step 0: 0
- Transient collapse during train-batch tracing: 1

## Main Finding

The objective ablations do not reproduce final collapse on the traced training condition batches: all 15 final train-batch traces remain above the hard std gate. The same 30-step setup does collapse in the normal Step 0 held-out evaluation path. This points away from a single loss term immediately destroying variance, and toward a split/evaluation geometry problem: the shared representation learned on train condition bags does not survive collection over held-out/test condition bags.

## Non-Collapsed Runs

- `full_attention`: rna_min_std=0.03387371823191643, image_min_std=0.04180159419775009, rna_rank=8.0, image_rank=8.0
- `full_mean`: rna_min_std=0.07088577747344971, image_min_std=0.06591279804706573, rna_rank=8.0, image_rank=8.0
- `rna_reconstruction_only`: rna_min_std=0.02133566327393055, image_min_std=0.04690124839544296, rna_rank=8.0, image_rank=8.0
- `image_reconstruction_only`: rna_min_std=0.040069740265607834, image_min_std=0.03647575154900551, rna_rank=8.0, image_rank=8.0
- `reconstruction_both`: rna_min_std=0.02133566327393055, image_min_std=0.03647575154900551, rna_rank=8.0, image_rank=8.0
- `jepa_only`: rna_min_std=0.029083844274282455, image_min_std=0.026848748326301575, rna_rank=8.0, image_rank=8.0
- `align_only`: rna_min_std=0.011356079019606113, image_min_std=0.012624138034880161, rna_rank=8.0, image_rank=8.0
- `reconstruction_plus_jepa`: rna_min_std=0.03444413095712662, image_min_std=0.04049130529165268, rna_rank=8.0, image_rank=8.0
- `reconstruction_plus_align`: rna_min_std=0.034026019275188446, image_min_std=0.029561152681708336, rna_rank=8.0, image_rank=8.0
- `full_no_adversaries`: rna_min_std=0.033803243190050125, image_min_std=0.041869040578603745, rna_rank=8.0, image_rank=8.0
- `full_no_distribution`: rna_min_std=0.014394856058061123, image_min_std=0.018611827865242958, rna_rank=8.0, image_rank=8.0
- `full_no_reconstruction`: rna_min_std=0.016795102506875992, image_min_std=0.028725387528538704, rna_rank=8.0, image_rank=8.0
- `full_no_jepa`: rna_min_std=0.03111516684293747, image_min_std=0.038939427584409714, rna_rank=8.0, image_rank=8.0
- `full_no_align`: rna_min_std=0.030364707112312317, image_min_std=0.039081472903490067, rna_rank=8.0, image_rank=8.0
- `mean_reconstruction_plus_align`: rna_min_std=0.0572701022028923, image_min_std=0.04586387425661087, rna_rank=8.0, image_rank=8.0

## Held-Out Eval Check

- `attention_step0_eval`: collapse=1, rna_min_std=0.007145956624299288, image_min_std=0.004709864035248756, rna_to_image_recall@1=0.0625, random_recall@1=0.0625, mean_prototype_recall@1=0.3125, batch_probe_balanced_accuracy=0.46875, rna_latent_r2=-0.3971430825795881, image_latent_r2=0.8904288405562112
- `mean_step0_eval`: collapse=1, rna_min_std=0.01232663169503212, image_min_std=0.005121666006743908, rna_to_image_recall@1=0.0625, random_recall@1=0.0625, mean_prototype_recall@1=0.3125, batch_probe_balanced_accuracy=0.5, rna_latent_r2=-0.9756273064498968, image_latent_r2=0.8964751476390077

## Dominant Step-0 Losses

- image_mask: 15

These are raw unweighted term magnitudes recorded for scale inspection. They should not be read as weighted objective dominance in ablations where a term weight is zero.

## Dominant Step-0 Gradient Groups

- image_encoder: 2
- jepa_predictors: 2
- rna_encoder: 11

## Interpretation

The 100-experiment search was not primarily failing because of one auxiliary loss weight. Loss ablations are stable on traced train batches, while held-out Step 0 evaluation still collapses and retrieves at random. Mean pooling improves held-out RNA variance but does not fix image-side collapse or retrieval. Alignment-only is the most fragile train-batch ablation, with transient near-threshold collapse, and removing distribution or reconstruction terms leaves final variance close to the gate. The next phase should audit the condition-bag split/evaluation path and then repair projection/aggregation normalization under that held-out metric, not run another broad architecture sweep.

## Recommendation

Do not use real data yet. Treat the current model as unsafe. Next action should be a focused split-generalization diagnostic: compare train, validation, and test condition-bag embedding distributions, then refactor the shared projection/aggregation path only after the evaluation mismatch is localized.
