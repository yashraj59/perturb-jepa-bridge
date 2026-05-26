# Encoder Projection Mode Audit

Wallclock minutes: `1.411`

## Main Finding

This audit localizes whether collapse is already present in raw encoder CLS embeddings, appears in the projection head, appears after L2 normalization, or is hidden by train-mode dropout.

## Key Rows

### attention

Mode `eval`:
- `rna_raw_cls` test min_std=0.030522, mean_std=0.0714817, bio_R2=-0.72508
- `rna_token_mean` test min_std=0.00960498, mean_std=0.0419357, bio_R2=0.104902
- `rna_projection_pre_norm` test min_std=0.0126947, mean_std=0.0344302, bio_R2=-0.338897
- `rna_projection_norm_instance_mean` test min_std=0.00342432, mean_std=0.00716876, bio_R2=0.0144976
- `rna_shared` test min_std=0.00714596, mean_std=0.024253, bio_R2=-0.294919
- `image_raw_cls` test min_std=0.00554016, mean_std=0.0116998, bio_R2=0.925624
- `image_patch_mean` test min_std=0.00478187, mean_std=0.0198444, bio_R2=0.92657
- `image_projection_pre_norm` test min_std=0.00396891, mean_std=0.0131367, bio_R2=0.900023
- `image_projection_norm_instance_mean` test min_std=0.00120752, mean_std=0.00348317, bio_R2=0.790294
- `image_shared` test min_std=0.00470986, mean_std=0.0127761, bio_R2=0.887541
- shared test recall@1=0.0625; token/patch mean test recall@1=0.0625

Mode `train_no_grad`:
- `rna_raw_cls` test min_std=0.0673646, mean_std=0.136282, bio_R2=-8.8461
- `rna_token_mean` test min_std=0.0109999, mean_std=0.041091, bio_R2=0.080856
- `rna_projection_pre_norm` test min_std=0.058022, mean_std=0.0759982, bio_R2=-1.3394
- `rna_projection_norm_instance_mean` test min_std=0.0129445, mean_std=0.01652, bio_R2=-0.229966
- `rna_shared` test min_std=0.0312298, mean_std=0.0593554, bio_R2=-1.00337
- `image_raw_cls` test min_std=0.0354573, mean_std=0.0855744, bio_R2=-6.33389
- `image_patch_mean` test min_std=0.014446, mean_std=0.0296178, bio_R2=0.56896
- `image_projection_pre_norm` test min_std=0.0491607, mean_std=0.097037, bio_R2=-1.4233
- `image_projection_norm_instance_mean` test min_std=0.012355, mean_std=0.0255348, bio_R2=-0.518184
- `image_shared` test min_std=0.0524967, mean_std=0.101447, bio_R2=-1.63152
- shared test recall@1=0.09375; token/patch mean test recall@1=0.0625

### mean

Mode `eval`:
- `rna_raw_cls` test min_std=0.016574, mean_std=0.0469605, bio_R2=-1.23278
- `rna_token_mean` test min_std=0.0113952, mean_std=0.0422719, bio_R2=0.0663232
- `rna_projection_pre_norm` test min_std=0.00886494, mean_std=0.0287046, bio_R2=-0.511841
- `rna_projection_norm_instance_mean` test min_std=0.00276373, mean_std=0.00857712, bio_R2=-0.0523646
- `rna_shared` test min_std=0.0123266, mean_std=0.0477492, bio_R2=-0.77936
- `image_raw_cls` test min_std=0.00443995, mean_std=0.0116713, bio_R2=0.92738
- `image_patch_mean` test min_std=0.00511188, mean_std=0.0206681, bio_R2=0.925108
- `image_projection_pre_norm` test min_std=0.00266916, mean_std=0.00922769, bio_R2=0.87349
- `image_projection_norm_instance_mean` test min_std=0.000695142, mean_std=0.00239547, bio_R2=0.699748
- `image_shared` test min_std=0.00512167, mean_std=0.0135657, bio_R2=0.891828
- shared test recall@1=0.0625; token/patch mean test recall@1=0.0625

Mode `train_no_grad`:
- `rna_raw_cls` test min_std=0.0605251, mean_std=0.122426, bio_R2=-15.9159
- `rna_token_mean` test min_std=0.0143666, mean_std=0.0416065, bio_R2=-0.147953
- `rna_projection_pre_norm` test min_std=0.0417785, mean_std=0.0667066, bio_R2=-0.546079
- `rna_projection_norm_instance_mean` test min_std=0.0119454, mean_std=0.0197586, bio_R2=-0.234678
- `rna_shared` test min_std=0.0868163, mean_std=0.134188, bio_R2=-4.35723
- `image_raw_cls` test min_std=0.0300445, mean_std=0.0813765, bio_R2=-2.23233
- `image_patch_mean` test min_std=0.0154258, mean_std=0.0293707, bio_R2=0.507021
- `image_projection_pre_norm` test min_std=0.0336294, mean_std=0.0560529, bio_R2=-1.25755
- `image_projection_norm_instance_mean` test min_std=0.00823154, mean_std=0.0145086, bio_R2=-0.571523
- `image_shared` test min_std=0.070692, mean_std=0.114585, bio_R2=-9.96053
- shared test recall@1=0.03125; token/patch mean test recall@1=0.0625

## Interpretation Rules

- Raw CLS collapsed but token/patch mean not collapsed: use token/patch pooled condition embeddings instead of CLS.
- Projection pre-norm healthy but normalized projection collapsed: change projection normalization/temperature geometry.
- Eval collapsed but train-no-grad healthy: dropout is masking eval collapse and train-batch traces are overoptimistic.
- Both eval and train-no-grad collapsed at raw CLS: encoder global token is the first failure point.

## Artifacts

- `ENCODER_PROJECTION_GEOMETRY.tsv`
- `ENCODER_PROJECTION_RETRIEVAL.tsv`
- `TRAINING_HISTORY_SUMMARY.tsv`
