# Codex Prompt: Build BioAction-JEPA — A Real Perturbation-Conditioned JEPA World Model for scRNA + Cell Imaging

## Copy/paste role instruction

You are Codex operating inside this repository. You have architectural freedom, but you must build a **real JEPA** model, not a renamed autoencoder, PLS clone, or lookup-table baseline.

Your task is to create **BioAction-JEPA**: a perturbation-conditioned, cross-modal joint-embedding predictive architecture for paired or condition-paired scRNA-seq and cell imaging data.

The central idea is:

> Treat a biological perturbation as an **action** in a latent biological world model. Given context observations from RNA and/or imaging, predict the latent teacher representation of missing target views, missing modalities, and perturbation-induced future biological states. Do this in representation space, not by reconstructing genes or pixels as the main objective.

This project currently contains useful components, but the current protected model of record is not a real learned JEPA. The current protected synthetic model is a rank-3 train-split-only PLS readout installed into raw-linear RNA/image heads. Keep it as a protected baseline and audit reference, but build a new encoder-first JEPA model whose main path is learned encoders + EMA target encoders + latent predictors.

Do not do open-ended hacking. Use the autoresearch protocol below: protected baseline, Step 0 baselines, tiered gates, literature log, diagnostics, and stop conditions.

---

# 0. What “real JEPA” means in this run

A model is **not** considered a real JEPA unless all of the following are true:

```text
1. The model has an online/context encoder that sees a partial or context view.
2. The model has a target encoder, preferably EMA teacher, that encodes target views with stop-gradient.
3. The model has an explicit predictor that receives context representations plus target queries/action tokens and predicts target representations.
4. The primary loss is in latent representation space.
5. Raw RNA reconstruction, count likelihood, image patch reconstruction, or pixel decoding are auxiliary only.
6. The final retrieval/counterfactual state uses encoder-derived representations, not PLS raw-linear heads.
7. The model can be trained with raw reconstruction weights set to zero and still has nonzero JEPA training signal.
8. The model logs anti-collapse diagnostics and has an explicit collapse-prevention objective.
9. The model supports missing-modality prediction: RNA -> image, image -> RNA, joint -> RNA/image.
10. The model supports action-conditioned transition prediction: control biological state + perturbation action -> perturbed teacher state.
```

Do not call a candidate “JEPA” if it only:

```text
- reconstructs masked genes or pixels;
- aligns two embeddings with contrastive loss only;
- uses PLS/readout heads as the main representation;
- trains a counterfactual decoder from frozen PLS embeddings;
- memorizes biological condition keys;
- predicts expression means directly without latent teacher targets;
- adds a small cosine loss to an otherwise non-JEPA autoencoder.
```

---

# 1. Scientific thesis

Build a model with this thesis:

> A biological cell state can be learned as a predictive latent world model. RNA and cell morphology are different observations of the same underlying state. Perturbations are actions that move that state. A real JEPA model should learn this state by predicting latent teacher targets across masked views, modalities, and perturbation transitions.

The intended novelty is:

```text
BioAction-JEPA =
  I-JEPA-style semantic target prediction
+ V-JEPA-style feature prediction without reconstruction as the main target
+ V-JEPA2-style action-conditioned world modeling
+ ImageBind/data2vec-style shared multimodal representation prediction
+ CellOT-style distributional bag/prototype matching
+ GEARS/scFoundation-style biological program/gene graph priors
+ autoresearch-bio evaluation discipline for perturbation biology.
```

This should become a paper-quality architecture, not merely a code patch.

Potential paper title:

```text
BioAction-JEPA: Perturbation-Conditioned Cross-Modal Joint-Embedding Predictive Learning for scRNA and Cell Imaging
```

---

# 2. Literature seed list to read and log before implementation

Create:

```text
outputs/autoresearch_bioaction_jepa_v1/papers_consulted.md
```

For each paper, record:

```text
Title
Authors
Venue/year
URL/DOI
Field
Technique extracted
How it maps to BioAction-JEPA
What code component it influences
Whether it preserves model identity
```

Do not implement entire papers. Extract concrete mechanisms.

## 2.1 JEPA / predictive representation learning

Read these first:

1. **I-JEPA — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**
   - URL: https://arxiv.org/abs/2301.08243
   - Extract: context block predicts large semantic target block representation; avoid pixel reconstruction as the main objective.
   - Map: RNA program blocks and morphology regions become target blocks.

2. **V-JEPA — Revisiting Feature Prediction for Learning Visual Representations from Video**
   - URL: https://arxiv.org/abs/2404.08471
   - Extract: stand-alone feature prediction objective; no pretrained encoder, no negatives, no text, no reconstruction as the main supervision.
   - Map: BioAction-JEPA must remain functional with reconstruction weights at zero.

3. **V-JEPA 2 — Self-Supervised Video Models Enable Understanding, Prediction, and Planning**
   - URL: https://arxiv.org/abs/2506.09985
   - Extract: video-trained representation plus small amount of action/robot trajectory data for action-conditioned world modeling.
   - Map: perturbations are biological actions; control-to-perturbed transitions are biological world-model steps.

4. **A Path Towards Autonomous Machine Intelligence**
   - URL: https://openreview.net/forum?id=BZ5a1r-kVsf
   - Extract: predictive world models learn abstract representations of observations and actions.
   - Map: biological action-conditioned latent state prediction.

## 2.2 Anti-collapse and teacher/student SSL

Read and extract only stabilization mechanisms:

1. **BYOL — Bootstrap Your Own Latent**
   - URL: https://arxiv.org/abs/2006.07733
   - Extract: online network predicts EMA target network representation without negatives.
   - Map: EMA target encoders, predictor asymmetry, stop-gradient.

2. **DINO — Emerging Properties in Self-Supervised Vision Transformers**
   - URL: https://arxiv.org/abs/2104.14294
   - Extract: self-distillation with no labels, momentum teacher, centering/sharpening, useful attention maps.
   - Map: optional prototype centering for cell-state target prototypes; attention diagnostics.

3. **VICReg — Variance-Invariance-Covariance Regularization**
   - URL: https://arxiv.org/abs/2105.04906
   - Extract: explicit variance floor and covariance decorrelation to prevent collapse.
   - Map: required anti-collapse loss on RNA, image, shared, transition-predicted latents.

4. **Barlow Twins — Self-Supervised Learning via Redundancy Reduction**
   - URL: https://arxiv.org/abs/2103.03230
   - Extract: cross-correlation identity objective to align paired views while reducing redundancy.
   - Map: cross-modal RNA/image embedding regularizer.

5. **data2vec — A General Framework for Self-supervised Learning in Speech, Vision and Language**
   - URL: https://arxiv.org/abs/2202.03555
   - Extract: predict contextualized latent teacher representations rather than local raw targets.
   - Map: teacher targets should be contextualized condition/bag/program latents, not raw gene/pixel tokens.

## 2.3 Multimodal learning from other fields

1. **ImageBind — One Embedding Space To Bind Them All**
   - URL: https://arxiv.org/abs/2305.05665
   - Extract: shared embedding across modalities; not every modality pair must be directly observed.
   - Map: support RNA-only, image-only, and paired condition batches; bind through common biological state.

2. **MultiMAE — Multi-modal Multi-task Masked Autoencoders**
   - URL: https://multimae.epfl.ch/
   - Extract: masked multi-modal pretraining across heterogeneous inputs.
   - Map: use modality dropout and cross-modal target queries, but keep JEPA latent prediction as the main objective rather than reconstruction.

3. **CrossMAE / cross-attention MAE variants**
   - URL: https://crossmae.github.io/
   - Extract: lightweight cross-attention decoders/predictors from context tokens to target queries.
   - Map: implement a query-based JEPA predictor that predicts requested target blocks/prototypes.

4. **MaskGIT — Masked Generative Image Transformer**
   - URL: https://arxiv.org/abs/2202.04200
   - Extract: iterative masked-token refinement, bidirectional conditioning.
   - Map: optional iterative latent refinement predictor for uncertain perturbation targets; do not make token generation the primary loss.

## 2.4 Single-cell, perturbation, and morphology literature

1. **scGPT — foundation model for single-cell biology**
   - URL: https://www.nature.com/articles/s41592-024-02201-0
   - Extract: transformer-style gene/cell modeling at scale.
   - Map: optional pretrained gene/cell embeddings or scGPT-style tokenization.

2. **scFoundation — large-scale foundation model on single-cell transcriptomics**
   - URL: https://www.nature.com/articles/s41592-024-02305-7
   - Extract: large-gene vocabulary, read-depth-aware pretraining, cell-state features.
   - Map: read-depth and count-aware auxiliary heads; gene-program tokenization.

3. **Geneformer / transfer learning for network biology**
   - URL: https://www.nature.com/articles/s41586-023-06139-9
   - Extract: gene-network-aware transformer pretraining.
   - Map: perturbation descriptors and gene graph priors for unseen perturbations.

4. **Deep-learning-based gene perturbation effect prediction benchmark**
   - URL: https://www.nature.com/articles/s41592-025-02772-6
   - Extract: simple baselines can beat or match foundation models in perturbation prediction.
   - Map: strict baselines and no overclaiming. Do not promote unless held-out perturbation/dose gates pass.

5. **CellOT — Learning single-cell perturbation responses using neural optimal transport**
   - URL: https://www.nature.com/articles/s41592-023-01969-x
   - Extract: perturbation responses are distributions, not just means; map control distributions to target distributions.
   - Map: distributional/prototype JEPA with OT/Hungarian/Sliced Wasserstein in latent space.

6. **GEARS — Predicting transcriptional outcomes of novel multigene perturbations**
   - URL: https://www.nature.com/articles/s41587-023-01905-6
   - Extract: gene-gene knowledge graph helps generalize to unseen perturbation combinations.
   - Map: optional gene graph / perturbation descriptor module.

7. **CPA — Compositional Perturbation Autoencoder**
   - URL: https://www.embopress.org/doi/full/10.15252/msb.202211517
   - Extract: factor perturbation, dose, cell type; predict unseen doses/cell types/combinations.
   - Map: action encoder factorization and dose-response monotonicity diagnostics.

8. **Self-supervision advances morphological profiling**
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11811211/
   - Extract: SSL on Cell Painting can improve morphology representations.
   - Map: image encoder/morphology targets and morphology-specific evaluation.

9. **Learning representations for image-based profiling of perturbations**
   - URL: https://www.nature.com/articles/s41467-024-45999-1
   - Extract: Cell Painting perturbation data is useful for representation learning.
   - Map: morphology phenotype retrieval, batch control, and perturbation phenotype metrics.

---

# 3. Current repository facts and protected baseline

Before coding, read these files completely if present:

```text
CURRENT_STATUS_AND_BEST_MODEL_CODE (1).md
FULL_ARCHITECTURE_CODE_BUNDLE.md
SKILL (2).md

perturb_jepa/config.py
perturb_jepa/models/bridge.py
perturb_jepa/models/rna_encoder.py
perturb_jepa/models/image_encoder.py
perturb_jepa/models/perturbation_encoder.py
perturb_jepa/models/bag_aggregator.py
perturb_jepa/losses.py
perturb_jepa/training/objectives.py
perturb_jepa/training/trainer.py
perturb_jepa/training/synthetic_biology_lite.py

scripts/run_synthetic_lite_step0.py
scripts/evaluate_prefit_pls_readout.py
scripts/train_pls_distilled_head.py
scripts/train_clone_counterfactual_decoder.py
scripts/run_family_m_transport_baselines.py
scripts/run_family_n_distillation.py
scripts/run_family_o_count_likelihood.py

tests/
outputs/autoresearch_synth_lite/
pyproject.toml
requirements.txt
```

## 3.1 Protected model of record

Current protected synthetic shared-geometry model of record:

```text
Model: closed-form rank-3 train-split-only PLS readout
RNA readout: raw_linear_pseudobulk
Image readout: raw_linear_pooled
Branch: autoresearch/synthetic-lite-v1
Data: synthetic only
Status: protected baseline / initializer / audit reference, not learned JEPA
```

Do not delete, overwrite, or silently replace this. It remains the protected model of record until a Tier 3 BioAction-JEPA candidate passes every gate.

## 3.2 Baseline numbers to record and verify

Record these numbers in `BASELINE_REGISTRY.md`, then verify from raw artifacts where possible.

Protected PLS shared geometry on `synth_micro`, seeds `0/1/2`:

```text
RNA->image recall@1: 0.2396 +/- 0.0295
RNA->image recall@5: 0.6667 +/- 0.1284
RNA latent R2:       0.5929 +/- 0.1395
Image latent R2:     0.9134 +/- 0.0206
Batch balanced acc:  0.4792 +/- 0.0780
```

Family N expression-space condition-mean reference on seed 2:

```text
Exact train biological-key coverage: 1.0000
Program recovery:        0.7520
Direction accuracy:      0.6899
logFC correlation:       0.7502
Pseudobulk correlation:  0.8725
Top50 DE overlap:        0.5683
Mean delta/target ratio: 0.7400
```

Family O count-aware condition-mean reference on seed 2:

```text
Exact train biological-key coverage: 1.0000
Program recovery:        0.7433
Direction accuracy:      0.7679
logFC correlation:       0.7562
Pseudobulk correlation:  0.8815
Top50 DE overlap:        0.6392
Poisson NLL:             48.4387
NB NLL:                  4.9933
```

Important: the exact-condition split is matching-baseline dominated. Train and test share biological keys in the documented seed, so train-only condition means are too strong. Do not evaluate novelty only on exact-key `test`. The serious benchmarks are held-out perturbations, held-out doses, held-out cell lines, and held-out perturbation × cell-line combinations.

---

# 4. Core architecture: BioAction-JEPA

Implement a new architecture rather than merely patching the old bridge. You may reuse existing encoders, aggregators, losses, and teacher code, but the new model should have a clean identity.

Create these files unless a better local organization is obvious:

```text
perturb_jepa/models/bioaction_jepa.py
perturb_jepa/training/bioaction_batches.py
perturb_jepa/training/bioaction_losses.py
perturb_jepa/training/bioaction_trainer.py
perturb_jepa/evaluation/bioaction_metrics.py
scripts/train_bioaction_jepa.py
scripts/evaluate_bioaction_jepa.py
tests/test_bioaction_jepa_model.py
tests/test_bioaction_jepa_losses.py
tests/test_bioaction_condition_pairs.py
tests/test_bioaction_eval_split.py
```

## 4.1 Model components

The model should expose something close to this structure:

```python
@dataclass(frozen=True)
class BioActionJEPAConfig:
    rna: RNAEncoderConfig
    image: ImageEncoderConfig
    perturbation: PerturbationEncoderConfig
    shared_dim: int = 128
    predictor_dim: int = 256
    num_state_prototypes: int = 8
    num_rna_program_targets: int = 16
    num_image_region_targets: int = 8
    num_condition_prototypes: int = 8
    target_query_dim: int = 128
    predictor_depth: int = 4
    predictor_heads: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.996
    use_vicreg: bool = True
    use_barlow: bool = True
    use_distributional_jepa: bool = True
    use_transition_jepa: bool = True
    use_cross_modal_jepa: bool = True
    use_intra_modal_jepa: bool = True
    use_inverse_action_head: bool = True
    use_gene_program_targets: bool = True
    use_graph_action_encoder: bool = False
    count_decoder_aux: bool = True
    image_decoder_aux: bool = False
    reconstruction_is_auxiliary: bool = True
    forbid_condition_key_features: bool = True
```

Core modules:

```python
class BioActionJEPA(nn.Module):
    rna_context_encoder: RNAEncoder
    image_context_encoder: ImageEncoder
    rna_target_encoder: EMA copy of RNAEncoder
    image_target_encoder: EMA copy of ImageEncoder

    rna_context_projector: projector to shared_dim
    image_context_projector: projector to shared_dim
    rna_target_projector: EMA/projector target path
    image_target_projector: EMA/projector target path

    rna_condition_aggregator: set/bag aggregator
    image_condition_aggregator: set/bag aggregator
    joint_condition_fuser: modality-aware set transformer or gated product-of-experts

    action_encoder: perturbation/cell/dose/time/descriptor encoder
    target_query_encoder: encodes requested target modality, level, block/prototype id, and horizon

    intra_modal_predictor: query-based predictor for RNA->RNA and image->image targets
    cross_modal_predictor: query-based predictor for RNA->image, image->RNA, joint->RNA/image targets
    transition_predictor: action-conditioned latent dynamics predictor
    inverse_action_head: optional target/source latent -> perturbation action classifier/regressor

    optional_count_decoder: NB/Poisson auxiliary decoder
    optional_morphology_decoder: auxiliary prototype decoder only
```

## 4.2 The predictor must be query-based, not just an MLP

Implement a small Transformer-style predictor that cross-attends target queries to context tokens:

```python
class TargetQueryPredictor(nn.Module):
    def forward(
        self,
        context_tokens: Tensor,     # [B, C, D]
        target_queries: Tensor,    # [B, T, D]
        action_tokens: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:                   # [B, T, D]
```

Minimum acceptable implementation:

```text
1. Project context tokens and target queries to predictor_dim.
2. Append or FiLM-condition with action token when provided.
3. Use TransformerDecoder layers or cross-attention blocks: queries attend to context.
4. Output normalized target predictions in shared_dim.
```

This is important. Real JEPA predicts specified target representations from a context representation. A single MLP over an already-pooled vector is too weak and too close to the old setup.

## 4.3 Multi-level targets

Predict targets at four levels:

### Level A: RNA program-block JEPA

Context:

```text
visible genes/program tokens from RNA cell or RNA bag
```

Target:

```text
EMA teacher latent for masked gene-program block(s)
```

Do not randomly mask only individual genes. Prefer semantic target groups:

```text
program blocks from synthetic gene_program_assignment
pathway blocks if real pathway annotations exist
high-variance gene blocks
DE-enriched blocks in training data only
random fallback blocks only if no annotations exist
```

### Level B: image morphology-region JEPA

Context:

```text
visible image patches/cell crops/regions
```

Target:

```text
EMA teacher latent for masked large morphology regions, channel subsets, or cell-neighborhood regions
```

Prefer large semantic masks, not tiny independent patches.

### Level C: cross-modal JEPA

Required tasks:

```text
RNA context -> image teacher target
image context -> RNA teacher target
joint RNA+image context -> held-out RNA target
joint RNA+image context -> held-out image target
RNA-only batch -> bind into shared biological state
image-only batch -> bind into shared biological state
```

Use target queries so the same predictor can ask for:

```text
target modality = RNA or image
target level = cell, program, bag, condition prototype
target id = block/prototype index
time horizon = same-condition or future/perturbed
```

### Level D: action-conditioned perturbation JEPA

This is the centerpiece.

Given:

```text
control RNA and/or image state
perturbation action token = perturbation id/descriptor + cell line + dose + time
```

Predict:

```text
EMA teacher latent of the perturbed condition state
```

This should work at condition-bag level even if RNA and image are not matched at the single-cell level.

Minimum transition objectives:

```text
control RNA -> perturbed RNA teacher state
control image -> perturbed image teacher state
control RNA -> perturbed image teacher state
control image -> perturbed RNA teacher state
joint control RNA+image -> perturbed joint teacher state
```

Optional but recommended:

```text
inverse transition: predicted perturbed state + inverse action -> control teacher state
composition: action A then action B approximately matches combined perturbation if combinations exist
monotonic dose consistency: larger dose should usually produce larger latent displacement for non-control perturbations, measured diagnostically not as a hard biological law
```

---

# 5. Loss design

Create `BioActionJEPALossWeights` with explicit weights.

Primary losses:

```text
rna_program_jepa
image_region_jepa
rna_to_image_jepa
image_to_rna_jepa
joint_to_rna_jepa
joint_to_image_jepa
transition_rna_jepa
transition_image_jepa
transition_joint_jepa
distributional_prototype_jepa
```

Stabilization losses:

```text
vicreg_variance
vicreg_covariance
barlow_cross_correlation
teacher_student_centering_diagnostics
latent_norm_regularization
```

Biology/downstream auxiliary losses:

```text
count_nb_nll_aux
poisson_nll_aux
program_delta_aux
direction_aux
morphology_prototype_aux
inverse_action_aux
cycle_latent_aux
```

Important weighting rule:

```text
The sum of latent JEPA losses must dominate the objective after warmup.
Raw reconstruction and count likelihood are auxiliary.
A candidate fails real-JEPA identity if reconstruction/count losses are necessary for nontrivial performance.
```

Suggested initial weights:

```text
rna_program_jepa:           1.0
image_region_jepa:          1.0
rna_to_image_jepa:          2.0
image_to_rna_jepa:          2.0
joint_to_rna_jepa:          1.0
joint_to_image_jepa:        1.0
transition_rna_jepa:        2.0
transition_image_jepa:      2.0
transition_joint_jepa:      2.0
distributional_jepa:        1.0
vicreg_variance:            0.1
vicreg_covariance:          0.05
barlow_cross_correlation:   0.05
count_nb_nll_aux:           0.1
program_delta_aux:          0.2
cycle_latent_aux:           0.1
inverse_action_aux:         0.05
raw_rna_reconstruction:     0.0 initially, max 0.05
raw_image_reconstruction:   0.0 initially, max 0.05
```

Use these latent losses:

```python
cosine_jepa_loss(pred, target)
smooth_l1_jepa_loss(normalize(pred), normalize(target))
masked_cosine_jepa_loss(pred_tokens, target_tokens, target_mask)
prototype_ot_jepa_loss(pred_prototypes, target_prototypes)
vicreg_loss(latents)
barlow_cross_correlation_loss(rna_latents, image_latents)
```

For distributional prototype JEPA, implement one of:

```text
Hungarian matching over predicted vs teacher prototypes
Sinkhorn/entropic OT if scipy/torch implementation is simple and tested
Sliced Wasserstein over latent prototypes
MMD over latent prototypes
```

Do not over-engineer the first implementation. Start with Hungarian or Sliced Wasserstein if available.

---

# 6. Data layer and batching

Create a condition-pair batch object.

```python
@dataclass
class BioActionConditionBatch:
    # source/control observation
    control_gene_ids: Tensor | None
    control_expression_values: Tensor | None
    control_counts: Tensor | None
    control_images: Tensor | None
    control_rna_bag_mask: Tensor | None
    control_image_bag_mask: Tensor | None

    # same-condition or target/perturbed observation
    target_gene_ids: Tensor | None
    target_expression_values: Tensor | None
    target_counts: Tensor | None
    target_images: Tensor | None
    target_rna_bag_mask: Tensor | None
    target_image_bag_mask: Tensor | None

    # metadata/action
    perturbation_id: Tensor
    perturbation_type_id: Tensor
    cell_line_id: Tensor
    batch_id: Tensor
    dose: Tensor
    time: Tensor
    descriptor: Tensor | None

    # labels for evaluation only; forbidden as model features when they leak exact identity
    condition_key: list[str] | None
    biological_key: list[tuple] | None
    split: list[str] | None
```

## 6.1 Synthetic data support

Use `SyntheticBiologyLiteDataset` first.

Required patches:

```text
1. Add --eval-split to scripts that currently hardcode test.
2. Ensure pair_records and counterfactual pair generation can use:
   - test
   - test_heldout_perturbation
   - test_heldout_dose
   - test_heldout_batch
3. Add a condition-pair dataloader that can sample:
   - same-condition RNA/image pairs for cross-modal JEPA
   - control -> perturbed pairs for transition JEPA
   - modality-dropout batches
   - source-only or target-only batches for robustness
```

Hard rule:

```text
Do not feed biological_key, condition_key, or exact target-key one-hot vectors to BioAction-JEPA.
```

## 6.2 Real data support

Implement enough real-data interface to be useful, even if tests use synthetic data.

Expected real inputs:

```text
AnnData path for RNA
image manifest TSV/CSV
image root directory
condition_key column
perturbation column
cell_line column
dose column
time column
batch/plate/well columns
split column
optional control label
optional perturbation descriptor path
```

Condition-level pairing is acceptable:

```text
RNA bag for condition C pairs with image bag for condition C.
The same physical cell is not required.
```

Add a README section documenting expected columns and examples.

---

# 7. Model forward contract

The model should expose two forward modes:

## 7.1 Representation mode

For retrieval/evaluation:

```python
outputs = model.encode_condition(
    gene_ids=...,
    expression_values=...,
    images=...,
    metadata=...,
    mode="context" or "target",
)
```

Returns:

```text
rna_cell_tokens
rna_program_tokens
rna_condition_prototypes
rna_condition_state
image_patch_tokens
image_region_tokens
image_condition_prototypes
image_condition_state
joint_condition_state
shared_state
batch_leakage_logits optional
perturbation_logits optional
```

## 7.2 JEPA prediction mode

For training:

```python
outputs = model.forward_jepa(batch, mask_spec, target_spec)
```

Returns:

```text
predicted target latents for each JEPA task
teacher target latents for each JEPA task, detached
masks and target ids
action embeddings
transition latents
optional auxiliary decoder outputs
diagnostics
```

Required output keys:

```text
rna_program_jepa_pred
rna_program_jepa_target
image_region_jepa_pred
image_region_jepa_target
rna_to_image_jepa_pred
rna_to_image_jepa_target
image_to_rna_jepa_pred
image_to_rna_jepa_target
joint_to_rna_jepa_pred
joint_to_rna_jepa_target
joint_to_image_jepa_pred
joint_to_image_jepa_target
transition_rna_jepa_pred
transition_rna_jepa_target
transition_image_jepa_pred
transition_image_jepa_target
transition_joint_jepa_pred
transition_joint_jepa_target
shared_state
joint_condition_state
rna_condition_state
image_condition_state
```

Teacher targets must be `detach()`ed and produced under `torch.no_grad()`.

---

# 8. Anti-collapse and diagnostics

Every training step must log:

```text
loss/total
loss/rna_program_jepa
loss/image_region_jepa
loss/rna_to_image_jepa
loss/image_to_rna_jepa
loss/transition_rna_jepa
loss/transition_image_jepa
loss/transition_joint_jepa
loss/vicreg_variance
loss/vicreg_covariance
loss/barlow

jepa_weighted_to_aux_ratio
raw_reconstruction_weighted_to_jepa_ratio
count_aux_weighted_to_jepa_ratio

latent/rna_std_mean
latent/image_std_mean
latent/joint_std_mean
latent/transition_pred_std_mean
latent/rna_effective_rank
latent/image_effective_rank
latent/joint_effective_rank
latent/collapse_fraction_dims_below_0.05
latent/cosine_mean_pairwise

teacher/ema_decay
teacher/target_norm_mean
teacher/pred_norm_mean
teacher/pred_target_cosine

batch_probe/balanced_accuracy
perturbation_probe/state_accuracy
condition_key_exact_feature_present  # must be 0 for BioAction-JEPA
pls_raw_linear_used_as_main_path      # must be 0 for BioAction-JEPA
```

Hard-fail collapse diagnostics:

```text
1. effective rank < 0.25 * shared_dim on Tier 2 mean
2. more than 50% dimensions with std < 0.05 after warmup
3. pairwise cosine mean > 0.90 for unrelated conditions
4. transition predictions become global mean latents
5. batch balanced accuracy is high while biological metrics do not improve
6. condition-key leakage detected
7. PLS raw-linear readout used as final retrieval state
```

---

# 9. Evaluation stack

Create `scripts/evaluate_bioaction_jepa.py` that reports all metrics below.

## 9.1 Real-JEPA identity metrics

```text
encoder_path_used: 1/0
pls_raw_linear_main_path_used: 0 required
condition_key_feature_present: 0 required
latent_prediction_loss_available_with_reconstruction_zero: 1 required
teacher_stop_gradient_verified: 1 required
ema_teacher_updated: 1 required
```

## 9.2 Cross-modal representation metrics

```text
RNA -> image recall@1/5/10
image -> RNA recall@1/5/10
mean reciprocal rank
same-condition cosine vs mismatched cosine
multi-positive retrieval by perturbation/cell/dose hierarchy
latent R2 to synthetic z_bio
latent R2 to synthetic z_tech, should be low/moderate not dominant
batch balanced accuracy probe
```

## 9.3 Perturbation-transition metrics

For control -> perturbed prediction:

```text
transition latent cosine to teacher target
transition latent R2
transition delta cosine
source-to-target improvement over source-as-target
held-out perturbation metrics
held-out dose metrics
held-out cell-line/combo metrics if available
```

## 9.4 RNA biological metrics

```text
program recovery
program delta correlation
signed DE agreement
direction accuracy
logFC correlation
pseudobulk correlation
top50 DE overlap
weighted delta R2
mean delta/target ratio
NB NLL if count decoder is enabled
Poisson NLL if count decoder is enabled
```

For direction-aware metrics, wrong-direction overlap is a biological failure.

## 9.5 Morphology metrics

```text
image-condition retrieval
morphology prototype cosine
morphology latent Sliced Wasserstein or MMD
perturbation phenotype nearest-neighbor accuracy
cell-painting-style profile correlation if features exist
batch/plate leakage probe
```

## 9.6 Distributional and population metrics

```text
MMD between predicted and target condition prototypes
Sliced Wasserstein between predicted and target prototypes
cluster coverage
entropy/diversity of predicted prototypes
technical duplicate ceiling if possible
source-as-target null
global-mean null
train-only condition mean null
variance-shrunk null
```

---

# 10. Step 0 baselines before architecture search

Create:

```text
outputs/autoresearch_bioaction_jepa_v1/step0_baselines/
outputs/autoresearch_bioaction_jepa_v1/BASELINE_REGISTRY.md
```

Run or reproduce:

```bash
python scripts/run_family_m_transport_baselines.py --dataset synth_micro --seed 2 --rank 3 --device cpu
python scripts/run_family_n_distillation.py --dataset synth_micro --seed 2 --rank 3 --device cpu
python scripts/run_family_o_count_likelihood.py --dataset synth_micro --seed 2 --rank 3 --device cpu
python scripts/evaluate_prefit_pls_readout.py --dataset synth_micro --seed 2 --rank 3 --device cpu
```

Patch scripts to support held-out split evaluation, then run baselines on:

```bash
python scripts/run_family_m_transport_baselines.py \
  --dataset synth_heldout_perturbation_lite \
  --eval-split test_heldout_perturbation \
  --seed 2 --rank 3 --device cpu

python scripts/run_family_m_transport_baselines.py \
  --dataset synth_dose_extrapolation_lite \
  --eval-split test_heldout_dose \
  --seed 2 --rank 3 --device cpu
```

If these commands require patching, patch them. If a script cannot support a split yet, document the limitation and add the split argument.

---

# 11. Required tests

Before training, implement tests.

## 11.1 Model tests

```text
- BioActionJEPA builds from config.
- forward_jepa works with RNA-only input.
- forward_jepa works with image-only input.
- forward_jepa works with paired RNA+image input.
- forward_jepa works with control->target transition batch.
- all required output keys exist.
- target tensors do not require grad.
- teacher parameters do not require grad.
- teacher parameters update under EMA.
- reconstruction weights can be zero while JEPA losses are nonzero.
- PLS raw-linear projection is not used in BioAction-JEPA representation mode.
```

## 11.2 Loss tests

```text
- cosine_jepa_loss is near zero for identical normalized inputs.
- masked target loss ignores unselected targets.
- VICReg variance loss activates on collapsed latents.
- Barlow/cross-correlation loss runs on paired RNA/image latents.
- distributional prototype loss handles [B,K,D] inputs.
- total loss has finite gradients.
```

## 11.3 Data tests

```text
- condition-pair loader creates control->perturbed pairs.
- no test target rows are used to build train-only teachers/baselines.
- heldout_perturbation split contains heldout perturbations only.
- heldout_dose split contains heldout doses only.
- biological_key/condition_key are not returned as numeric model features.
```

## 11.4 Eval tests

```text
- evaluate_bioaction_jepa.py runs on a tiny synthetic checkpoint.
- metrics include real-JEPA identity checks.
- metrics include exact_match_fraction for lookup baselines.
- metrics JSON and markdown reports are written.
```

Run at minimum:

```bash
pytest tests/test_bioaction_jepa_model.py \
       tests/test_bioaction_jepa_losses.py \
       tests/test_bioaction_condition_pairs.py \
       tests/test_bioaction_eval_split.py
```

---

# 12. Training script contract

Create:

```text
scripts/train_bioaction_jepa.py
```

Required CLI:

```bash
python scripts/train_bioaction_jepa.py \
  --dataset synth_micro \
  --seed 0 \
  --device cpu \
  --steps 100 \
  --batch-size 4 \
  --shared-dim 64 \
  --predictor-dim 128 \
  --num-state-prototypes 4 \
  --rna-mask-mode program \
  --image-mask-mode block \
  --modality-dropout 0.25 \
  --transition-jepa-weight 2.0 \
  --cross-modal-jepa-weight 2.0 \
  --reconstruction-weight 0.0 \
  --count-aux-weight 0.1 \
  --output-dir outputs/autoresearch_bioaction_jepa_v1/experiments/EXP_NAME
```

The script must write:

```text
config.json
metrics_train.jsonl
metrics_eval.json
checkpoint.pt
model_card.md
jepa_identity_report.md
collapse_diagnostics.json
```

The training script must support:

```text
--dataset synth_micro
--dataset synth_heldout_perturbation_lite
--dataset synth_dose_extrapolation_lite
--eval-split test
--eval-split test_heldout_perturbation
--eval-split test_heldout_dose
--no-reconstruction
--rna-only
--image-only
--paired
--transition-only
--disable-count-aux
--disable-pls-bootstrap
--pls-bootstrap-weight FLOAT
--anneal-pls-bootstrap-steps INT
```

## 12.1 Optional PLS bootstrap rule

PLS may be used only as a bootstrap teacher or audit comparator.

Allowed:

```text
small initial distillation loss from encoder state to train-only PLS shared state
annealed to zero
logged as bootstrap/pls_weight
never used as final retrieval embedding
never used as target for transition JEPA after warmup
```

Forbidden:

```text
using raw_linear_pseudobulk or raw_linear_pooled as main BioAction-JEPA state
using PLS output as the only teacher target
calling PLS clone a JEPA model
```

---

# 13. Evaluation script contract

Create:

```text
scripts/evaluate_bioaction_jepa.py
```

Required CLI:

```bash
python scripts/evaluate_bioaction_jepa.py \
  --checkpoint outputs/autoresearch_bioaction_jepa_v1/experiments/EXP_NAME/checkpoint.pt \
  --dataset synth_heldout_perturbation_lite \
  --eval-split test_heldout_perturbation \
  --seed 0 \
  --device cpu \
  --output-dir outputs/autoresearch_bioaction_jepa_v1/experiments/EXP_NAME/eval_heldout_perturbation
```

Write:

```text
metrics.json
metrics.md
retrieval_metrics.json
transition_metrics.json
biology_metrics.json
morphology_metrics.json
collapse_diagnostics.json
jepa_identity_report.md
prediction_arrays.npz if not too large
```

`jepa_identity_report.md` must explicitly answer:

```text
Is this a real JEPA? yes/no
Are encoder readouts the main path? yes/no
Are PLS raw-linear heads used as main path? yes/no
Are teacher targets stop-grad? yes/no
Can loss run with reconstruction=0? yes/no
Are condition-key one-hots used? yes/no
Which JEPA tasks are active?
What fraction of weighted objective is JEPA vs auxiliary?
```

---

# 14. Autoresearch loop

Create:

```text
outputs/autoresearch_bioaction_jepa_v1/results.tsv
outputs/autoresearch_bioaction_jepa_v1/research_journal.md
outputs/autoresearch_bioaction_jepa_v1/architectural_changes_log.md
outputs/autoresearch_bioaction_jepa_v1/family_allocation.md
outputs/autoresearch_bioaction_jepa_v1/BASELINE_REGISTRY.md
outputs/autoresearch_bioaction_jepa_v1/papers_consulted.md
outputs/autoresearch_bioaction_jepa_v1/external_resources.md
outputs/autoresearch_bioaction_jepa_v1/identity_violations_considered.md
outputs/autoresearch_bioaction_jepa_v1/insights/
```

## 14.1 results.tsv columns

Use at least:

```text
commit	experiment_num	family	tier_reached	status	dataset	eval_split	seed_list	primary_metric	secondary_metric	protected_metric_summary	jepa_identity_pass	collapse_pass	architectural_change	description
```

## 14.2 Journal entry template

```markdown
## Experiment <N>: <Title>

**Hypothesis**: <failure mode + why mechanism should help>

**Literature basis**: <papers and extracted mechanism>

**Family**: <family name>

**Implementation**: <files changed, modules added, parameter count>

**Real-JEPA identity**: <which JEPA tasks active, reconstruction weight, PLS usage>

**Initialization / identity preservation**: <teacher init, bootstrap, smoke tests>

**Tier result**: <metrics>

**Diagnostics**: <collapse, batch leakage, contribution ratios, aux/main ratios>

**Biology result**: <program/direction/distribution/morphology metrics>

**Decision**: <exact label>

**Learning**: <what this teaches>

**Artifact retention**: <retain/delete/checkpoint path>
```

## 14.3 Decision labels

Use exact labels:

```text
BASELINE_COMPLETE
IMPLEMENTATION_SMOKE_PASS
IMPLEMENTATION_SMOKE_FAIL
TIER1_KEEP_REAL_JEPA_SIGNAL
TIER1_DISCARD_NOT_REAL_JEPA
TIER1_DISCARD_NO_SIGNAL
TIER1_DISCARD_COLLAPSE
TIER1_DISCARD_BATCH_LEAKAGE
TIER1_DISCARD_BIOLOGY_REGRESSION
TIER1_DISCARD_IMPLEMENTATION_MISMATCH
TIER2_PASS_CLEAN
TIER2_PASS_HIGH_RISK_DO_NOT_PROMOTE
TIER2_FAIL_HIGH_VARIANCE
TIER2_FAIL_COLLAPSE
TIER2_FAIL_PROTECTED_REGRESSION
TIER2_FAIL_BIOLOGY_REGRESSION
TIER3_PASS_NEW_MODEL_OF_RECORD
TIER3_FAIL_USEFUL_FAILURE
FAMILY_COOLDOWN
FAMILY_RETIRED
SEARCH_CLOSED_NO_NEW_BASELINE
```

---

# 15. Architectural families

You have freedom, but use these pre-registered families. Start with the minimal real JEPA, then deepen only productive families.

## Family A: Minimal Real BioAction-JEPA

Motivation:

```text
The current repo has encoders and JEPA-like token losses, but the protected result is PLS raw-linear readout. Need a clean encoder-first JEPA model.
```

Hypothesis:

```text
A query-based predictor with EMA teacher targets can learn shared biological state better than raw-linear PLS on held-out splits.
```

Experiments:

```text
A1: RNA/image condition-state cross-modal JEPA only.
A2: Add intra-modal RNA program and image region JEPA.
A3: Add transition JEPA control->perturbed.
A4: Add distributional prototype JEPA.
```

Stop/pivot:

```text
Retire Family A if it cannot pass real-JEPA identity tests or collapses across two seeds.
```

## Family B: ImageBind/data2vec-style missing-modality binding

Motivation:

```text
Real datasets may not have single-cell-level RNA/image pairing; condition-level pairing and missing modalities must work.
```

Hypothesis:

```text
Modality dropout plus contextual teacher targets will bind RNA and morphology in a shared biological state without requiring all pair types.
```

Experiments:

```text
B1: modality dropout in paired batches.
B2: RNA-only and image-only batches with shared target prototypes.
B3: weak positives by perturbation/cell/dose hierarchy.
B4: target query conditioning by modality/level/horizon.
```

Stop/pivot:

```text
Cooldown if cross-modal retrieval improves but transition biology regresses or batch leakage rises.
```

## Family C: V-JEPA2-style biological action world model

Motivation:

```text
Perturbation prediction is the scientific goal. A perturbation should be treated as an action that moves latent cell state.
```

Hypothesis:

```text
An action-conditioned latent transition predictor will generalize to held-out perturbations/doses better than direct target mean decoders.
```

Experiments:

```text
C1: transition predictor from control joint state + action -> perturbed joint teacher state.
C2: cross-modal transitions: control RNA -> perturbed image, control image -> perturbed RNA.
C3: inverse action head and cycle latent consistency.
C4: dose/time continuous FiLM conditioning.
C5: iterative latent refinement predictor for uncertain targets.
```

Stop/pivot:

```text
Retire or redesign if transition predictions become global means or source-as-target is not beaten on held-out splits.
```

## Family D: Biological program and graph priors

Motivation:

```text
Random gene masking is noisy; biological perturbations act through programs, pathways, and gene networks.
```

Hypothesis:

```text
Program-block JEPA and graph-informed action descriptors improve held-out perturbation generalization.
```

Experiments:

```text
D1: synthetic gene_program_assignment program targets.
D2: train-only coexpression programs if no external pathways exist.
D3: optional Reactome/GO/MSigDB pathway target blocks if easy to add and provenance is logged.
D4: optional GEARS-style gene graph action encoder.
D5: optional pretrained gene embeddings from Geneformer/scGPT/scFoundation if locally feasible; otherwise document as deferred.
```

Constraints:

```text
No named-gene hacks. No test-derived pathways. Log all external resources, versions, and coverage.
```

Stop/pivot:

```text
Cooldown if program recovery improves but direction accuracy or held-out transition metrics regress.
```

## Family E: Distributional cell-state JEPA

Motivation:

```text
A condition is a distribution of cells/images, not just a mean vector.
```

Hypothesis:

```text
Predicting sets of teacher prototypes with OT/Sliced-Wasserstein loss captures population structure better than a single pooled state.
```

Experiments:

```text
E1: K teacher condition prototypes per modality.
E2: Hungarian/Sliced-Wasserstein prototype matching.
E3: diversity/coverage regularization.
E4: compare mean-state vs prototype-distribution transition JEPA.
```

Stop/pivot:

```text
Retire if prototype diversity collapses or compute cost dominates without metric gains.
```

---

# 16. Tiered evaluation gates

## Tier 0: implementation smoke

Run tests and a tiny training job:

```bash
pytest tests/test_bioaction_jepa_model.py tests/test_bioaction_jepa_losses.py tests/test_bioaction_condition_pairs.py
python scripts/train_bioaction_jepa.py --dataset synth_micro --seed 0 --device cpu --steps 5 --batch-size 2 --shared-dim 16 --predictor-dim 32 --output-dir outputs/autoresearch_bioaction_jepa_v1/smoke
python scripts/evaluate_bioaction_jepa.py --checkpoint outputs/autoresearch_bioaction_jepa_v1/smoke/checkpoint.pt --dataset synth_micro --eval-split test --seed 0 --device cpu --output-dir outputs/autoresearch_bioaction_jepa_v1/smoke/eval
```

Pass requires:

```text
tests pass
loss finite
JEPA losses nonzero with reconstruction=0
teacher targets stop-grad
no PLS main path
no condition-key features
no collapse in first few steps
```

## Tier 1: single-seed signal

Datasets:

```text
synth_micro test
synth_heldout_perturbation_lite test_heldout_perturbation
synth_dose_extrapolation_lite test_heldout_dose
```

Seeds:

```text
seed 0 initially
```

Pass requires:

```text
1. real-JEPA identity pass
2. no hard collapse
3. at least one cross-modal retrieval metric improves over an encoder-only random/initial baseline
4. transition prediction beats source-as-target on at least one held-out split
5. no severe batch leakage
6. required reports written
```

Tier 1 does not promote. It only permits Tier 2.

## Tier 2: multi-seed validation

Seeds:

```text
0, 1, 2
```

Datasets:

```text
synth_micro test
synth_heldout_perturbation_lite test_heldout_perturbation
synth_dose_extrapolation_lite test_heldout_dose
```

Pass requires:

```text
1. Tier 1 signal holds in mean metrics.
2. Standard deviation is not larger than the claimed gain.
3. No seed-specific collapse.
4. Encoder-path RNA->image retrieval is competitive with or improves toward protected PLS on synth_micro.
5. Held-out transition metrics beat source-as-target and global-mean nulls.
6. Exact condition-key leakage remains zero.
7. Program/direction metrics do not regress catastrophically.
```

Tier 2 still does not promote if held-out generalization is weak.

## Tier 3: no-regression / generalization validation

Required datasets/splits:

```text
synth_heldout_perturbation_lite / test_heldout_perturbation
synth_dose_extrapolation_lite / test_heldout_dose
synth_batch_confound_lite / test or heldout batch if configured
optional real dataset if paths are available
```

Promotion requires all:

```text
1. real-JEPA identity pass
2. no PLS/raw-linear main path
3. no biological-key/condition-key leakage
4. held-out perturbation transition metrics beat source-as-target, global mean, and train-only fallback baselines
5. held-out dose transition metrics beat source-as-target, global mean, and train-only fallback baselines
6. cross-modal retrieval is meaningfully better than random/encoder-only and competitive with protected PLS where comparable
7. batch leakage probe not worse than protected baseline by more than pre-registered slack
8. no collapse hard fail
9. biology stack passes: direction, program, logFC/pseudobulk/top-DE metrics collectively improve or no-regress
10. full documentation exists
```

Only Tier 3 can supersede the protected model of record.

---

# 17. Practical first implementation path

Do this in order.

## Step 1: Make the benchmark usable

Patch all hardcoded eval splits:

```text
scripts/run_synthetic_lite_step0.py
scripts/run_family_m_transport_baselines.py
scripts/run_family_n_distillation.py
scripts/run_family_o_count_likelihood.py
scripts/train_pls_distilled_head.py
scripts/train_clone_counterfactual_decoder.py
scripts/evaluate_prefit_pls_readout.py
```

Add:

```python
parser.add_argument("--eval-split", default="test")
```

Then replace hardcoded:

```python
split="test"
```

with:

```python
split=args.eval_split
```

and similarly for `pair_records(dataset, split=...)`.

Add tests proving held-out split arguments work.

## Step 2: Implement minimal BioAction-JEPA model

Implement:

```text
BioActionJEPAConfig
BioActionJEPA
TargetQueryPredictor
BioActionConditionBatch
BioActionJEPALossWeights
bioaction_jepa_loss
```

Use existing RNAEncoder/ImageEncoder/PerturbationEncoder where possible.

Do not wire raw-linear PLS into BioAction-JEPA except for optional bootstrap loss.

## Step 3: Add same-condition cross-modal JEPA

Implement:

```text
RNA context -> image condition teacher state
image context -> RNA condition teacher state
joint context -> held-out modality state
```

## Step 4: Add transition JEPA

Implement:

```text
control state + action token -> perturbed teacher state
```

Start with pooled condition states. Then add prototypes.

## Step 5: Add program/region target masks

Implement semantic masks:

```text
RNA: gene_program_assignment blocks for synthetic data
Image: contiguous patch blocks / channel groups / large regions
```

Fallback to random blocks if no annotations exist.

## Step 6: Add collapse prevention

Implement VICReg/covariance/Barlow diagnostics and loss.

## Step 7: Train/evaluate smoke

Run tests and smoke commands.

## Step 8: Tier 1 experiments

Run the smallest useful candidates and document decisions.

---

# 18. Hard restrictions

Forbidden:

```text
1. Do not use condition_key or biological_key one-hot as model input.
2. Do not train on test target rows.
3. Do not promote on exact-key test alone.
4. Do not use raw-linear PLS heads as final BioAction-JEPA state.
5. Do not let raw reconstruction dominate the objective.
6. Do not remove protected PLS/Family N/Family O baselines.
7. Do not overwrite existing baseline artifacts.
8. Do not hand-pick named genes as losses for promotion.
9. Do not claim a result is novel unless it passes held-out support gates.
10. Do not continue after a stop condition fires.
```

Allowed:

```text
1. Create new model files.
2. Create new training and evaluation scripts.
3. Refactor reusable utilities carefully.
4. Add configs and tests.
5. Use PLS as a short annealed bootstrap teacher.
6. Add external biology resources only with provenance logging.
7. Add optional pretrained gene embeddings if provenance and fallback are clean.
```

---

# 19. Stop conditions

Stop the autonomous loop and write `final_report.md` if any occurs:

```text
1. hard experiment cap reached: 20 experiments unless user later extends it
2. three consecutive Tier 1 failures from collapse or not-real-JEPA identity
3. two Tier 2 failures from same mechanism family with same failure mode
4. held-out split support cannot be implemented without modifying data leakage rules
5. metrics show no headroom beyond simple baselines
6. batch leakage dominates latent state across two candidate families
7. required provenance or baseline ambiguity cannot be resolved
8. user-directed closure
```

When stopping:

```text
finish current run if already started
write journal entry
update results.tsv
update family_allocation.md
write final_report.md
stop launching new experiments
```

---

# 20. Final deliverables

At the end of the implementation phase, the repo should contain:

```text
perturb_jepa/models/bioaction_jepa.py
perturb_jepa/training/bioaction_batches.py
perturb_jepa/training/bioaction_losses.py
perturb_jepa/training/bioaction_trainer.py
perturb_jepa/evaluation/bioaction_metrics.py
scripts/train_bioaction_jepa.py
scripts/evaluate_bioaction_jepa.py
tests/test_bioaction_jepa_model.py
tests/test_bioaction_jepa_losses.py
tests/test_bioaction_condition_pairs.py
tests/test_bioaction_eval_split.py
outputs/autoresearch_bioaction_jepa_v1/BASELINE_REGISTRY.md
outputs/autoresearch_bioaction_jepa_v1/papers_consulted.md
outputs/autoresearch_bioaction_jepa_v1/research_journal.md
outputs/autoresearch_bioaction_jepa_v1/results.tsv
outputs/autoresearch_bioaction_jepa_v1/final_report.md if stop condition fires
```

`model_card.md` for the best candidate must include:

```text
Model name
Commit
Config
Training data and splits
JEPA tasks active
Teacher/target design
Whether raw reconstruction is auxiliary or zero
Whether PLS is used and how
Baselines compared
Tier reached
Promotion status
Known failures
Recommended next experiment
```

---

# 21. Launch sequence

Start with:

```bash
mkdir -p outputs/autoresearch_bioaction_jepa_v1/{step0_baselines,experiments,insights}
```

Then:

```text
1. Read required files.
2. Write papers_consulted.md from the literature seed list.
3. Write BASELINE_REGISTRY.md with current PLS/Family N/Family O numbers and provenance.
4. Patch --eval-split support.
5. Implement BioAction-JEPA minimal model and tests.
6. Run tests.
7. Run smoke training/evaluation.
8. Write jepa_identity_report.md.
9. Run Tier 1.
10. Decide using exact labels.
```

Begin now. Do not ask for another architecture clarification; this prompt grants architectural freedom within the protected-baseline and real-JEPA constraints above.
