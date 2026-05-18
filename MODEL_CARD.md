# Perturb-JEPA Bridge Model Card

## Model Summary

Perturb-JEPA Bridge is a scaffold for unpaired perturbation modeling across
single-cell RNA-seq and label-free microscopy. It trains RNA and image encoders
with masked reconstruction and masked latent JEPA teacher/student objectives,
then aligns condition-level bags instead of assuming cell-image pairs.

## Intended Use

- Cross-modal retrieval between perturbation RNA profiles and image embeddings.
- Counterfactual perturbation response evaluation at condition-bag level.
- Leakage diagnostics for metadata-only and batch-only baselines.
- Synthetic smoke tests for training, checkpointing, and evaluation plumbing.

The current repository is not a validated biomedical predictor. It is a research
and engineering scaffold.

## Non-goals

- Pixel-level image generation.
- Individual-cell or cell-to-image matching.
- Clinical decision support or therapeutic recommendation.
- Claims of causal validity without held-out perturbation, batch, dose/time, and
  cell-line validation.

## Assumptions

Required biological metadata should include `perturbation`, `dose`, `time`, and
`cell_line`, with optional `perturbation_type`, `target_gene`, `compound_id`,
`moa`, and `pathway`. Technical metadata should include `batch`, `plate`, `run`,
`well`, `site`, `z_plane`, `imaging_channel`, `sequencing_lane`, and
`library_id` when available. The default condition key is
`perturbation|dose|time|cell_line`.

Splits should be leakage-safe and grouped by perturbation, dose, time, cell line,
and batch. Retrieval scoring must use embeddings only; metadata is used after
scoring to define relevance, enrichment labels, and strata.

`condition_key`, `condition_key_fine`, and `condition_id` are the same
four-field biological key. `perturbation_type` is optional metadata and is only
included in the explicit `condition_key_with_type` column.

## Leakage Risks

Batch, plate, well, site, run, z-plane, sequencing lane, or library identifiers
can be confounded with perturbation identity. These fields must never be included
in the biological condition key or concatenated into RNA/image encoder inputs.
They may be used for grouping, split construction, adversarial batch prediction,
negative/positive selection, and evaluation stratification.

Metadata-only and batch-only baselines are required. If batch-only retrieval is
competitive with learned embeddings, the dataset or split is likely leaking
technical acquisition information.

## Counterfactual Interpretation

Counterfactual predictions are condition-level distributional forecasts. The
model predicts a treated prototype distribution relative to a control prototype
distribution and returns uncertainty through log-variance outputs. These outputs
should be interpreted as bag-level latent response distributions, not as
individual-cell counterfactuals. Bridge-level training does not optimize a
counterfactual loss unless explicit control and treated targets are supplied; the
RNA counterfactual script is the explicit control-to-treated pathway.

## Required Validation Splits

Report random sample splits only as a convenience baseline. Scientific claims
should include held-out batch, held-out perturbation, held-out dose/time,
held-out cell line, and held-out MoA splits when MoA labels exist.

Held-out perturbation generalization requires descriptor features. If the model
uses only perturbation ID embeddings, unseen perturbations map to `unknown` and
the result is not a true perturbation extrapolation. Suitable descriptors include
chemical fingerprints, target-gene embeddings, MoA/pathway embeddings, or
pretrained perturbation descriptors.

## Evaluation

Cross-modal retrieval reports RNA->image and image->RNA Recall@1/5/10, mAP,
median rank, same-perturbation enrichment, optional same-MoA enrichment, and
stratified metrics over batch, perturbation, dose, time, and cell line.

RNA counterfactual metrics include pseudobulk correlation, logFC correlation,
top-k differential-expression overlap, direction accuracy, optional pathway score
correlation, and latent MMD. Supported grouping modes include condition,
perturbation, dose-time, and held-out perturbation.

Image counterfactual metrics include distance to observed bag embedding, true
condition retrieval rank, replicate correlation, dose/time ordering accuracy, and
same-MoA enrichment.

## Baselines

- Metadata-only retrieval uses only `perturbation`, `dose`, `time`, and
  `cell_line`.
- Batch-only retrieval uses only technical acquisition metadata.
- Mean prototype alignment maps source rows to target-space mean prototypes by
  condition metadata. Eval-target fitting is reported as
  `mean_prototype_oracle`; train-target fitting is reported as
  `mean_prototype_trainfit`.

## Limitations

Synthetic entrypoints verify code paths but do not establish biological validity.
Real-data stage scripts can load AnnData RNA files and image manifests, but
bridge/fine-tune training requires actual overlap in biological condition labels
between modalities. Checkpoint evaluation reuses the saved metadata vocab, with
unknown eval categories mapped to ID 0 unless strict vocab mode is enabled. MoA
and pathway metrics are only meaningful when labels or gene sets are curated
consistently across modalities.

## Expected Failure Modes

- Collapsed embeddings that reduce both batch prediction and perturbation
  prediction to random.
- Strong retrieval caused by shared technical metadata rather than biological
  signal.
- Poor held-out perturbation performance despite strong random-split retrieval.
- Dose/time extrapolation errors when training data cover sparse response grids.
- Misleading MoA enrichment when MoA annotations are noisy or reused across
  unrelated mechanisms.
