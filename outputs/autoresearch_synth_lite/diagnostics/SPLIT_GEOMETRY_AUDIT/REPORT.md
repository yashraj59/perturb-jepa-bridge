# Split Geometry Audit

Dataset: `synth_micro`
Seed: `0`
Wallclock minutes: `1.317`

## Main Finding

The synthetic retrieval target is well-posed but the learned model geometry is already collapsed when all train condition bags are collected in eval mode. The regular synthetic split is label-interpolation, not unseen-condition extrapolation: train, val, and test share the same condition and bag keys. However, the condition-bag cell counts are imbalanced: train bags average `6.0` cells, while test bags average `1.0` cell. The minimum val/test bag size is `1.0`. Normal Step 0 evaluation asks for `8` cells per bag, so val/test bags are sampled with replacement from very small groups. That cell-count issue is real, but it is not sufficient to explain the failure because train collection also collapses below the hard std gate.

## Oracle And Raw Baselines

- True latent test recall@1: `1`
- Raw ridge RNA->image test recall@1: `0.21875`
- Raw RNA->z_bio test R2: `0.277057`
- Raw image->z_bio test R2: `0.988235`

## Collapse Location

### attention

- Full-bag train RNA shared min std: `0.00400319`
- Full-bag test RNA shared min std: `0.00714596`
- Full-bag train image shared min std: `0.00457279`
- Full-bag test image shared min std: `0.00470986`
- Single-cell train image shared min std: `0.00532048`
- Single-cell test image shared min std: `0.00470987`
- Full-bag test shared recall@1: `0.0625`

### mean

- Full-bag train RNA shared min std: `0.00909638`
- Full-bag test RNA shared min std: `0.0123266`
- Full-bag train image shared min std: `0.00422609`
- Full-bag test image shared min std: `0.00512167`
- Single-cell train image shared min std: `0.00506197`
- Single-cell test image shared min std: `0.00512168`
- Full-bag test shared recall@1: `0.0625`

## Interpretation

The true-latent oracle works and raw ridge RNA-to-image is above random on test, so the synthetic generator is not completely underdetermined. The learned model fails before retrieval: instance-level projected embeddings are already below the collapse threshold, then aggregation/state heads preserve or amplify the low-variance geometry. The next diagnostic should inspect raw encoder CLS outputs, pre-normalized projection outputs, normalized projection outputs, and train-mode vs eval-mode dropout. This distinguishes an encoder signal problem from projection normalization and from dropout-masked collapse.

## Artifacts

- `SPLIT_SUMMARY.tsv`: split coverage and condition-bag cell counts.
- `LATENT_SUPPORT.tsv`: true latent/raw modality support distances against train.
- `ORACLE_AND_RAW_BASELINES.tsv`: true-latent and ridge raw-modality baselines.
- `MODEL_STAGE_GEOMETRY.tsv`: variance/rank by model stage, split, aggregator, and eval bag size.
- `MODEL_RETRIEVAL_BY_SPLIT.tsv`: retrieval by stage pair, split, aggregator, and eval bag size.
