# Norman Context Audit

File inspected:

```text
data/raw/gears_norman/norman/perturb_processed.h5ad
```

## Basic Shape

- Cells: `91,205`
- Genes: `5,045`
- Conditions: `284`
- Cell type: `A549` only

## Observed Metadata

Available `obs` columns:

```text
condition
cell_type
dose_val
control
condition_name
```

No exposed batch, plate, well, donor, lane, library, site, or split metadata was found in this processed h5ad.

## Dose Interpretation

`dose_val` values:

- `1`: control rows
- `1+1`: combinatorial perturbation rows

This is not a chemical concentration series. For this Norman-specific experiment, `dose_val` should be treated as guide-count / perturbation-composition notation and not as a chemical dose axis.

## Perturbation Structure

- `condition` has `284` unique values.
- Most rows are perturbation rows (`control = 0`).
- Controls: `7,353` rows.
- `single_or_ctrl` count from condition strings containing `ctrl`: `55,760`.
- double perturbation rows without `ctrl`: `35,445`.

## Consequence For Phase 2

Norman can support a genetic perturbation action setup:

```text
action = gene perturbation or gene-pair perturbation
context = A549
dose = ignored / fixed
batch = unavailable in this processed file
```

Norman cannot directly validate `z_tech` batch separation unless additional batch/acquisition metadata is recovered from another source. Batch-disentanglement must therefore be tested on synthetic data or another real dataset with exposed batch metadata.

## Norman BioTech-JEPA Diagnostic

The user approved ignoring batch and chemical dose for this specific Norman experiment. A low-compute RNA-only BioTech-JEPA diagnostic was run:

```text
outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ002_norman_rna_only_seed0
```

Important constraints:

- Training uses GEARS-style train conditions only.
- Evaluation uses `test` target conditions only after training.
- Action input is a gene multi-hot descriptor, not an exact condition-key one-hot.
- `batch_id` is fixed to zero and is not interpreted as biological metadata.
- `dose` is fixed to one to represent guide presence, not chemical concentration.

Result:

- Transition source cosine improvement: `+0.0313`
- Transition-to-target recall@1: `0.0625`
- Target `z_bio` effective rank: `7.4066`
- Batch disentanglement cannot be evaluated from this processed h5ad because no batch/acquisition label is exposed.
