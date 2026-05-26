# External Resources

## Norman / GEARS Processed Data

- Download URL: https://dataverse.harvard.edu/api/access/datafile/6154020
- Local h5ad: `data/raw/gears_norman/norman/perturb_processed.h5ad`
- Local h5ad bytes: `2228610012`
- Version marker: Harvard Dataverse datafile ID `6154020`, GEARS processed `norman/perturb_processed.h5ad`.
- License: not embedded in the h5ad; do not redistribute the raw file from this repo without checking Dataverse terms.
- Gene-symbol resolution: `outputs/autoresearch_norman_v1/step0_baselines/gene_symbol_resolution.tsv` maps h5ad var index IDs to `var['gene_name']`.
- Ortholog mapping: none applied.
- Metadata caveat: h5ad `cell_type` is `A549`; protocol and papers refer to Norman 2019 K562. This run preserves source metadata and does not relabel cells.
