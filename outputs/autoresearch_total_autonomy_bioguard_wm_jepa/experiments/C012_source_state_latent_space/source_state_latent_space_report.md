# C012 Source-State Latent-Space Audit

## Decision
`C012_SOURCE_STATE_SIGNAL_STRONGER_OUTSIDE_Z_BIO`

## Evidence
- Best top-third latent space: `z_bio_online`.
- Best top-third same-cell-line purity: `0.911229`.
- z_bio top-third same-cell-line purity: `0.715682`.
- Best-minus-z_bio purity gain: `0.195547`.

## Interpretation
This audit checks whether source-state/cell-line structure is better represented in z_tech or joint source latents than in z_bio. It does not use these metadata fields as model inputs or promotion criteria.
