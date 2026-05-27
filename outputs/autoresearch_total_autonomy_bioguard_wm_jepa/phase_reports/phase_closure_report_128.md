# Phase Closure Report 128

## Trigger
F102 strict scRNA plus imaging dataset discovery/preflight

## Interpretation
F102 found an actionable public paired single-cell RNA plus imaging candidate:
PerturbMulti. This is more suitable for fresh validation than cpg0003 Rosetta
because it includes single-cell RNA data and per-cell image/protein intensity
metadata rather than L1000 bulk-like signatures.

No model was trained and no model was promoted.

## Candidate Summary
| candidate | state | key evidence |
| --- | --- | --- |
| PerturbMulti | candidate found, RNA obs pending | public HF repository; CRISPR RNA H5AD; protein H5AD; perturbation metadata; spatial coordinates; per-cell image tar archives |
| HYPED | not actionable yet | paper describes long-read scRNA plus live-cell imaging, but no public payload manifest was found |

## Next Step
Run F103 metadata-only pairing preflight for PerturbMulti. Do not load the RNA
matrix into RAM. Do not start model validation until RNA obs columns and
RNA/protein/image cell-ID overlap pass.
