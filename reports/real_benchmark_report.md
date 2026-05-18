# Real Perturb-JEPA Benchmark Report

## Executive Summary

Runner status: `blocked_insufficient_compound_overlap`.

Real data were downloaded and inspected, but model training was not run. The mandated first BF-MoA plate, `P015080`, contains only two compound labels (`CBK200949` and `dmso`) and has zero normalized compound overlap with the A549 10uM/72h sci-Plex3 subset. This violates the hard minimum of six shared compounds for condition-level bridge training.

No cross-modal retrieval, counterfactual, or baseline metric should be interpreted as available for this run.

## Data Status

- sci-Plex3 downloaded: `data/raw/SrivatsanTrapnell2020_sciplex3.h5ad`
- BF-MoA metadata downloaded: `data/raw/bf_moa_data_tables.tar.gz`
- BF-MoA Figshare record downloaded: `data/raw/bf_moa_figshare.json`
- Smallest BF-MoA plate downloaded/extracted: `P015080`
- Raw BF-MoA manifest built: `data/processed/bf_moa_manifest_raw.csv`
- Aligned bridge inputs not created because preprocessing halted on zero shared compounds.

## Compound Intersection

After normalization and parenthetical synonym parsing, the selected sci-Plex3 subset had many named compounds, but the selected BF-MoA plate had only:

| BF-MoA compound | MoA | Sites |
|---|---|---:|
| `CBK200949` | topoisomerase inhibitor | 5 |
| `dmso` | dmso | 5 |

Shared compounds with A549 10uM/72h sci-Plex3: none.

## Implementation Notes

The code now supports `data.condition_key: condition_key_medium` in condition-bag bridge training and checkpoint retrieval evaluation. This is needed for sci-Plex3 A549/K562/MCF7 to bridge against BF-MoA U2OS without placing `cell_line` in the biological join key.

The new `scripts/prepare_real_pairing.py` script performs the requested normalization, synonym parsing, BF channel filtering, six-channel image assembly, overlap check, and hard halt when fewer than six shared compounds remain.

## Reproducibility

See `results/reproducibility_manifest.json` for command and checksum details, and `metrics/REPORT.md` for the full run report.
