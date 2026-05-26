# Phase 4 Baseline Registry

## Active Model Of Record

- Protected model: rank-3 train-split-only PLS raw-linear readout
- Role: protected baseline and audit reference only
- Status: model of record remains unchanged unless a future Tier 3 pass supersedes it
- PLS restriction: never used as the BioFlow-JEPA main representation path

## Required Phase 2/3 References

| Reference | Dataset/split | Metric | Value |
| --- | --- | --- | ---: |
| `BTJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | transition source cosine improvement | +0.0161 |
| `BTJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | transition-to-target recall@1 | 0.4375 |
| `BTJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | batch allocation gap | +0.2500 |
| `BTJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | RNA->image recall@1 | 0.1875 |
| `BTJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | image->RNA recall@1 | 0.0000 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | transition source cosine improvement | -0.1695 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | transition-to-target recall@1 | 0.0968 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | delta cosine | -0.0332 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | delta prediction effective rank | 4.7813 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | delta teacher effective rank | 12.4196 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | image->RNA recall@1 | 0.1290 |
| `BMJ001` | `synth_genetic_anchor_lite/test_heldout_perturbation` | RNA->image recall@1 | 0.0968 |
| `BTJ002` | `norman_gears_processed/test` | transition source cosine improvement | +0.0313 |
| `BTJ002` | `norman_gears_processed/test` | transition-to-target recall@1 | 0.0625 |

Raw artifact source files:

- `outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/metrics_eval.json`
- `outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ002_norman_rna_only_seed0/metrics_eval.json`
- `outputs/autoresearch_biomech_jepa_phase3/experiments/BMJ001_delta_synth_seed0/metrics_eval.json`
- `outputs/autoresearch_biomech_jepa_phase3/final_report.md`

No discrepancies were found between the prompt values and raw artifacts for the values above.
