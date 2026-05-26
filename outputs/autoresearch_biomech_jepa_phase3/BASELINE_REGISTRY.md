# Phase 3 Baseline Registry

## Active Model Of Record

- Protected model: rank-3 train-split-only PLS raw-linear readout
- Role: protected model of record and audit baseline only
- Promotion rule: no Phase 3 Tier 1 or Tier 2 result can promote. Only a future Tier 3 pass can supersede this protected model.
- PLS restriction: may be used only as protected baseline, audit reference, or short annealed bootstrap teacher with weight decayed to zero. It is not a JEPA path.

## Phase 2 Diagnostic References

These references are not promoted baselines.

| Reference | Dataset/split | Seed | Metric | Value |
| --- | --- | ---: | --- | ---: |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | RNA->image recall@1 | 0.1875 |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | image->RNA recall@1 | 0.0000 |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | transition-to-target recall@1 | 0.4375 |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | transition source cosine improvement | +0.0161 |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | joint `z_bio` batch-probe accuracy | 0.1875 |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | joint `z_tech` batch-probe accuracy | 0.4375 |
| `BTJ001_synth_genetic_anchor_seed0` | `synth_genetic_anchor_lite/test_heldout_perturbation` | 0 | joint `z_bio` effective rank | 7.5103 |
| `BTJ002_norman_rna_only_seed0` | `norman_gears_processed/test` | 0 | RNA-only diagnostic | 1.0000 |
| `BTJ002_norman_rna_only_seed0` | `norman_gears_processed/test` | 0 | transition-to-target recall@1 | 0.0625 |
| `BTJ002_norman_rna_only_seed0` | `norman_gears_processed/test` | 0 | transition source cosine improvement | +0.0313 |
| `BTJ002_norman_rna_only_seed0` | `norman_gears_processed/test` | 0 | target `z_bio` effective rank | 7.4066 |

## Protected And Audit References From Earlier Phases

| Reference | Dataset/split | Metric | Value |
| --- | --- | --- | ---: |
| protected PLS synthetic model of record | `synth_heldout_perturbation_lite/test_heldout_perturbation` | RNA->image recall@1 | 0.1852 |
| protected PLS synthetic model of record | `synth_dose_extrapolation_lite/test_heldout_dose` | RNA->image recall@1 | 0.1806 |
| Family N expression-space reference | Phase 1/2 reports | status | audit reference only |
| Family O count-aware reference | Phase 1/2 reports | status | audit reference only |
| genetic-anchor audit | `synth_genetic_anchor_lite/test_heldout_perturbation` | split-half RNA->image same-bio recall@1 ceiling | 0.6389 |
| genetic-anchor audit | `synth_genetic_anchor_lite/test_heldout_perturbation` | max raw/protected batch-probe excess over majority | 0.6667 |
| genetic-anchor audit | `synth_genetic_anchor_lite/test_heldout_perturbation` | max Phase 1/zero-step representation batch-probe excess over majority | 0.6148 |

## Norman Metadata Constraints

- Norman file: `data/raw/gears_norman/norman/perturb_processed.h5ad`
- Cell type: A549 only
- Exposed batch/acquisition metadata: none
- Dose handling: fixed guide presence only, not chemical concentration
- Promotion limitation: Norman cannot promote a cross-modal or batch-disentanglement model from this processed h5ad.
