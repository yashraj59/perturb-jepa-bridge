# Family Allocation

| Family | Experiments used | Tier 1 keeps | Tier 2 passes | Tier 3 wins | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 Factorized Bio/Tech Latent | 2 | 0 | 0 | 0 | Tier 1 diagnostics run |
| B2 Cross-Batch Consensus Teacher | 0 | 0 | 0 | 0 | not implemented |
| B3 Environment-Stable Transition JEPA | 0 | 0 | 0 | 0 | not implemented |
| B4 Graph/Program Action Prior | 0 | 0 | 0 | 0 | gated by Stage 1 |
| B5 Latent OT Population Transition | 0 | 0 | 0 | 0 | gated by Stage 1 |

Architecture search is closed until the Stage 1 audit passes every reopening criterion.

Stage 1 result: `PHASE2_AUDIT_COMPLETE_DO_NOT_REOPEN`. All B1-B5 families remain unused because held-out perturbation and held-out dose splits lack cross-batch train anchors or a valid substitute for biological teacher construction.

## Genetic Amendment Update

After the user clarified that Norman/CRISPR-style perturbation should ignore chemical dose, `synth_genetic_anchor_lite` was added and audited.

Genetic anchor Stage 1 result: `PHASE2_AUDIT_COMPLETE_REOPEN`.

Architecture families became eligible for a targeted Tier 1 implementation on the synthetic genetic-anchor setting.

## Tier 1 Run Update

The user instructed that BioTech-JEPA should be run for both synthetic and Norman. B1 was used for two low-compute diagnostics:

- `BTJ001_synth_genetic_anchor_seed0`: paired synthetic RNA+image with held-out genetic perturbations.
- `BTJ002_norman_rna_only_seed0`: Norman RNA-only genetic perturbation diagnostic with batch/dose ignored.

Tier 1 keeps: `0`. Neither run supersedes the protected PLS model. The synthetic run shows the intended `z_tech > z_bio` batch-signal allocation but only modest transition gain. The Norman run is RNA-only and cannot validate cross-modal or batch disentanglement.
