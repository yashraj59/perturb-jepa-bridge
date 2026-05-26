# Delta Latent Cache Summary

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Checkpoint: `outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt`
- Cache device: `cpu`
- Train rows: `72`
- Eval rows: `27`
- z_bio dim: `24`
- action descriptor dim: `20`
- Train delta rank/std/norm: `13.5627` / `0.0846` / `0.4310`
- Eval delta rank/std/norm: `11.7819` / `0.0832` / `0.4252`

Cached fields include teacher/online source and target `z_bio`, teacher `z_tech`, action descriptors, perturbation/cell-line/batch labels, split labels, and condition labels for diagnostics only.
