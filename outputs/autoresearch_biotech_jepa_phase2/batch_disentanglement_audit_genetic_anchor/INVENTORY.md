# Inventory

- Datasets: `synth_genetic_anchor_lite, synth_batch_confound_lite`
- Eval splits: `test_heldout_perturbation, test`
- Seeds: `0, 1, 2`
- Device: `cpu`
- Phase 1 checkpoint references are loaded only if local checkpoint files exist.

## Tables

- `split_confounding.tsv`: `36` rows
- `anchor_summary.tsv`: `6` rows
- `raw_signal_batch_probe.tsv`: `54` rows
- `teacher_target_batch_probe.tsv`: `48` rows
- `split_half_ceiling.tsv`: `6` rows
- `loss_geometry.tsv`: `5` rows
