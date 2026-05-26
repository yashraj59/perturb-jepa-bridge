# Inventory

- Datasets: `synth_micro, synth_heldout_perturbation_lite, synth_dose_extrapolation_lite, synth_batch_confound_lite`
- Eval splits: `test, test_heldout_perturbation, test_heldout_dose, test`
- Seeds: `0, 1, 2`
- Device: `cpu`
- Phase 1 checkpoint references are loaded only if local checkpoint files exist.

## Tables

- `split_confounding.tsv`: `72` rows
- `anchor_summary.tsv`: `12` rows
- `raw_signal_batch_probe.tsv`: `108` rows
- `teacher_target_batch_probe.tsv`: `136` rows
- `split_half_ceiling.tsv`: `12` rows
- `loss_geometry.tsv`: `5` rows
