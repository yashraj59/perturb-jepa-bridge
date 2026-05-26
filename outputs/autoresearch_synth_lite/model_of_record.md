# Model Of Record

- Commit: `0436cccd99a733fec6b57dfdb1fe1e6f4982922f`
- Model: `perturb_jepa.models.bridge.PerturbJEPABridge`
- Protected: JEPA training loop, RNAEncoder scRNA-seq path, masked/predictive objective, forward output contract, counterfactual outputs.
- Allowed changes: lightweight losses, schedules, pooling, objective weights, and diagnostics that preserve the core concept.

## Synthetic Geometry Model Of Record

- Mechanism: rank-3 prefit PLS raw-linear readout.
- RNA readout: `raw_linear_pseudobulk`.
- Image readout: `raw_linear_pooled`.
- Source artifact: `outputs/autoresearch_synth_lite/diagnostics/PREFIT_PLS_READOUT/TIER2_TIER3_SUMMARY.md`.
- Status: promoted for synthetic condition-level shared geometry after Tier 2 and focused Tier 3 pass.
- Caveat: this is a closed-form readout initializer/baseline, not yet a trained JEPA checkpoint promotion.

## Trainable Geometry Handoff Baseline

- Mechanism: separate trainable linear student clone of the rank-3 PLS readout.
- RNA student output: `rna_distilled_linear_shared`.
- Image student output: `image_distilled_linear_shared`.
- Source artifact: `outputs/autoresearch_synth_lite/diagnostics/PLS_DISTILLED_HEAD/TIER2_SUMMARY.md`.
- Status: Tier 2 clean engineering baseline; it exactly matches the protected PLS geometry and does not replace the frozen retrieval path.
- Caveat: this is not an improvement over PLS. It is the safe starting point for residual trainable adapters.
