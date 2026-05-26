# BioFlow-JEPA BFJ001 Model Card

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Device: `cuda`
- Steps: `50`
- Frozen encoders: `True`
- Transition mode: `vector_field`
- Delta whitening: `True`
- Source-improvement hinge: `True`
- Decision label: `BFJ_TIER1_DISCARD_NO_SIGNAL`

## Metrics

- transition_source_cosine_improvement: `-0.0104`
- delta_cosine: `-0.1054`
- delta_prediction_effective_rank: `7.6852`
- delta_teacher_effective_rank: `11.4999`
- transition_to_target_recall@1: `0.0500`
- delta_magnitude_ratio: `0.7867`
- image_to_RNA recall@1: `0.1000`
- RNA_to_image recall@1: `0.1500`

Protected PLS remains the model of record. This candidate cannot promote in Phase 4.
