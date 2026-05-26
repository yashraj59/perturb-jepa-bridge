# Leakage Report

- Mode: `frozen_neural_low_rank_equivalence`
- Train rows used for fitting: `72`
- Eval rows used for scoring only: `27`
- Fit uses train rows only: `True`
- Action feature names used by model: `action_descriptor_0, action_descriptor_1, action_descriptor_2, action_descriptor_3, action_descriptor_4, action_descriptor_5, action_descriptor_6, action_descriptor_7, action_descriptor_8, action_descriptor_9, action_descriptor_10, action_descriptor_11, action_descriptor_12, action_descriptor_13, action_descriptor_14, action_descriptor_15, action_descriptor_16, action_descriptor_17, action_descriptor_18, action_descriptor_19`
- Forbidden key tensors present: `False`
- `condition_key` used only as an evaluation/retrieval label, not as a model tensor.
- `biological_key` and exact target-key one-hot features are absent from model tensors.
- Test/eval target means are not used for fitting floor, basis, or residuals.
- Pooled train+test target statistics are not used.
- Raw-linear PLS is not used as the BioSpectral representation path.
