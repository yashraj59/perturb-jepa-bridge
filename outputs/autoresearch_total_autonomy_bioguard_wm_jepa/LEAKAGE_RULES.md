# Leakage Rules

Forbidden in model inputs, training targets, calibration, and selection:
- condition_key
- biological_key
- exact target-key one-hot features
- eval/test target means
- eval/test target rows for fitting
- eval/test whitening/statistics
- pooled train+test statistics
- batch id as biological transition shortcut
- raw-linear PLS as candidate main representation path

Allowed:
- labels for diagnostics and retrieval scoring only
- train-only action descriptors
- batch labels for z_tech diagnostics only
