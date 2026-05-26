# Identity Violations Considered

- PLS/raw-linear main path: forbidden except audit-only baseline.
- `condition_key`, `biological_key`, exact target-key one-hot: forbidden as model, residual, calibration, or selection features.
- Eval/test target rows: scoring-only; forbidden for fitting, whitening, residual selection, and calibration.
