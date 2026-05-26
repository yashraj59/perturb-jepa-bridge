# Identity Violations Considered

- PLS/raw-linear main path: forbidden; audit-only model of record.
- `condition_key`, `biological_key`, exact target-key one-hot: forbidden as model inputs.
- Test/eval target means and pooled train+test statistics: forbidden.
- Batch id as biological transition shortcut: forbidden.
