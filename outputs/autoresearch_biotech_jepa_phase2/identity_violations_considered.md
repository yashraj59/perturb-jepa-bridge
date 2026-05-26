# Identity Violations Considered

- Rejected: longer training of the prior BioAction Family A architecture. Reason: amendment explicitly says the blocker is not insufficient training.
- Rejected: increasing batch-centroid invariance weight again. Reason: Phase 1 Family F already failed at weights `0.5` and `5.0`.
- Rejected: using `condition_key`, `biological_key`, exact target-key one-hots, or test target means as model inputs or JEPA targets. Reason: leakage and prompt violation.
- Rejected: promoting exact `synth_micro/test` results alone. Reason: exact-key split is matching dominated and not a promotion basis.
