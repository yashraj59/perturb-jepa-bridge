# F011 Protected-Floor Seed-Stability Audit

## Decision
`F011_RECALL_FLOOR_SEED_SPECIFIC_CONTINUOUS_METRICS_STABLE`

## Evidence
- Seed0 protected condition recall@1: `0.481481`
- Fresh-seed condition recall@1 mean: `0.296296`
- Fresh-seed condition recall@1 max: `0.333333`
- Seed0 minus fresh mean recall gap: `0.185185`
- Seed0 transition improvement: `0.005662`
- Fresh-seed transition improvement mean: `0.012686`
- Seed0 delta cosine: `0.397963`
- Fresh-seed delta cosine mean: `0.464903`

## Interpretation
The protected recall gate is not stable across synthetic seeds: seed0 is substantially higher than every fresh seed, while transition improvement and delta cosine do not show the same degradation. This does not change the protected model of record or any locked gate by itself. It does mean more architecture search against the seed0 recall floor is likely chasing a seed-specific retrieval artifact unless the benchmark/metric definition is revisited under a separate, explicit amendment.

## Consequence
Do not promote any model. Do not change gates here. The next debate council should favor benchmark/metric redesign documentation over another architecture candidate.
