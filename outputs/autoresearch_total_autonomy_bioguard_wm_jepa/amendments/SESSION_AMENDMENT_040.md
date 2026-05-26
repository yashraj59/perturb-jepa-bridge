# Session Amendment 040: Activate Multiseed v1 Benchmark

## Trigger
`F013_MULTISEED_BENCHMARK_REGISTRY_BUILT_PROPOSED_GATES_NOT_ACTIVE`

## Evidence
F013 built `synthetic_genetic_anchor_lite_multiseed_v1` across five synthetic seeds. The old seed0 benchmark remains locked as historical reference, but future architecture work should target the new named benchmark unless an amendment explicitly says otherwise.

## Active Model Of Record
Protected rank-3 train-split-only PLS raw-linear readout remains the model of record. Activation of this benchmark does not promote a model.

## Active Search Benchmark
`synthetic_genetic_anchor_lite_multiseed_v1`

## Baseline Registry
- full action-ridge transition mean: `0.011281` +/- `0.008060`
- full action-ridge delta cosine mean: `0.451515` +/- `0.057127`
- full action-ridge condition recall mean: `0.333333` +/- `0.077690`

## Provisional Active Gates For v1
These gates apply only to `synthetic_genetic_anchor_lite_multiseed_v1` and do not alter the old benchmark.

Primary continuous gates:
- mean transition improvement must be >= `0.003221` (baseline mean - 1 std)
- mean delta cosine must be >= `0.394388` (baseline mean - 1 std)

Secondary/no-regression gates:
- condition recall@1 must be >= `0.255644` (baseline mean - 1 std)
- effective rank must remain within 20% of full action-ridge mean
- magnitude ratio must remain within 25% of full action-ridge mean

Promotion rule:
A candidate can only be considered a real improvement if it exceeds the full action-ridge mean on at least one primary continuous metric without failing secondary gates. Tier 3 promotion still requires a full no-regression package.

## Exact Next Experiment
`BGWM003_MULTISEED_V1_ACTION_ADALN_SANITY`

## Implementation Tasks
- Run the existing action-AdaLN residual candidate on seed caches 0-4.
- Use train-only calibration per seed.
- Aggregate against the v1 registry.
- Do not save/promote checkpoints unless a nonzero residual passes gates.

## Do-Not-Run List
Do not mutate old `test_heldout_perturbation` gates. Do not use eval target rows for fitting or residual selection. Do not promote a Tier 1/Tier 2 sanity result.

## Hard Escalation Check
No hard escalation trigger present.

## Continuation Rule
Ordinary failure triggers another Debate Council. Do not wait for the user.

Council-selected proposal: F014 activate multiseed benchmark gates
