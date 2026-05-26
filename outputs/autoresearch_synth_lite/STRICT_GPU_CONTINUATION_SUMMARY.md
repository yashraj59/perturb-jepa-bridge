# Strict GPU Continuation Summary

Target experiment range: 111-210

Completed in this invocation: 100

Recorded rows in target range: 100

## Decision Labels

- TIER1_DISCARD_COUNTERFACTUAL_GATE_INCOMPLETE: 16
- TIER1_DISCARD_SEED2_PROGRAM_REGRESSION: 72
- TIER1_DISCARD_SEED2_PROGRAM_UNDER_GATE: 12

## Key Findings

- All 100 continuation trials used CUDA after a pre-run GPU check. Maximum recorded CUDA allocation was `0.060` GB.
- The MLP program decoder loss grid did not repair seed 2; best program delta remained negative at about `-0.0574`.
- The shallow linear program decoder without ridge initialization changed the seed-2 failure from regression to weak positive recovery, best program delta `+0.0206`, still under the `+0.05` gate.
- Train-split-only prefit ridge initialization for the linear program decoder was the strongest mechanism. All 16 ridge variants produced seed-2 program deltas around `+0.25`.
- No trial passed the full counterfactual gate. The best ridge variant by program recovery was Experiment 180 with program delta `+0.2523`, direction delta `+0.5598`, logFC delta `+0.0492`, pseudobulk delta `+0.0101`, and top-50 overlap delta `+0.0167`.
- The best ridge variant by top-50 overlap was Experiment 192 with top-50 delta `+0.0200`, logFC delta `+0.0636`, and program delta `+0.2494`; it still missed the top-50 gate of `+0.03`.

## Interpretation

The active blocker is no longer seed-2 program recovery in isolation. A ridge-initialized linear program decoder can recover seed-2 program effects while preserving protected geometry. The remaining blocker is aligning that program-level recovery with top-DE overlap, and secondarily avoiding marginal logFC misses, without using real marker/pathway resources.
