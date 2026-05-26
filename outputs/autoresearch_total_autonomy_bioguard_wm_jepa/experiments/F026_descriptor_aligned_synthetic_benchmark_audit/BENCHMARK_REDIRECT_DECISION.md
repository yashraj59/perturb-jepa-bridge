# F026 Benchmark Redirect Decision

## Decision
`F026_DESCRIPTOR_ALIGNED_BENCHMARK_APPROVES_STEP0_REDESIGN`

## Actionable Interpretation
- Historical benchmark: `synth_genetic_anchor_lite`
- New benchmark: `synth_program_aligned_genetic_lite`
- Program descriptor geometry fixed: `True`
- True synthetic z_bio with non-exact program action passes active gates: `True`
- Observed RNA PCA with non-exact program action passes active gates: `True`

If reopened, the next JEPA work should use the new named benchmark for synthetic Step 0 and representation training. The old benchmark remains a locked historical artifact and should not be silently mutated.
