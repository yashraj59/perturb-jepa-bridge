# Papers Consulted

This Phase 2 log records mechanisms extracted for diagnostic and possible future BioTech-JEPA work. No architecture search is reopened until the Stage 1 audit passes.

| Family | Representative papers | Mechanism extracted | Mapping to BioTech-JEPA | Stage |
| --- | --- | --- | --- | --- |
| JEPA / world models | I-JEPA, V-JEPA, V-JEPA 2 | Online/context encoders, EMA target encoders, stop-gradient latent targets, query predictors, action-conditioned transition prediction | Keep real JEPA identity while moving biological transition prediction onto `z_bio` | Audit prerequisite / possible Stage 2 |
| Domain generalization and causal invariance | DANN, IRM, causal/domain representation learning | Split invariant and environment-dependent factors; use environment-stable prediction rather than only scrubbing batch | Separate `z_bio` and `z_tech`; require per-environment transition stability | Possible B1/B3 |
| Single-cell batch modeling | scVI, scANVI, sysVI, integration benchmarks | Model technical variation separately and evaluate biological retention, not batch removal alone | `z_tech` explains batch/library/dropout/image intensity; `z_bio` must retain perturbation signal | Stage 1 audit / possible B1 |
| Perturbation prediction | CellOT, GEARS, CPA | Perturbations are distributional/compositional actions; graph/program priors can improve held-out perturbations | Latent transition JEPA, optional OT population loss, optional graph/program action prior | Possible B4/B5 |
