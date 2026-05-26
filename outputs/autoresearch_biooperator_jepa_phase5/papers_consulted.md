# Papers Consulted

| Source | Field | Mechanism Extracted | Phase 5 Use |
| --- | --- | --- | --- |
| V-JEPA 2 / V-JEPA 2-AC, https://arxiv.org/abs/2506.09985 | latent world models | stop-gradient latent prediction with action-conditioned transition | used as identity constraint for future BioOperator-JEPA wrapper |
| Flow Matching for Generative Modeling, https://arxiv.org/abs/2210.02747 | vector-field learning | learn vector fields against known trajectories | deferred; Phase 5 first requires ridge/operator contracts |
| Efficient Flow Matching using Latent Variables, https://arxiv.org/html/2505.04486v2 | latent flow matching | latent-variable transport efficiency | deferred until deterministic operator floor is passed |
| Koopman/control-inspired latent dynamics, https://openreview.net/forum?id=fkrYDQaHOJ | controlled dynamics | low-rank or structured action-conditioned linear operator | used in planned low-rank control-affine operator |
| CellOT, https://www.nature.com/articles/s41592-023-01969-x | single-cell perturbation transport | preserve population structure across perturbation maps | deferred until single-condition latent operator contracts pass |
| Distributional Transport for Single-Cell Perturbation Prediction, https://arxiv.org/html/2511.13124v1 | stochastic transport | stochastic dynamic maps for unpaired populations | rejected for Phase 5 Tier 1; stochastic transport would bypass deterministic floor |
