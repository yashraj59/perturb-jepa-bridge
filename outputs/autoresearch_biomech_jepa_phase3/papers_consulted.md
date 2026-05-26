# Papers Consulted

These notes record the paper-level ideas Phase 3 is allowed to translate into mechanisms. They are not claims of benchmark equivalence.

| Anchor | Technique extracted | Mapping to BioMechanistic-JEPA |
| --- | --- | --- |
| I-JEPA | Predict latent targets selected by queries instead of reconstructing pixels directly. | RNA program queries, image region queries, and prototype target queries remain latent JEPA targets. |
| V-JEPA / V-JEPA 2 | Predict future latent state in a world-model style, including action-conditioned prediction. | Treat a perturbation as the biological action and predict `control z_bio + action -> perturbed z_bio`. |
| BYOL / DINO-style EMA teachers | Online network predicts stop-gradient targets from a slowly moving teacher. | Keep EMA RNA/image target encoders and detached teacher latents. |
| VICReg / Barlow Twins | Prevent collapse with variance floors, covariance penalties, and decorrelation. | Monitor and penalize collapse in `z_bio`, `z_tech`, action deltas, predicted deltas, and population prototypes. |
| GEARS | Genetic perturbation structure and combinatorial perturbation generalization. | Use non-leaky gene/gene-pair action descriptors and optional program action tokens. |
| CPA | Compositional perturbation effects. | Factor actions into gene effects, pair interactions, cell context, and dose only for chemical screens. |
| CellOT / neural optimal transport | Population-level perturbation maps. | Add prototype/set transition loss between predicted and teacher target biological prototypes. |
| scVI/sysVI-style technical modeling | Model technical nuisance explicitly rather than erasing everything adversarially. | Keep `z_tech` for batch/acquisition variation and restrict retrieval/transition to `z_bio`. |
| Robotics/action world models | Action-token transition operators over state. | Encode perturbation as action tokens used by delta and population transition predictors. |
