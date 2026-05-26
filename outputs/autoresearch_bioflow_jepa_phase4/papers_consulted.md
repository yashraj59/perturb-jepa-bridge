# Papers Consulted

| Source | Field | Technique extracted | BioFlow-JEPA mapping | Used |
| --- | --- | --- | --- | --- |
| I-JEPA, CVPR 2023, https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf | JEPA | Predict latent targets rather than reconstructing raw pixels. | Keep latent RNA/image target prediction and EMA teacher targets. | planned |
| V-JEPA, 2024, https://arxiv.org/abs/2404.08471 | JEPA/world models | Feature-prediction-first latent world model. | Transition prediction remains in latent `z_bio`, not raw expression/image space. | planned |
| V-JEPA 2, 2025, https://arxiv.org/abs/2506.09985 | action-conditioned JEPA | Action-conditioned latent prediction. | Perturbation/gene descriptor is the biological action. | planned |
| Flow Matching, 2022, https://arxiv.org/abs/2210.02747 | flow/transport | Learn vector fields between source and target distributions. | Replace one-shot delta MLP with latent vector field in `z_bio`. | planned |
| Conditional Flow Matching / OT-CFM, 2023, https://arxiv.org/abs/2302.00482 | conditional transport | Condition velocity on context/action. | Action-conditioned velocity from control to perturbed latent state. | planned |
| Rectified Flow, 2022/2023, https://arxiv.org/abs/2209.03003 | flow/transport | Straight-line source-target velocity objective. | Teacher delta is velocity target along source-target interpolation. | planned |
| Learning Koopman Invariant Subspaces, NeurIPS 2017, https://arxiv.org/abs/1710.04340 | latent dynamics | Stable low-rank linear transition structure. | Optional low-rank action-conditioned transition operator. | planned |
| Universal linear embeddings of nonlinear dynamics, Nature Communications 2018, https://www.nature.com/articles/s41467-018-07210-0 | latent dynamics | Linear dynamics in learned coordinates. | Low-rank Koopman-style action operator in `z_bio`. | planned |
| Koopman Operators in Robot Learning survey, 2025, https://arxiv.org/html/2408.04200v2 | control/robotics | Controlled latent operators for actions. | Perturbation as biological action. | planned |
| CellOT, https://www.nature.com/articles/s41592-023-01969-x | single-cell perturbation | Population-level transport metric/null. | Baselines and transport-style diagnostics. | audit |
| GEARS, https://www.nature.com/articles/s41587-023-01905-6 | genetic perturbation | Gene/gene-pair perturbation descriptors. | Non-leaky action descriptors. | audit |
| CPA, https://www.embopress.org/doi/full/10.15252/msb.202211517 | perturbation modeling | Compositional action/context discipline. | Do not use exact condition-key lookup; keep dose only for chemical screens. | audit |
| BYOL, https://arxiv.org/abs/2006.07733 | teacher/student | EMA target network stabilization. | EMA target encoders remain. | planned |
| VICReg, https://arxiv.org/abs/2105.04906 | anti-collapse | Variance/covariance losses. | Delta rank/variance diagnostics and losses. | planned |
| Barlow Twins, https://arxiv.org/abs/2103.03230 | anti-collapse | Redundancy reduction. | Bio/tech decorrelation diagnostics. | planned |
| DINO, https://arxiv.org/abs/2104.14294 | teacher/student | Momentum encoder and representation diagnostics. | EMA target and collapse diagnostics. | planned |
