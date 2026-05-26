# Papers Consulted

- Assran et al., I-JEPA, arXiv:2301.08243: representation-space JEPA prediction.
- Bardes et al., VICReg, arXiv:2105.04906: variance/covariance anti-collapse.
- Zbontar et al., Barlow Twins, arXiv:2103.03230: cross-correlation redundancy reduction.
- Lotfollahi et al., scGen/CPA: perturbation-response latent modeling.
- Roohani et al., GEARS; Cui et al., scGPT; Systema benchmark: perturbation prediction and simple baseline discipline.
- Chen and He, SimSiam, arXiv:2011.10566: collapse can exist in Siamese/self-prediction systems; stop-gradient/predictor geometry must be audited rather than assumed safe.
- Grill et al., BYOL, NeurIPS 2020: teacher/student SSL can work without negatives, but target-network and normalization details are core mechanism, not implementation trivia.
- He et al., Masked Autoencoders, arXiv:2111.06377: masked reconstruction learns token/patch representations; this does not guarantee a useful global CLS condition embedding.
- Split-geometry consequence for this repo: prioritize encoder/projection/dropout diagnostics, explicit variance/covariance gates, and simple oracle/raw baselines before reopening architecture search.
- Hotelling, "Relations Between Two Sets of Variates", Biometrika 1936: canonical correlation motivates fitting the shared space from cross-view covariance before adding nonlinear JEPA machinery.
- Wold/NIPALS partial least squares lineage: PLS is an appropriate low-rank cross-covariance tool when the train condition count is small relative to genes and pixels.
- Izenman, "Reduced-rank regression for the multivariate linear model", Journal of Multivariate Analysis 1975: low-rank linear maps are a defensible baseline for multivariate cross-modal prediction.
- Schoenemann, "A generalized solution of the orthogonal Procrustes problem", Psychometrika 1966: SVD/Procrustes alignment supports closed-form shared-space alignment when learned rotations are unstable.
- Repair consequence: use a closed-form, train-split-only, low-rank PLS whitening readout as the protected bridge initializer; then add JEPA training around it only after no-collapse/no-leakage gates hold.

## Web Literature Check: Distillation And Nonlinear Multiview Heads

- Andrew et al., "Deep Canonical Correlation Analysis", ICML 2013, https://proceedings.mlr.press/v28/andrew13.html: nonlinear multiview heads can be trained to maximize regularized correlation, but this strengthens the need for explicit regularization when data are small.
- Bardes et al., "VICReg", arXiv:2105.04906, https://arxiv.org/abs/2105.04906: variance and covariance terms address collapse, but our raw MLP distillation failure shows noncollapse alone does not guarantee retrieval alignment.
- Zbontar et al., "Barlow Twins", arXiv:2103.03230, https://arxiv.org/abs/2103.03230: cross-correlation identity objectives motivate future residual-adapter gates, but the current low-rank PLS clone is safer than random high-capacity heads.
- Grill et al., "BYOL", arXiv:2006.07733, https://arxiv.org/abs/2006.07733: teacher-student self-supervision depends on target-network and normalization details; in this repo the teacher geometry must be frozen or exactly cloned before SGD modifications.
- Caron et al., "DINO", arXiv:2104.14294, https://arxiv.org/abs/2104.14294: self-distillation can work without labels, but collapse-control mechanisms are central; random student heads should not be trusted without independent gates.
- van den Oord et al., "Contrastive Predictive Coding", arXiv:1807.03748, https://arxiv.org/abs/1807.03748: predictive latent objectives align with JEPA-style representation learning, but retrieval failures here indicate the condition-level shared geometry must be protected separately from token/patch prediction.

Distillation consequence: the next trainable mechanism should be a zero-initialized residual adapter around the exact linear PLS clone, not another randomly initialized MLP head.

## Sparse Perturbation Residual Rationale

- PerturbedVAE, "What Makes a Representation Good for Single-Cell Perturbation Prediction?", arXiv:2605.19343, https://arxiv.org/abs/2605.19343: motivates separating perturbation-invariant expression from sparse perturbation-specific signal.
- PertAdapt, "Unlocking Single-Cell Foundation Models for Genetic Perturbation Prediction via Condition-Sensitive Adaptation", https://sciety.org/articles/activity/10.1101/2025.11.21.689655: motivates adaptive loss weighting for perturbation-sensitive genes/top-DE genes.
- Systema, "A Framework for Evaluating Genetic Perturbation Response Prediction Beyond Systematic Variation", https://brbiclab.epfl.ch/projects/systema: motivates perturbed-mean and matching-mean baselines and delta/top-DE/program-focused evaluation.
- Conditional Monge Gap, "Towards generalizable single-cell perturbation modeling via the Conditional Monge Gap", arXiv:2504.08328, https://arxiv.org/abs/2504.08328: retained as the conditional transport fallback after sparse residual failures.
- TxPert, "using multiple knowledge graphs for prediction of transcriptomic perturbation effects", Nature Biotechnology, https://www.nature.com/articles/s41587-026-03113-4: graph-prior direction deferred to real-data/OOD work; not used in the synthetic-only run.

## Family M Transport Rationale

- CellOT, "Learning single-cell perturbation responses using neural optimal transport", Nature Methods, https://www.nature.com/articles/s41592-023-01969-x: motivates modeling perturbation response as a map from control to perturbed cell-state distributions.
- CINEMA-OT, "Causal identification of single-cell experimental perturbation effects with CINEMA-OT", Nature Methods, https://www.nature.com/articles/s41592-023-02040-5: motivates no-confounder/counterfactual matching and supports explicitly excluding technical batch from the matching key/features.
- scDRP, "Single-cell disentangled representations for perturbation modeling and treatment effect estimation", PubMed record https://pubmed.ncbi.nlm.nih.gov/41394575/: motivates disentanglement plus conditional optimal transport for individualized treatment effects; used only as rationale, not as an imported method or resource.
- Conditional Monge Gap, arXiv:2504.08328, https://arxiv.org/abs/2504.08328: motivates conditional transport across perturbation context such as treatment, dose, and cell type.
- Systema, https://brbiclab.epfl.ch/projects/systema: remains the baseline-discipline anchor; direct perturbed-mean and residualized matching baselines must be beaten before neural progress claims.

Family M consequence: direct no-batch matched perturbed mean is the strongest exact-condition synthetic baseline, while kNN and Sinkhorn residual transport did not justify optional neural transport on seed 2.
