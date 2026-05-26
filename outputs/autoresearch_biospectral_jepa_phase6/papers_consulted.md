# Papers Consulted

| Source | Mechanism extracted | Phase 6 mapping |
| --- | --- | --- |
| V-JEPA 2, https://arxiv.org/abs/2506.09985 | action-conditioned latent prediction with stop-gradient targets | preserve real JEPA identity for later BSJ006 wrapper |
| Dynamic Mode Decomposition with Control, https://epubs.siam.org/doi/10.1137/15M1013857 | separate state dynamics from control effects and audit low-order approximations | compare full ridge, reduced-rank ridge, and BOJ002 control-affine form |
| Koopman / EDMD-style operator learning, https://faculty.washington.edu/kutz/page1/page13/ | finite-rank latent operators need validation | rank bottleneck audit before architecture search |
| LoRA / adaptive low-rank, https://arxiv.org/abs/2505.22694 | fixed rank may be too restrictive; use adaptive rank experts | rank ladder and spectral residual design if reopened |
| CellOT, https://www.nature.com/articles/s41592-023-01969-x | perturbation responses are distributional maps | defer population transport until floor-preserving operator passes |
| CPA, https://www.embopress.org/doi/full/10.15252/msb.202211517 | compositional perturbation/context factors | keep action descriptor discipline; Norman dose is guide presence only |
| GEARS, https://www.nature.com/articles/s41587-023-01905-6 | gene/gene-pair action priors | optional future action features, not a replacement for floor preservation |
| GPerturb, https://www.nature.com/articles/s41467-025-61165-7 | uncertainty/kernel perturbation effects | optional kernel residual only after spectral residual evidence |
