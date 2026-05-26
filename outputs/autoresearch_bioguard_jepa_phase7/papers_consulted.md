# Papers Consulted

| Source | Design idea used in Phase 7 |
| --- | --- |
| V-JEPA / V-JEPA 2 | Latent action-conditioned future-state prediction; mapped to control `z_bio` + perturbation action -> perturbed teacher `z_bio`. |
| Safe policy improvement literature | Learned residuals are not deployed when evidence suggests underperforming a protected baseline. |
| Conformal risk control / split calibration | Use train-only calibration to decide residual scale/gate, defaulting to zero residual. |
| Double/debiased ML and cross-fitting | Fit flexible residuals on one fold and evaluate them on action-held-out calibration folds. |
| CellOT / GEARS / CPA | Treat perturbations as biological actions and prioritize held-out perturbation generalization over train residual fit. |
