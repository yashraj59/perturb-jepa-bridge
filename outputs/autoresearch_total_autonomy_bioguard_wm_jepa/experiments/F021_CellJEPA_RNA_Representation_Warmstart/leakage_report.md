# F021 Cell-JEPA Leakage Report

- Local Cell-JEPA paper consulted: `papers/2602.02093v1.pdf`
- Train fit rows: synthetic train split only.
- Eval target rows: scoring only.
- Model inputs: gene ids and expression values only.
- `condition_key` and `biological_key`: not model inputs.
- Perturbation ID one-hot: not used in RNA warmstart input.
- Held-out target means: not used for warmstart training.
- Reconstruction role: auxiliary only; `w_jepa` is required to exceed `w_rec`.
- Transition status: diagnostic rerun of train-only action-ridge floor on frozen z_bio, not model promotion.
