# Repository Guidelines

## Project Structure & Module Organization

`perturb_jepa/` contains the Python package. Core areas are `data/` for schemas,
sampling, images, and scRNA ingestion; `models/` for encoders, bridge modules,
EMA, aggregators, and adversaries; `training/` for objectives, trainers,
checkpoints, and synthetic data; `evaluation/` and `baselines/` for metrics and
comparison methods. CLI entrypoints live in `scripts/`. YAML defaults are in
`configs/`, documentation assets are in `docs/`, notebooks are in `notebooks/`,
and regression tests are in `tests/`.

## Build, Test, and Development Commands

Install editable development dependencies:

```bash
python -m pip install -e ".[data,dev]"
```

Run the full test suite:

```bash
pytest
```

Run a quick synthetic training check:

```bash
python scripts/train_smoke.py --steps 2
```

Exercise config-driven synthetic training with checkpoint output:

```bash
python scripts/train_synthetic.py --steps 10 --checkpoint-out checkpoints/synthetic.pt
```

Use console scripts after install, for example `perturb-jepa-smoke-train --steps 2`.

## Coding Style & Naming Conventions

Use Python 3.10+ with 4-space indentation, type hints where they clarify public
interfaces, and small functions that match the existing package layout. Keep
module names lowercase with underscores, class names in `PascalCase`, and
functions, variables, and config keys in `snake_case`. Prefer explicit metadata
fields and structured `pandas`/PyTorch operations over ad hoc string handling.
No formatter is configured in `pyproject.toml`; keep edits consistent with nearby
code.

## Testing Guidelines

Tests use `pytest` and are discovered from `tests/` via `pyproject.toml`.
Name new tests `test_<behavior>.py` and test functions `test_<expected_result>`.
Add or update focused tests for schema changes, leakage checks, loss behavior,
training checkpoints, and evaluation metrics. Run `pytest` before opening a PR;
for narrow changes, run the relevant file first, such as
`pytest tests/test_metrics.py`.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects such as `Fix leakage-safe bridge
real-data path` and `Add real-data training and evaluation paths`. Follow that
style: start with a verb, keep the subject specific, and avoid noisy prefixes.

Pull requests should include a concise purpose statement, key implementation
notes, test commands and results, and any config or data-interface changes. Link
issues when applicable. For notebook, docs image, or evaluation output changes,
include enough context for reviewers to reproduce the artifact.

## Security & Configuration Tips

Do not commit large raw datasets, generated checkpoints, or private paths. Keep
technical metadata such as batch, plate, well, site, lane, and library ID out of
biological condition keys and encoder inputs unless a test explicitly covers the
leakage behavior.
