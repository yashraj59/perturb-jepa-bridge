from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.run_family_m_transport_baselines import _eval_control_splits, pair_records


def test_family_m_eval_split_yields_heldout_pairs():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_heldout_perturbation_lite", seed=2))
    records = pair_records(
        dataset,
        split="test_heldout_perturbation",
        control_splits=_eval_control_splits("test_heldout_perturbation"),
    )
    assert records
    assert {record["target_split"] for record in records} == {"test_heldout_perturbation"}
    assert {record["control_split"] for record in records} == {"train"}


def test_required_scripts_expose_eval_split_help():
    scripts = [
        "scripts/run_synthetic_lite_step0.py",
        "scripts/run_family_m_transport_baselines.py",
        "scripts/run_family_n_distillation.py",
        "scripts/run_family_o_count_likelihood.py",
        "scripts/evaluate_prefit_pls_readout.py",
        "scripts/train_bioaction_jepa.py",
        "scripts/evaluate_bioaction_jepa.py",
    ]
    for script in scripts:
        result = subprocess.run([sys.executable, script, "--help"], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
        assert "--eval-split" in result.stdout

