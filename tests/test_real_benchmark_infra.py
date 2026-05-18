import json
import subprocess
import sys

import pandas as pd

from perturb_jepa.data.schema import add_condition_key
from perturb_jepa.data.splits import assert_group_split_integrity, grouped_hash_split
from scripts.evaluate_real_benchmark import main as evaluate_real_main


def test_download_public_data_dry_run_writes_manifest(tmp_path):
    manifest = tmp_path / "download_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_public_data.py",
            "--dry-run",
            "--dataset",
            "all-metadata",
            "--manifest-out",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert "bf-moa" in result.stdout
    assert {item["key"] for item in payload["datasets"]} >= {"bf-moa", "jump-cell-painting", "spatiallibd"}


def test_real_benchmark_new_clis_expose_help():
    scripts = [
        "scripts/build_rna_manifest.py",
        "scripts/build_image_manifest.py",
        "scripts/build_paired_manifest.py",
        "scripts/evaluate_real_benchmark.py",
        "scripts/run_biological_validation.py",
    ]
    for script in scripts:
        result = subprocess.run([sys.executable, script, "--help"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, script
        assert "usage:" in result.stdout


def test_condition_key_and_grouped_split_exclude_technical_metadata():
    frame = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugA", "drugB", "drugB"],
            "dose": ["1", "1", "1", "1"],
            "time": ["24", "24", "24", "24"],
            "cell_line": ["A", "A", "A", "A"],
            "batch": ["b1", "b2", "b1", "b2"],
            "plate": ["p1", "p2", "p1", "p2"],
            "well": ["A01", "A02", "B01", "B02"],
            "site": ["1", "2", "1", "2"],
        }
    )
    keyed = add_condition_key(frame)
    split = grouped_hash_split(keyed, ["condition_key"], fractions={"train": 0.5, "test": 0.5}, seed=0)

    assert keyed["condition_key"].tolist() == ["drugA|1|24|A", "drugA|1|24|A", "drugB|1|24|A", "drugB|1|24|A"]
    assert "b1" not in keyed.loc[0, "condition_key"]
    assert "p1" not in keyed.loc[0, "condition_key"]
    assert_group_split_integrity(split, ["condition_key"])


def test_evaluate_real_benchmark_dry_run_writes_unavailable_outputs(tmp_path):
    exit_code = evaluate_real_main(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--baselines-dir",
            str(tmp_path / "baselines"),
        ]
    )

    metrics = pd.read_csv(tmp_path / "evaluation" / "real_benchmark_metrics.csv")
    baselines = pd.read_csv(tmp_path / "baselines" / "real_benchmark_baselines.csv")
    assert exit_code == 0
    assert metrics.loc[0, "result_type"] == "not available"
    assert baselines.loc[0, "result_type"] == "not available"
