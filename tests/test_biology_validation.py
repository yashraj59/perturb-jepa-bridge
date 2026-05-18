import json

import numpy as np
import pandas as pd
import pytest

from perturb_jepa.evaluation.biology import (
    de_direction_accuracy,
    log_fold_change,
    pseudobulk_aggregate,
    read_gmt,
    topk_de_overlap,
)
from scripts.run_biological_validation import main as biology_main


def test_pseudobulk_logfc_and_de_metrics():
    expression = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [5.0, 1.0, 2.0],
            [7.0, 1.0, 4.0],
        ]
    )
    metadata = pd.DataFrame({"perturbation": ["control", "control", "drugA", "drugA"]})

    bulk = pseudobulk_aggregate(expression, metadata, groupby="perturbation", gene_names=["g1", "g2", "g3"])
    control = bulk.loc[bulk["perturbation"].eq("control"), ["g1", "g2", "g3"]].to_numpy()[0]
    treated = bulk.loc[bulk["perturbation"].eq("drugA"), ["g1", "g2", "g3"]].to_numpy()[0]
    observed_lfc = log_fold_change(treated, control)
    predicted_lfc = np.array([observed_lfc[0], -0.5, observed_lfc[2]])

    assert bulk.loc[bulk["perturbation"].eq("drugA"), "n_cells"].item() == 2
    assert observed_lfc[0] == pytest.approx(np.log2(6.0 / 2.0), rel=1e-5)
    assert topk_de_overlap(predicted_lfc, observed_lfc, k=2) == 1.0
    assert de_direction_accuracy(predicted_lfc, observed_lfc, k=2) == 1.0


def test_read_gmt_skips_short_rows(tmp_path):
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("pathway_a\tdesc\tg1\tg2\nshort\tdesc\n", encoding="utf-8")

    assert read_gmt(gmt) == {"pathway_a": ["g1", "g2"]}


def test_biological_validation_cli_writes_artifacts(tmp_path):
    expression = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.1, 1.0],
            [4.0, 1.0, 3.0],
            [4.2, 1.1, 2.8],
        ],
        dtype=float,
    )
    predicted = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.0],
            [3.8, 1.1, 2.6],
            [4.0, 1.2, 2.5],
        ],
        dtype=float,
    )
    metadata = pd.DataFrame(
        {
            "perturbation": ["DMSO", "DMSO", "drugA", "drugA"],
            "perturbation_type": ["control", "control", "compound", "compound"],
            "dose": [0, 0, 1, 1],
            "time": [24, 24, 24, 24],
            "cell_line": ["A", "A", "A", "A"],
            "moa": ["control", "control", "kinase", "kinase"],
        }
    )
    paths = {
        "expression": tmp_path / "expression.npy",
        "predicted": tmp_path / "predicted.npy",
        "metadata": tmp_path / "metadata.csv",
        "genes": tmp_path / "genes.txt",
        "gmt": tmp_path / "sets.gmt",
        "output": tmp_path / "out",
    }
    np.save(paths["expression"], expression)
    np.save(paths["predicted"], predicted)
    metadata.to_csv(paths["metadata"], index=False)
    paths["genes"].write_text("g1\ng2\ng3\n", encoding="utf-8")
    paths["gmt"].write_text("response\tdesc\tg1\tg3\n", encoding="utf-8")

    exit_code = biology_main(
        [
            "--expression",
            str(paths["expression"]),
            "--metadata",
            str(paths["metadata"]),
            "--predicted-expression",
            str(paths["predicted"]),
            "--genes",
            str(paths["genes"]),
            "--gmt",
            str(paths["gmt"]),
            "--topk",
            "2",
            "--output-dir",
            str(paths["output"]),
        ]
    )

    metrics = pd.read_csv(paths["output"] / "metrics.csv")
    pathway = pd.read_csv(paths["output"] / "pathway_scores.csv")
    summary = json.loads((paths["output"] / "validation_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metrics.loc[0, "topk_de_overlap"] == 1.0
    assert pathway.loc[0, "pathway"] == "response"
    assert (paths["output"] / "figures" / "topk_de_overlap.png").exists()
    assert "mean_topk_de_overlap" in summary["metrics"]
