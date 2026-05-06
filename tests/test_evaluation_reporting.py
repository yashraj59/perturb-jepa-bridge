import numpy as np
import pandas as pd

from perturb_jepa.evaluation.reporting import (
    cell_line_transfer_report,
    dose_response_monotonicity_report,
    grouped_metric_report,
    heldout_perturbation_prediction_report,
)


def test_grouped_metric_report_includes_overall_and_per_group_rows():
    control = np.array([1.0, 1.0])
    observed = np.array(
        [
            [2.0, 1.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ]
    )
    predicted = np.array(
        [
            [2.0, 1.0],
            [3.0, 1.0],
            [1.0, 2.5],
        ]
    )
    metadata = pd.DataFrame({"perturbation": ["drugA", "drugA", "drugB"]})

    report = grouped_metric_report(
        predicted,
        observed,
        control,
        metadata,
        groupby="perturbation",
        topk=1,
    )

    assert report["group"].tolist() == ["overall", "drugA", "drugB"]
    assert report["n_samples"].tolist() == [3, 2, 1]
    assert report.loc[report["group"] == "drugA", "delta_mae"].item() == 0.0
    assert report.loc[report["group"] == "drugB", "delta_mae"].item() == 0.25
    assert "top1_jaccard_delta" in report.columns


def test_grouped_metric_report_accepts_sample_specific_controls():
    observed = np.array([[2.0, 2.0], [5.0, 5.0]])
    predicted = np.array([[2.0, 3.0], [4.0, 5.0]])
    control = np.array([[1.0, 1.0], [4.0, 4.0]])
    metadata = pd.DataFrame({"batch": ["b1", "b2"]})

    report = grouped_metric_report(
        predicted,
        observed,
        control,
        metadata,
        groupby="batch",
        topk=1,
        include_overall=False,
    )

    assert report["group"].tolist() == ["b1", "b2"]
    np.testing.assert_allclose(report["delta_mae"].to_numpy(), np.array([0.5, 0.5]))


def test_grouped_metric_report_includes_topk_de_recovery():
    control = np.zeros(3)
    observed = np.array([[0.0, 3.0, 1.0]])
    predicted = np.array([[0.0, 2.0, 1.0]])

    report = grouped_metric_report(predicted, observed, control, topk=1)

    assert report.loc[0, "top1_de_recovery"] == 1.0


def test_heldout_perturbation_prediction_report_filters_unseen_perturbations():
    control = np.zeros(2)
    observed = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    predicted = observed.copy()
    train_metadata = pd.DataFrame({"perturbation": ["drugA"]})
    eval_metadata = pd.DataFrame({"perturbation": ["drugA", "drugB", "drugC"]})

    report = heldout_perturbation_prediction_report(
        predicted,
        observed,
        control,
        eval_metadata,
        train_metadata,
        topk=1,
    )

    assert report["group"].tolist() == ["overall", "drugB", "drugC"]
    assert report["n_samples"].tolist() == [2, 1, 1]
    assert report["delta_mae"].tolist() == [0.0, 0.0, 0.0]


def test_cell_line_transfer_report_groups_seen_and_heldout_cell_lines():
    control = np.zeros(2)
    observed = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
    predicted = observed.copy()
    train_metadata = pd.DataFrame({"cell_line": ["U2OS"]})
    eval_metadata = pd.DataFrame({"cell_line": ["U2OS", "A549", "A549"]})

    report = cell_line_transfer_report(
        predicted,
        observed,
        control,
        eval_metadata,
        train_metadata,
        topk=1,
    )

    assert set(report["cell_line_transfer"].dropna()) == {"seen", "held_out"}
    heldout = report[report["group"] == "held_out|A549"].iloc[0]
    assert heldout["n_samples"] == 2
    assert heldout["delta_mae"] == 0.0


def test_dose_response_monotonicity_report_uses_synthetic_metadata_groups():
    control = np.zeros(2)
    predicted = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
        ]
    )
    metadata = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugA", "drugA", "drugB", "drugB"],
            "cell_line": ["U2OS", "U2OS", "U2OS", "A549", "A549"],
            "dose": ["1uM", "2uM", "3uM", "1uM", "2uM"],
        }
    )

    report = dose_response_monotonicity_report(predicted, control, metadata)

    assert report.loc[report["group"] == "drugA|U2OS", "dose_response_monotonicity"].item() == 1.0
    assert report.loc[report["group"] == "drugB|A549", "dose_response_monotonicity"].item() == 0.0
