import numpy as np
import pandas as pd

from perturb_jepa.evaluation.reporting import grouped_metric_report


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
