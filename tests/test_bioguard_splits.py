import numpy as np
import pandas as pd
import pytest

from perturb_jepa.training.bioguard_splits import ActionGroupedResidualSplitConfig, ActionGroupedResidualSplitter


def _metadata():
    return pd.DataFrame(
        {
            "split": ["train"] * 12,
            "perturbation_id": np.repeat([1, 2, 3, 4], 3),
            "cell_line_id": [0, 1, 2] * 4,
            "batch_id": [0, 0, 1] * 4,
        }
    )


def test_no_eval_rows_enter_fit_or_calibration():
    frame = _metadata()
    frame.loc[0, "split"] = "test"
    splitter = ActionGroupedResidualSplitter(ActionGroupedResidualSplitConfig(n_folds=2, seed=0))

    with pytest.raises(ValueError, match="non-train rows"):
        splitter.split(frame)


def test_leave_action_out_prevents_action_group_leakage():
    splitter = ActionGroupedResidualSplitter(ActionGroupedResidualSplitConfig(n_folds=4, seed=0, leave_action_out=True))
    folds = splitter.split(_metadata())

    assert len(folds) == 4
    for fold in folds:
        assert not set(fold.fit_actions).intersection(fold.calibration_actions)
        assert fold.fit_indices.size > 0
        assert fold.calibration_indices.size > 0


def test_fold_construction_is_deterministic():
    config = ActionGroupedResidualSplitConfig(n_folds=3, seed=17)
    left = ActionGroupedResidualSplitter(config).split(_metadata())
    right = ActionGroupedResidualSplitter(config).split(_metadata())

    assert [tuple(fold.calibration_actions) for fold in left] == [tuple(fold.calibration_actions) for fold in right]
    assert [fold.calibration_indices.tolist() for fold in left] == [fold.calibration_indices.tolist() for fold in right]


def test_each_fold_reports_action_coverage():
    report = ActionGroupedResidualSplitter(ActionGroupedResidualSplitConfig(n_folds=2, seed=1)).report(_metadata())

    assert {"fold_id", "fit_actions", "calibration_actions", "fit_rows", "calibration_rows"}.issubset(report.columns)
    assert report["fit_actions"].str.len().min() > 0
    assert report["calibration_actions"].str.len().min() > 0


def test_tiny_data_refuses_random_row_fallback_for_action_grouped_split():
    frame = pd.DataFrame(
        {
            "split": ["train"] * 4,
            "perturbation_id": [1, 1, 1, 1],
            "cell_line_id": [0, 0, 1, 1],
            "batch_id": [0, 1, 0, 1],
        }
    )
    splitter = ActionGroupedResidualSplitter(ActionGroupedResidualSplitConfig(n_folds=4, seed=0))

    with pytest.raises(ValueError, match="at least two action groups"):
        splitter.split(frame)


def test_too_many_folds_documents_grouped_fallback_without_row_split():
    folds = ActionGroupedResidualSplitter(ActionGroupedResidualSplitConfig(n_folds=10, seed=0)).split(_metadata())

    assert len(folds) == 4
    assert all("requested_10_folds" in fold.fallback_reason for fold in folds)
    for fold in folds:
        assert not set(fold.fit_actions).intersection(fold.calibration_actions)
