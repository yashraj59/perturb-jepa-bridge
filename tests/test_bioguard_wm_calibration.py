import numpy as np
import pandas as pd

from perturb_jepa.training.bioguard_wm_calibration import make_action_grouped_folds, select_residual_scale_crossfit, select_residual_scale_small_cap_continuous


def test_calibration_defaults_to_floor_when_recall_lcb_negative():
    rows = []
    for fold in range(4):
        rows.append({"scale": 0.0, "floor_gap_transition": 0.0, "floor_gap_recall": 0.0, "floor_gap_delta_cosine": 0.0, "action_negative_gap": 0.0})
        rows.append({"scale": 0.1, "floor_gap_transition": 0.0002, "floor_gap_recall": -0.03 if fold == 0 else 0.0, "floor_gap_delta_cosine": 0.001, "action_negative_gap": 0.001})

    selected = select_residual_scale_crossfit(rows, [0.0, 0.1])

    assert selected.residual_scale == 0.0
    assert selected.status == "CALIBRATION_DEFAULT_TO_FLOOR"


def test_calibration_selects_residual_when_all_gates_safe():
    rows = []
    for _ in range(4):
        rows.append({"scale": 0.0, "floor_gap_transition": 0.0, "floor_gap_recall": 0.0, "floor_gap_delta_cosine": 0.0, "action_negative_gap": 0.0})
        rows.append({"scale": 0.1, "floor_gap_transition": 0.001, "floor_gap_recall": 0.02, "floor_gap_delta_cosine": 0.01, "action_negative_gap": 0.02})

    selected = select_residual_scale_crossfit(rows, [0.0, 0.1])

    assert selected.residual_scale > 0.0
    assert selected.status == "CALIBRATION_SELECTED_NONZERO_RESIDUAL"


def test_action_grouped_folds_keep_actions_disjoint():
    records = pd.DataFrame({"action_id": np.repeat(["a", "b", "c", "d"], 2)})
    folds = make_action_grouped_folds(records, n_folds=3, action_key="action_id", seed=0)

    assert len(folds) == 3
    for fold in folds:
        assert not set(fold["fit_actions"]).intersection(fold["calibration_actions"])


def test_small_cap_continuous_selector_uses_tiny_safe_scale():
    rows = []
    for _ in range(4):
        rows.append({"scale": 0.0, "floor_gap_transition": 0.0, "floor_gap_recall": 0.0, "floor_gap_delta_cosine": 0.0})
        rows.append({"scale": 0.025, "floor_gap_transition": 0.0002, "floor_gap_recall": 0.0, "floor_gap_delta_cosine": 0.001})
        rows.append({"scale": 0.05, "floor_gap_transition": 0.0004, "floor_gap_recall": 0.0, "floor_gap_delta_cosine": 0.002})
        rows.append({"scale": 0.1, "floor_gap_transition": 0.0020, "floor_gap_recall": -0.1, "floor_gap_delta_cosine": 0.004})

    selected = select_residual_scale_small_cap_continuous(rows, scale_grid=[0.0, 0.025, 0.05, 0.1])

    assert selected.residual_scale == 0.05
    assert selected.status == "CALIBRATION_SELECTED_SMALL_CAP_CONTINUOUS_RESIDUAL"


def test_small_cap_continuous_selector_defaults_on_recall_drop():
    rows = []
    for fold in range(4):
        rows.append({"scale": 0.0, "floor_gap_transition": 0.0, "floor_gap_recall": 0.0, "floor_gap_delta_cosine": 0.0})
        rows.append({"scale": 0.05, "floor_gap_transition": 0.0004, "floor_gap_recall": -0.02 if fold == 0 else 0.0, "floor_gap_delta_cosine": 0.002})

    selected = select_residual_scale_small_cap_continuous(rows, scale_grid=[0.0, 0.05])

    assert selected.residual_scale == 0.0
    assert selected.status == "CALIBRATION_DEFAULT_TO_FLOOR"
