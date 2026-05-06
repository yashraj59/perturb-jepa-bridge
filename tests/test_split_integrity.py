import pandas as pd
import pytest

from perturb_jepa.data.splits import (
    heldout_batch_split,
    heldout_cell_line_split,
    heldout_dose_time_split,
    heldout_moa_split,
    heldout_perturbation_split,
    random_sample_split,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "perturbation": ["drugA", "drugA", "drugB", "drugB", "drugC", "drugC", "drugD", "drugD"],
            "dose": ["1uM", "1uM", "10uM", "10uM", "1uM", "1uM", "10uM", "10uM"],
            "time": ["24h", "24h", "48h", "48h", "24h", "24h", "48h", "48h"],
            "cell_line": ["U2OS", "U2OS", "A549", "A549", "U2OS", "U2OS", "A549", "A549"],
            "batch": ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"],
            "moa": ["m1", "m1", "m2", "m2", "m1", "m1", "m3", "m3"],
        }
    )


def _assert_no_group_overlap(split: pd.DataFrame, group_cols: list[str]) -> None:
    train_groups = set(
        split.loc[split["split"] == "train", group_cols].astype(str).agg("|".join, axis=1).tolist()
    )
    test_groups = set(split.loc[split["split"] == "test", group_cols].astype(str).agg("|".join, axis=1).tolist())
    assert train_groups.isdisjoint(test_groups)


def test_random_sample_split_assigns_requested_labels():
    split = random_sample_split(_metadata(), fractions={"train": 0.5, "val": 0.25, "test": 0.25}, seed=1)

    assert set(split["split"]) == {"train", "val", "test"}
    assert len(split) == 8


def test_heldout_splits_keep_groups_exclusive():
    frame = _metadata()

    _assert_no_group_overlap(heldout_batch_split(frame, heldout_batches=["b2"]), ["batch"])
    _assert_no_group_overlap(
        heldout_dose_time_split(frame, heldout_dose_times=[("10uM", "48h")]),
        ["dose", "time"],
    )
    _assert_no_group_overlap(
        heldout_perturbation_split(frame, heldout_perturbations=["drugD"]),
        ["perturbation"],
    )
    _assert_no_group_overlap(heldout_cell_line_split(frame, heldout_cell_lines=["A549"]), ["cell_line"])
    _assert_no_group_overlap(heldout_moa_split(frame, heldout_moas=["m3"]), ["moa"])


def test_heldout_split_rejects_unknown_groups_and_missing_moa_labels():
    with pytest.raises(ValueError, match="not present"):
        heldout_batch_split(_metadata(), heldout_batches=["missing"])

    with pytest.raises(ValueError, match="moa"):
        heldout_moa_split(_metadata().drop(columns=["moa"]))
