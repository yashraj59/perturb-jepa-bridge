import numpy as np
import pandas as pd

from perturb_jepa.data.conditions import build_condition_bags
from perturb_jepa.data.sampling import (
    MISSING_NEGATIVE_INDEX,
    add_stratified_hard_negative_indices,
    sample_stratified_hard_negatives,
    stratified_hard_negative_candidates,
)
from perturb_jepa.data.schema import add_condition_key, add_hierarchical_condition_keys


def test_hierarchical_condition_keys_append_expected_levels():
    frame = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugA", "drugB"],
            "perturbation_type": ["compound", "compound", "compound"],
            "dose": ["10uM", "10uM", "5uM"],
            "time": ["48h", "72h", "48h"],
            "cell_line": ["U2OS", "U2OS", "A549"],
        }
    )

    keyed = add_hierarchical_condition_keys(frame)

    assert keyed["condition_key_coarse"].tolist() == ["drugA", "drugA", "drugB"]
    assert keyed["condition_key_medium"].tolist() == ["drugA|10uM|48h", "drugA|10uM|72h", "drugB|5uM|48h"]
    assert keyed["condition_key_fine"].tolist() == [
        "drugA|compound|10uM|48h|U2OS",
        "drugA|compound|10uM|72h|U2OS",
        "drugB|compound|5uM|48h|A549",
    ]
    assert keyed["condition_key"].tolist() == keyed["condition_key_fine"].tolist()

    custom = add_condition_key(frame, columns=("perturbation", "dose"), output_col="perturbation_dose_key")
    assert custom["perturbation_dose_key"].tolist() == ["drugA|10uM", "drugA|10uM", "drugB|5uM"]

    bags = build_condition_bags(frame, key_col="condition_key_coarse")
    assert bags.keys == ["drugA", "drugB"]
    np.testing.assert_array_equal(bags.counts, np.array([2, 1]))


def _negative_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "perturbation": ["drugA", "drugB", "drugC", "drugA", "drugD"],
            "cell_line": ["U2OS", "U2OS", "U2OS", "U2OS", "A549"],
            "batch": ["batch1", "batch1", "batch1", "batch1", "batch1"],
            "plate": ["plate1", "plate1", "plate2", "plate1", "plate1"],
            "time": ["24h", "24h", "24h", "24h", "24h"],
        }
    )


def test_stratified_hard_negative_candidates_prefer_exact_nuisance_match():
    frame = _negative_frame()

    exact = stratified_hard_negative_candidates(frame, 0)
    np.testing.assert_array_equal(exact.indices, np.array([1]))
    assert exact.matched_nuisance_columns == ("cell_line", "batch", "plate", "time")
    assert not exact.used_fallback

    no_fallback = stratified_hard_negative_candidates(frame, 2, fallback_to_any=False)
    assert no_fallback.indices.size == 0
    assert no_fallback.matched_nuisance_columns == ("cell_line", "batch", "plate", "time")
    assert not no_fallback.used_fallback

    fallback = stratified_hard_negative_candidates(frame, 2)
    assert set(fallback.indices.tolist()) == {0, 1, 3, 4}
    assert fallback.matched_nuisance_columns == ()
    assert fallback.used_fallback


def test_stratified_hard_negative_candidates_ignore_missing_nuisance_columns():
    frame = _negative_frame().drop(columns="plate")

    candidates = stratified_hard_negative_candidates(frame, 0)

    assert set(candidates.indices.tolist()) == {1, 2}
    assert candidates.matched_nuisance_columns == ("cell_line", "batch", "time")
    assert not candidates.used_fallback


def test_sample_stratified_hard_negatives_pads_and_reports_fallbacks():
    frame = _negative_frame()

    samples = sample_stratified_hard_negatives(frame, n_negatives=2, replace=False, seed=7)

    assert samples.indices.shape == (len(frame), 2)
    assert samples.indices[0].tolist() == [1, MISSING_NEGATIVE_INDEX]
    assert samples.candidate_counts[0] == 1
    assert samples.used_fallback[2]
    assert set(samples.indices[2][samples.indices[2] != MISSING_NEGATIVE_INDEX].tolist()).issubset({0, 1, 3, 4})

    annotated = add_stratified_hard_negative_indices(frame, n_negatives=1, seed=11)
    assert "hard_negative_index" in annotated.columns
    assert "hard_negative_index_candidate_count" in annotated.columns
    assert "hard_negative_index_used_fallback" in annotated.columns


def test_sample_stratified_hard_negatives_returns_sentinel_when_no_different_perturbation():
    frame = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugA"],
            "cell_line": ["U2OS", "U2OS"],
            "batch": ["batch1", "batch1"],
            "plate": ["plate1", "plate1"],
            "time": ["24h", "24h"],
        }
    )

    samples = sample_stratified_hard_negatives(frame, n_negatives=1, seed=3)

    np.testing.assert_array_equal(samples.indices, np.array([[MISSING_NEGATIVE_INDEX], [MISSING_NEGATIVE_INDEX]]))
    np.testing.assert_array_equal(samples.candidate_counts, np.array([0, 0]))
    assert not samples.used_fallback.any()
