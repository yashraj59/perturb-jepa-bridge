import pandas as pd

from perturb_jepa.data.schema import add_condition_key, normalize_image_manifest
from perturb_jepa.data.splits import assert_group_split_integrity, grouped_hash_split


def test_condition_keys_and_grouped_split_have_no_leakage():
    frame = pd.DataFrame(
        {
            "image_path": ["a.tif", "b.tif", "c.tif", "d.tif"],
            "plate": ["p1", "p1", "p2", "p2"],
            "well": ["A01", "A01", "B01", "B01"],
            "site": ["1", "2", "1", "2"],
            "channel_or_z": ["BF1", "BF2", "BF1", "BF2"],
            "perturbation": ["drugA", "drugA", "drugB", "drugB"],
            "compound": ["drugA", "drugA", "drugB", "drugB"],
            "moa": ["m1", "m1", "m2", "m2"],
            "target_gene": ["", "", "", ""],
            "dose": ["10uM", "10uM", "10uM", "10uM"],
            "time": ["48h", "48h", "48h", "48h"],
            "cell_line": ["U2OS", "U2OS", "U2OS", "U2OS"],
            "batch": ["p1", "p1", "p2", "p2"],
        }
    )
    normalized = normalize_image_manifest(frame)
    assert "condition_key" in normalized.columns
    split = grouped_hash_split(normalized, ["condition_key", "batch"])
    assert_group_split_integrity(split, ["condition_key", "batch"])


def test_add_condition_key_uses_declared_columns():
    frame = pd.DataFrame(
        {
            "perturbation": ["x"],
            "perturbation_type": ["compound"],
            "dose": ["1uM"],
            "time": ["24h"],
            "cell_line": ["A549"],
        }
    )
    keyed = add_condition_key(frame)
    assert keyed.loc[0, "condition_key"] == "x|compound|1uM|24h|A549"
