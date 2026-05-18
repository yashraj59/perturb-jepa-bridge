import numpy as np
import pandas as pd

from perturb_jepa.data.condition_bags import (
    ImageConditionBagDataset,
    PairedConditionBagDataset,
    RNAConditionBagDataset,
)


def _rna_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["rna0", "rna1", "rna2", "rna3"],
            "perturbation": ["drugA", "drugA", "drugA", "drugB"],
            "dose": ["10uM", "10uM", "10uM", "5uM"],
            "time": ["48h", "48h", "48h", "24h"],
            "cell_line": ["U2OS", "U2OS", "U2OS", "A549"],
            "batch": ["rna_b1", "rna_b2", "rna_b3", "rna_b4"],
            "sequencing_lane": ["L1", "L2", "L3", "L4"],
        }
    )


def _image_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["img0", "img1", "img2"],
            "perturbation": ["drugA", "drugA", "drugB"],
            "dose": ["10uM", "10uM", "5uM"],
            "time": ["48h", "48h", "24h"],
            "cell_line": ["U2OS", "U2OS", "A549"],
            "plate": ["plate1", "plate2", "plate3"],
            "well": ["A01", "A02", "B01"],
            "site": ["1", "2", "1"],
        }
    )


def test_paired_condition_bag_matches_biology_not_sample_identity():
    rna = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    images = np.arange(3 * 1 * 2 * 2, dtype=np.float32).reshape(3, 1, 2, 2)
    rna_dataset = RNAConditionBagDataset(
        rna,
        _rna_metadata(),
        rna_bag_size=2,
        min_rna_bag_size=2,
        split="val",
    )
    image_dataset = ImageConditionBagDataset(
        images,
        _image_metadata(),
        image_bag_size=1,
        min_image_bag_size=1,
        split="val",
    )

    paired = PairedConditionBagDataset(rna_dataset, image_dataset)
    item = paired[0]

    assert item["bio_key"] == ("drugA", "10uM", "48h", "U2OS")
    assert item["condition_id"] == "drugA|10uM|48h|U2OS"
    assert item["rna"]["x"].shape == (2, 3)
    assert item["image"]["x"].shape == (1, 1, 2, 2)
    assert item["rna"]["sample_ids"] == ["rna0", "rna1"]
    assert item["image"]["sample_ids"] == ["img0"]
    assert item["rna"]["sample_ids"] != item["image"]["sample_ids"]
    assert item["rna"]["tech"][0]["batch"] == "rna_b1"
    assert item["image"]["tech"][0]["plate"] == "plate1"
    assert item["rna"]["cell_meta"][0]["sample_id"] == "rna0"


def test_train_bags_randomly_subsample_and_eval_bags_are_deterministic():
    rna = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    metadata = pd.DataFrame(
        {
            "sample_id": [f"rna{i}" for i in range(5)],
            "perturbation": ["drugA"] * 5,
            "dose": ["10uM"] * 5,
            "time": ["48h"] * 5,
            "cell_line": ["U2OS"] * 5,
        }
    )
    train = RNAConditionBagDataset(rna, metadata, rna_bag_size=2, split="train", seed=7)
    val = RNAConditionBagDataset(rna, metadata, rna_bag_size=2, split="val", seed=7)

    first_train = train[0]["sample_ids"]
    second_train = train[0]["sample_ids"]
    first_val = val[0]["sample_ids"]
    second_val = val[0]["sample_ids"]

    assert first_train != second_train
    assert first_val == ["rna0", "rna1"]
    assert second_val == ["rna0", "rna1"]


def test_train_bags_can_balance_samples_across_batches():
    rna = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    metadata = pd.DataFrame(
        {
            "sample_id": [f"rna{i}" for i in range(6)],
            "perturbation": ["drugA"] * 6,
            "dose": ["10uM"] * 6,
            "time": ["48h"] * 6,
            "cell_line": ["U2OS"] * 6,
            "batch": ["b1", "b1", "b2", "b2", "b3", "b3"],
        }
    )
    train = RNAConditionBagDataset(
        rna,
        metadata,
        rna_bag_size=3,
        split="train",
        seed=7,
        balanced_sample_col="batch",
    )

    item = train[0]
    selected_batches = {row["batch"] for row in item["cell_meta"]}

    assert selected_batches == {"b1", "b2", "b3"}


def test_condition_bags_can_use_medium_key_without_cell_line_matching():
    rna = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    images = np.arange(2 * 1 * 2 * 2, dtype=np.float32).reshape(2, 1, 2, 2)
    rna_metadata = _rna_metadata().iloc[:2].copy()
    image_metadata = _image_metadata().iloc[:2].copy()
    rna_metadata["cell_line"] = "A549"
    image_metadata["cell_line"] = "U2OS"
    rna_metadata["condition_key_medium"] = "drugA|10uM|48h"
    image_metadata["condition_key_medium"] = "drugA|10uM|48h"

    rna_dataset = RNAConditionBagDataset(
        rna,
        rna_metadata,
        rna_bag_size=2,
        min_rna_bag_size=1,
        split="val",
        condition_key_col="condition_key_medium",
    )
    image_dataset = ImageConditionBagDataset(
        images,
        image_metadata,
        image_bag_size=1,
        min_image_bag_size=1,
        split="val",
        condition_key_col="condition_key_medium",
    )

    item = PairedConditionBagDataset(rna_dataset, image_dataset)[0]

    assert item["condition_id"] == "drugA|10uM|48h"
    assert item["condition"]["condition_key"] == "drugA|10uM|48h"
    assert item["condition"]["cell_line"] == "A549"
