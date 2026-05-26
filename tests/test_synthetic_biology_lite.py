from __future__ import annotations

import builtins

import numpy as np

from perturb_jepa.training.synthetic_biology_lite import (
    SyntheticBiologyLiteConfig,
    generate_synthetic_biology_lite,
    synthetic_lite_config,
)


def test_generator_does_not_read_real_data(monkeypatch):
    real_open = builtins.open

    def guarded_open(*args, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        if "r" in mode:
            raise AssertionError("synthetic generator attempted to read from disk")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig())
    assert dataset.expression_values.shape[1] == 128


def test_same_seed_gives_identical_output():
    config = SyntheticBiologyLiteConfig(seed=11)
    first = generate_synthetic_biology_lite(config)
    second = generate_synthetic_biology_lite(config)
    np.testing.assert_allclose(first.expression_values, second.expression_values)
    np.testing.assert_allclose(first.images, second.images)
    assert first.metadata["cell_id"].tolist() == second.metadata["cell_id"].tolist()


def test_different_seed_gives_different_output():
    first = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=11))
    second = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=12))
    assert not np.allclose(first.expression_values, second.expression_values)


def test_train_val_test_cells_do_not_leak():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=1))
    split_to_ids = {
        split: set(frame["cell_id"])
        for split, frame in dataset.metadata.groupby("split")
    }
    assert split_to_ids["train"].isdisjoint(split_to_ids["val"])
    assert split_to_ids["train"].isdisjoint(split_to_ids["test"])
    assert split_to_ids["val"].isdisjoint(split_to_ids["test"])


def test_heldout_perturbations_are_held_out():
    dataset = generate_synthetic_biology_lite(
        SyntheticBiologyLiteConfig(num_perturbations=5, heldout_perturbations=(4,), seed=2)
    )
    train_perturbations = set(dataset.metadata.loc[dataset.metadata["split"] == "train", "perturbation_id"])
    heldout = dataset.metadata[dataset.metadata["perturbation_id"] == 4]
    assert not heldout.empty
    assert set(heldout["split"]) == {"test_heldout_perturbation"}
    assert 4 not in train_perturbations


def test_heldout_doses_are_held_out():
    dataset = generate_synthetic_biology_lite(
        SyntheticBiologyLiteConfig(doses=(0.0, 1.0, 2.0), heldout_doses=(2.0,), seed=3)
    )
    train_doses = set(dataset.metadata.loc[dataset.metadata["split"] == "train", "dose"])
    heldout = dataset.metadata[dataset.metadata["dose"] == 2.0]
    assert not heldout.empty
    assert set(heldout["split"]) == {"test_heldout_dose"}
    assert 2.0 not in train_doses


def test_heldout_batches_are_held_out():
    dataset = generate_synthetic_biology_lite(
        SyntheticBiologyLiteConfig(num_batches=3, heldout_batches=(2,), seed=4)
    )
    train_batches = set(dataset.metadata.loc[dataset.metadata["split"] == "train", "batch_id"])
    heldout = dataset.metadata[dataset.metadata["batch_id"] == 2]
    assert not heldout.empty
    assert set(heldout["split"]) == {"test_heldout_batch"}
    assert 2 not in train_batches


def test_batch_corrupted_rna_differs_from_clean_rna():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=5))
    assert dataset.clean_rna.shape == dataset.expression_values.shape
    assert float(np.mean(np.abs(dataset.clean_rna - dataset.expression_values))) > 0.1


def test_synthetic_control_exists_in_every_cell_line_batch_group():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=6))
    controls = dataset.metadata[dataset.metadata["perturbation_id"] == 0]
    groups = controls.groupby(["cell_line_id", "batch_id"])
    assert len(groups) == dataset.config.num_cell_lines * dataset.config.num_batches
    assert all(len(group) > 0 for _, group in groups)


def test_genetic_anchor_config_uses_fixed_dose_and_cross_batch_replicates():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_genetic_anchor_lite", seed=9))
    assert set(dataset.metadata["dose"]) == {1.0}
    train = dataset.metadata[(dataset.metadata["split"] == "train") & (dataset.metadata["perturbation_id"] != 0)]
    train_batches_per_key = train.groupby(["perturbation_id", "cell_line_id", "dose", "time"])["batch_id"].nunique()
    assert int(train_batches_per_key.min()) >= 2
    heldout = dataset.metadata[dataset.metadata["split"] == "test_heldout_perturbation"]
    heldout_batches_per_key = heldout.groupby(["perturbation_id", "cell_line_id", "dose", "time"])["batch_id"].nunique()
    assert int(heldout_batches_per_key.min()) >= 2


def test_program_aligned_genetic_config_has_program_predictable_heldout_effects():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_program_aligned_genetic_lite", seed=9))
    assert dataset.config.perturbation_effect_mode == "program_aligned"
    assert set(dataset.metadata["dose"]) == {1.0}
    train_perturbations = set(dataset.metadata.loc[dataset.metadata["split"] == "train", "perturbation_id"])
    heldout = tuple(dataset.config.heldout_perturbations)
    for perturbation_id in heldout:
        program_id = perturbation_id % dataset.config.num_programs
        same_program_train = [
            train_id
            for train_id in train_perturbations
            if int(train_id) != dataset.config.control_perturbation_id and int(train_id) % dataset.config.num_programs == program_id
        ]
        different_program_train = [
            train_id
            for train_id in train_perturbations
            if int(train_id) != dataset.config.control_perturbation_id and int(train_id) % dataset.config.num_programs != program_id
        ]
        assert same_program_train
        heldout_direction = dataset.perturbation_directions[perturbation_id]
        same_mean = dataset.perturbation_directions[np.asarray(same_program_train, dtype=int)].mean(axis=0)
        different_mean = dataset.perturbation_directions[np.asarray(different_program_train, dtype=int)].mean(axis=0)
        same_cosine = float(np.dot(heldout_direction, same_mean) / (np.linalg.norm(heldout_direction) * np.linalg.norm(same_mean)))
        different_cosine = float(np.dot(heldout_direction, different_mean) / (np.linalg.norm(heldout_direction) * np.linalg.norm(different_mean)))
        assert same_cosine > 0.95
        assert same_cosine > different_cosine + 0.5


def test_program_aligned_split_contract_configs_do_not_mutate_locked_baseline():
    locked = synthetic_lite_config("synth_program_aligned_genetic_lite", seed=9)
    extrapolative = synthetic_lite_config("synth_program_aligned_extrapolative_holdout_lite", seed=9)
    random_holdout = synthetic_lite_config("synth_program_aligned_random_holdout_lite", seed=9)
    stratified = synthetic_lite_config("synth_program_aligned_stratified_holdout_lite", seed=9)

    assert locked.heldout_perturbations == (9, 10, 11)
    assert extrapolative.heldout_perturbations == locked.heldout_perturbations
    assert random_holdout.heldout_perturbations == (1, 4, 11)
    assert stratified.heldout_perturbations == (5, 6, 7)
    assert all(config.perturbation_effect_mode == "program_aligned" for config in (extrapolative, random_holdout, stratified))


def test_condition_bags_are_compatible_with_bridge_batch_shapes():
    dataset = generate_synthetic_biology_lite(SyntheticBiologyLiteConfig(seed=7))
    batch = next(dataset.iter_condition_batches(split="train", batch_size=3, bag_size=4, seed=8))
    assert batch.gene_ids.shape == (3, 4, dataset.config.genes)
    assert batch.expression_values.shape == (3, 4, dataset.config.genes)
    assert batch.images.shape == (3, 4, dataset.config.image_channels, dataset.config.image_size, dataset.config.image_size)
    assert batch.image_patch_mask.shape == (3, 4, (dataset.config.image_size // dataset.config.patch_size) ** 2)
