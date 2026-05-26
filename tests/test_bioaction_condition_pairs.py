from __future__ import annotations

from perturb_jepa.training.bioaction_batches import condition_pair_records, exact_train_key_fraction, iter_bioaction_condition_batches
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


def test_condition_pair_loader_creates_control_to_perturbed_pairs():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    records = condition_pair_records(dataset, split="train")
    assert records
    assert all(record["perturbation_id"] != dataset.config.control_perturbation_id for record in records)
    batch = next(iter_bioaction_condition_batches(dataset, split="train", batch_size=2, steps=1, seed=0))
    assert batch.control_expression_values is not None
    assert batch.target_expression_values is not None
    assert batch.control_expression_values.shape == batch.target_expression_values.shape
    assert batch.condition_key is not None
    assert batch.biological_key is not None


def test_heldout_perturbation_has_no_exact_train_target_key_coverage():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_heldout_perturbation_lite", seed=0))
    assert exact_train_key_fraction(dataset, eval_split="test_heldout_perturbation") == 0.0
    records = condition_pair_records(dataset, split="test_heldout_perturbation")
    assert records
    assert {record["split"] for record in records} == {"test_heldout_perturbation"}


def test_heldout_dose_has_heldout_dose_rows():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_dose_extrapolation_lite", seed=0))
    records = condition_pair_records(dataset, split="test_heldout_dose")
    assert records
    assert {float(record["dose"]) for record in records}.issubset(set(dataset.config.heldout_doses))


def test_keys_are_labels_not_numeric_model_features():
    dataset = generate_synthetic_biology_lite(synthetic_lite_config("synth_micro", seed=0))
    batch = next(iter_bioaction_condition_batches(dataset, split="train", batch_size=2, steps=1, seed=0))
    tensor_fields = batch.__dict__
    assert "condition_key" in tensor_fields
    assert "biological_key" in tensor_fields
    assert not hasattr(batch, "condition_key_tensor")
    assert not hasattr(batch, "biological_key_tensor")

