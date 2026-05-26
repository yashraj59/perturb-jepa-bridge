import numpy as np
import torch

from perturb_jepa.config import ExperimentConfig
from perturb_jepa.training.prefit_readout import (
    fit_pls_readout,
    install_prefit_pls_distillation_head,
    install_prefit_pls_readout,
    load_prefit_pls_readout,
    save_prefit_pls_readout,
)
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import forward_batch


def test_prefit_pls_readout_round_trip_preserves_transform(tmp_path):
    rng = np.random.default_rng(0)
    rna = rng.normal(size=(12, 8)).astype(np.float32)
    image = rng.normal(size=(12, 6)).astype(np.float32)
    readout = fit_pls_readout(rna, image, rank=3)

    path = save_prefit_pls_readout(readout, tmp_path / "prefit.json")
    loaded = load_prefit_pls_readout(path)

    assert loaded.rank == 3
    np.testing.assert_allclose(loaded.rna.transform(rna), readout.rna.transform(rna))
    np.testing.assert_allclose(loaded.image.transform(image), readout.image.transform(image))


def test_install_prefit_pls_readout_selects_frozen_raw_linear_heads():
    rng = np.random.default_rng(1)
    rna = rng.normal(size=(12, 64)).astype(np.float32)
    image = rng.normal(size=(12, 3 * 32 * 32)).astype(np.float32)
    readout = fit_pls_readout(rna, image, rank=3)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "shared_dim": 3,
                "rna_condition_readout": "raw_linear_pseudobulk",
                "rna_pseudobulk_normalize": False,
                "image_condition_readout": "raw_linear_pooled",
                "image_raw_normalize": False,
            }
        }
    )
    model = config.build_model()
    install_prefit_pls_readout(model, readout, freeze=True)
    batch = make_synthetic_bridge_batch(batch_size=2, genes=64, vocab_size=128)

    outputs = forward_batch(model, batch)
    expected_rna = readout.rna.transform(batch.expression_values.detach().numpy())
    expected_image = readout.image.transform(batch.images.detach().numpy().reshape(batch.images.shape[0], -1))

    np.testing.assert_allclose(outputs["rna_shared"].detach().numpy(), expected_rna, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs["image_shared"].detach().numpy(), expected_image, rtol=1e-5, atol=1e-5)
    assert all(not parameter.requires_grad for parameter in model.rna_raw_linear_projection.parameters())
    assert all(not parameter.requires_grad for parameter in model.image_raw_linear_projection.parameters())


def test_distilled_head_outputs_do_not_replace_prefit_retrieval_path():
    rng = np.random.default_rng(2)
    rna = rng.normal(size=(12, 64)).astype(np.float32)
    image = rng.normal(size=(12, 3 * 32 * 32)).astype(np.float32)
    readout = fit_pls_readout(rna, image, rank=3)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "shared_dim": 3,
                "rna_condition_readout": "raw_linear_pseudobulk",
                "rna_pseudobulk_normalize": False,
                "image_condition_readout": "raw_linear_pooled",
                "image_raw_normalize": False,
            }
        }
    )
    model = config.build_model()
    install_prefit_pls_readout(model, readout, freeze=True)
    batch = make_synthetic_bridge_batch(batch_size=2, genes=64, vocab_size=128)

    outputs = forward_batch(model, batch)

    assert "rna_distilled_shared" in outputs
    assert "image_distilled_shared" in outputs
    torch.testing.assert_close(outputs["rna_retrieval"], outputs["rna_shared"])
    torch.testing.assert_close(outputs["image_retrieval"], outputs["image_shared"])
    assert not torch.allclose(outputs["rna_distilled_shared"], outputs["rna_retrieval"])
    assert not torch.allclose(outputs["image_distilled_shared"], outputs["image_retrieval"])


def test_prefit_pls_can_be_cloned_into_separate_distillation_head():
    rng = np.random.default_rng(3)
    rna = rng.normal(size=(12, 64)).astype(np.float32)
    image = rng.normal(size=(12, 3 * 32 * 32)).astype(np.float32)
    readout = fit_pls_readout(rna, image, rank=3)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "shared_dim": 3,
                "rna_condition_readout": "raw_linear_pseudobulk",
                "rna_pseudobulk_normalize": False,
                "image_condition_readout": "raw_linear_pooled",
                "image_raw_normalize": False,
            }
        }
    )
    model = config.build_model()
    install_prefit_pls_readout(model, readout, freeze=True)
    install_prefit_pls_distillation_head(model, readout, freeze=False)
    batch = make_synthetic_bridge_batch(batch_size=2, genes=64, vocab_size=128)

    outputs = forward_batch(model, batch)

    torch.testing.assert_close(outputs["rna_distilled_linear_shared"], outputs["rna_retrieval"])
    torch.testing.assert_close(outputs["image_distilled_linear_shared"], outputs["image_retrieval"])
    torch.testing.assert_close(outputs["rna_distilled_residual_shared"], outputs["rna_retrieval"])
    torch.testing.assert_close(outputs["image_distilled_residual_shared"], outputs["image_retrieval"])
    assert model.rna_distilled_residual_scale.item() == 0.0
    assert model.image_distilled_residual_scale.item() == 0.0
    assert all(parameter.requires_grad for parameter in model.rna_distilled_linear_projection.parameters())
    assert all(parameter.requires_grad for parameter in model.image_distilled_linear_projection.parameters())
    assert all(not parameter.requires_grad for parameter in model.rna_raw_linear_projection.parameters())
    assert all(not parameter.requires_grad for parameter in model.image_raw_linear_projection.parameters())


def test_counterfactual_rna_residual_adds_source_expression():
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "counterfactual_rna_residual": True,
                "rna": {"vocab_size": 64, "max_genes": 64},
            }
        }
    )
    model = config.build_model()
    batch = make_synthetic_bridge_batch(batch_size=2, genes=64, vocab_size=64)

    outputs = forward_batch(model, batch)

    assert "counterfactual_rna_delta" in outputs
    torch.testing.assert_close(
        outputs["counterfactual_rna"],
        batch.expression_values + outputs["counterfactual_rna_delta"],
    )


def test_program_factorized_counterfactual_decoder_broadcasts_program_deltas():
    assignment = (0, 1, 0, 2, 1, 2)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "counterfactual_rna_residual": True,
                "counterfactual_rna_program_factorized": True,
                "counterfactual_rna_num_programs": 3,
                "counterfactual_rna_program_assignment": assignment,
                "rna": {"vocab_size": 6, "max_genes": 6},
            }
        }
    )
    model = config.build_model()
    for parameter in model.counterfactual_program_decoder.parameters():
        parameter.data.zero_()
    model.counterfactual_program_decoder.net[-1].bias.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    batch = make_synthetic_bridge_batch(batch_size=2, genes=6, vocab_size=6)

    outputs = forward_batch(model, batch)

    expected_delta = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 3.0]).expand_as(batch.expression_values)
    torch.testing.assert_close(outputs["counterfactual_rna_program_gene_delta"], expected_delta)
    torch.testing.assert_close(outputs["counterfactual_rna_delta"], expected_delta)
    torch.testing.assert_close(outputs["counterfactual_rna"], batch.expression_values + expected_delta)


def test_within_program_counterfactual_residual_has_zero_program_means():
    assignment = (0, 1, 0, 2, 1, 2)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "counterfactual_rna_residual": True,
                "counterfactual_rna_program_factorized": True,
                "counterfactual_rna_num_programs": 3,
                "counterfactual_rna_program_assignment": assignment,
                "counterfactual_rna_within_program_residual": True,
                "rna": {"vocab_size": 6, "max_genes": 6},
            }
        }
    )
    model = config.build_model()
    for parameter in model.counterfactual_program_decoder.parameters():
        parameter.data.zero_()
    for parameter in model.rna_distribution_decoder.parameters():
        parameter.data.zero_()
    model.rna_distribution_decoder.net[-1].bias.data.copy_(torch.tensor([1.0, 3.0, 2.0, 6.0, 10.0, 14.0]))
    batch = make_synthetic_bridge_batch(batch_size=2, genes=6, vocab_size=6)

    outputs = forward_batch(model, batch)

    residual = outputs["counterfactual_rna_within_program_residual"]
    assignment_tensor = torch.tensor(assignment)
    for program in assignment_tensor.unique():
        mask = assignment_tensor.eq(program)
        torch.testing.assert_close(residual[:, mask].mean(dim=1), torch.zeros(batch.expression_values.shape[0]))


def test_source_conditioned_program_decoder_exposes_source_program_means():
    assignment = (0, 1, 0, 2, 1, 2)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "counterfactual_rna_residual": True,
                "counterfactual_rna_program_factorized": True,
                "counterfactual_rna_num_programs": 3,
                "counterfactual_rna_program_assignment": assignment,
                "counterfactual_rna_program_conditioned": True,
                "rna": {"vocab_size": 6, "max_genes": 6},
            }
        }
    )
    model = config.build_model()
    batch = make_synthetic_bridge_batch(batch_size=2, genes=6, vocab_size=6)
    batch.expression_values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 6.0, 10.0, 14.0],
            [2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
        ]
    )

    outputs = forward_batch(model, batch)

    expected = torch.tensor(
        [
            [2.0, 6.0, 10.0],
            [4.0, 8.0, 12.0],
        ]
    )
    torch.testing.assert_close(outputs["counterfactual_rna_source_program_context"], expected)
    assert model.counterfactual_program_decoder.net[0].in_features == model.config.shared_dim + 3


def test_program_decoder_metadata_context_excludes_batch_id():
    assignment = (0, 1, 0, 2, 1, 2)
    config = ExperimentConfig.from_dict(
        {
            "model": {
                "counterfactual_rna_program_factorized": True,
                "counterfactual_rna_num_programs": 3,
                "counterfactual_rna_program_assignment": assignment,
                "counterfactual_rna_program_metadata_context": True,
                "perturbation": {"num_perturbations": 4, "num_cell_lines": 2, "num_batches": 5},
                "rna": {"vocab_size": 6, "max_genes": 6},
            }
        }
    )
    model = config.build_model()
    batch = make_synthetic_bridge_batch(
        batch_size=2,
        genes=6,
        vocab_size=6,
        num_perturbations=4,
        num_cell_lines=2,
        num_batches=5,
    )

    outputs = forward_batch(model, batch)

    expected_width = 4 + 2 + 2
    assert outputs["counterfactual_rna_metadata_context"].shape == (2, expected_width)
    assert model.counterfactual_program_decoder.net[0].in_features == model.config.shared_dim + expected_width
