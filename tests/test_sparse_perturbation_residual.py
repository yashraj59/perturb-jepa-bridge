import torch

from perturb_jepa.models.sparse_perturbation_residual import (
    SparsePerturbationResidualHead,
    active_synthetic_effect_mask,
    sparse_delta_losses,
)


def test_sparse_residual_head_starts_as_source_identity_delta():
    head = SparsePerturbationResidualHead(
        num_genes=6,
        num_programs=3,
        program_assignment=(0, 1, 0, 2, 1, 2),
        num_perturbations=4,
        num_cell_lines=2,
        hidden_dim=8,
        z_pert_dim=5,
        dictionary_rank=2,
    )
    source = torch.randn(2, 6)

    outputs = head(
        source_rna=source,
        perturbation_id=torch.tensor([1, 2]),
        cell_line_id=torch.tensor([0, 1]),
        dose=torch.tensor([1.0, 1.0]),
        time=torch.tensor([0.0, 0.0]),
    )

    torch.testing.assert_close(outputs["sparse_delta_hat"], torch.zeros_like(source))
    torch.testing.assert_close(outputs["sparse_program_gene_delta"], torch.zeros_like(source))
    torch.testing.assert_close(outputs["sparse_low_rank_delta"], torch.zeros_like(source))


def test_sparse_residual_metadata_context_excludes_batch_id():
    head = SparsePerturbationResidualHead(
        num_genes=6,
        num_programs=3,
        program_assignment=(0, 1, 0, 2, 1, 2),
        num_perturbations=4,
        num_cell_lines=2,
        hidden_dim=8,
        z_pert_dim=5,
        dictionary_rank=2,
    )
    source = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    outputs = head(
        source_rna=source,
        perturbation_id=torch.tensor([1, 2]),
        cell_line_id=torch.tensor([0, 1]),
        dose=torch.tensor([1.0, 3.0]),
        time=torch.tensor([0.0, 2.0]),
    )

    expected_width = 4 + 2 + 2
    assert outputs["sparse_metadata_context"].shape == (2, expected_width)
    assert outputs["sparse_context"].shape == (2, 3 + expected_width)


def test_active_synthetic_effect_mask_uses_top_programs_and_top_de_genes():
    assignment = torch.tensor([0, 0, 1, 1, 2, 2])
    target_delta = torch.tensor([[5.0, 4.0, 0.2, 0.1, 1.0, 1.0]])

    program_mask, top_de_mask, effect_mask = active_synthetic_effect_mask(
        target_delta,
        assignment,
        top_de_k=1,
        top_programs=1,
    )

    assert torch.equal(program_mask, torch.tensor([[True, True, False, False, False, False]]))
    assert torch.equal(top_de_mask, torch.tensor([[True, False, False, False, False, False]]))
    assert torch.equal(effect_mask, torch.tensor([[True, True, False, False, False, False]]))


def test_sparse_delta_losses_backpropagate_to_sparse_head():
    head = SparsePerturbationResidualHead(
        num_genes=6,
        num_programs=3,
        program_assignment=(0, 1, 0, 2, 1, 2),
        num_perturbations=4,
        num_cell_lines=2,
        hidden_dim=8,
        z_pert_dim=5,
        dictionary_rank=2,
    )
    outputs = head(
        source_rna=torch.randn(2, 6),
        perturbation_id=torch.tensor([1, 2]),
        cell_line_id=torch.tensor([0, 1]),
        dose=torch.tensor([1.0, 1.0]),
        time=torch.tensor([0.0, 0.0]),
    )
    target_delta = torch.randn(2, 6)

    losses = sparse_delta_losses(
        outputs["sparse_delta_hat"],
        target_delta,
        z_inv=torch.randn(2, 3),
        z_pert=outputs["sparse_z_pert"],
        program_assignment=head.program_assignment,
        top_de_k=2,
        top_programs=1,
    )
    losses["total"].backward()

    assert head.low_rank_dictionary.grad is not None
    assert head.program_decoder.net[-1].weight.grad is not None
