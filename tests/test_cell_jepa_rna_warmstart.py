import torch

from perturb_jepa.models.cell_jepa_rna import CellJEPARNAConfig, CellJEPARNAWarmstart, make_cell_jepa_views, per_cell_quantile_binning


def test_quantile_binning_preserves_zero_and_orders_nonzero_values():
    values = torch.tensor([[0.0, 4.0, 1.0, 2.0], [3.0, 0.0, 9.0, 6.0]])

    binned = per_cell_quantile_binning(values, num_bins=4)

    assert binned[0, 0].item() == 0.0
    assert binned[0, 2] < binned[0, 3] < binned[0, 1]
    assert binned[1, 1].item() == 0.0
    assert binned[1, 0] < binned[1, 3] < binned[1, 2]


def test_cell_jepa_view_masks_expression_values_without_gene_identity_masking():
    values = torch.tensor([[0.0, 2.0, 3.0, 4.0, 5.0]])
    generator = torch.Generator().manual_seed(7)

    view = make_cell_jepa_views(values, mask_prob=0.5, mask_value=-1.0, expressed_gene_subsample=3, generator=generator)

    assert view.teacher_values.shape == values.shape
    assert view.student_values.shape == values.shape
    assert view.expression_value_mask.any()
    assert torch.all(view.student_values[view.expression_value_mask] == -1.0)
    assert torch.all(view.teacher_values == values)
    assert view.subsample_keep_mask.dtype == torch.bool


def test_cell_jepa_forward_uses_stop_gradient_teacher_and_jepa_dominant_loss():
    config = CellJEPARNAConfig(vocab_size=8, dim=8, depth=1, heads=2, max_genes=8, w_jepa=20.0, w_rec=1.0)
    model = CellJEPARNAWarmstart(config)
    gene_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    values = torch.rand(2, 8)
    view = make_cell_jepa_views(values, mask_prob=0.4, mask_value=-1.0, expressed_gene_subsample=None, generator=torch.Generator().manual_seed(3))

    outputs = model.forward_from_view(gene_ids, view)
    outputs["loss"].backward()
    model.update_teacher()

    assert outputs["target_embedding"].requires_grad is False
    assert outputs["weighted_jepa_to_rec_ratio"].item() > 1.0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
