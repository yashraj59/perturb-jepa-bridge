from __future__ import annotations

import torch

from perturb_jepa.training.bioaction_losses import (
    BioActionJEPALossWeights,
    barlow_cross_correlation_loss,
    batch_mean_invariance_loss,
    bioaction_jepa_loss,
    cosine_jepa_loss,
    masked_cosine_jepa_loss,
    prototype_ot_jepa_loss,
    vicreg_components,
)


def test_cosine_jepa_loss_identical_is_near_zero():
    x = torch.randn(4, 3, 8)
    assert float(cosine_jepa_loss(x, x).detach()) < 1e-5


def test_masked_cosine_ignores_unselected_targets():
    pred = torch.randn(2, 3, 4)
    target = pred.clone()
    target[:, 1:] = torch.randn_like(target[:, 1:])
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[:, 0] = True
    assert float(masked_cosine_jepa_loss(pred, target, mask).detach()) < 1e-5


def test_vicreg_variance_activates_on_collapse():
    collapsed = torch.zeros(8, 16)
    variance, covariance = vicreg_components(collapsed, target_std=0.05)
    assert float(variance.detach()) > 0.0
    assert torch.isfinite(covariance)


def test_barlow_and_distributional_losses_run():
    x = torch.randn(8, 16)
    y = torch.randn(8, 16)
    assert torch.isfinite(barlow_cross_correlation_loss(x, y))
    assert torch.isfinite(prototype_ot_jepa_loss(x[:, None, :].expand(-1, 3, -1), y[:, None, :].expand(-1, 3, -1)))


def test_batch_mean_invariance_loss_detects_batch_centroids():
    latents = torch.cat((torch.ones(4, 8), -torch.ones(4, 8)), dim=0)
    batch_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert float(batch_mean_invariance_loss(latents, batch_id).detach()) > 0.0


def test_total_loss_has_finite_gradients():
    pred = torch.randn(4, 2, 8, requires_grad=True)
    target = torch.randn(4, 2, 8)
    outputs = {
        "rna_program_jepa_pred": pred,
        "rna_program_jepa_target": target,
        "rna_program_jepa_available": torch.tensor(True),
        "image_region_jepa_pred": pred,
        "image_region_jepa_target": target,
        "image_region_jepa_available": torch.tensor(True),
        "rna_to_image_jepa_pred": pred,
        "rna_to_image_jepa_target": target,
        "rna_to_image_jepa_available": torch.tensor(True),
        "image_to_rna_jepa_pred": pred,
        "image_to_rna_jepa_target": target,
        "image_to_rna_jepa_available": torch.tensor(True),
        "joint_to_rna_jepa_pred": pred,
        "joint_to_rna_jepa_target": target,
        "joint_to_rna_jepa_available": torch.tensor(True),
        "joint_to_image_jepa_pred": pred,
        "joint_to_image_jepa_target": target,
        "joint_to_image_jepa_available": torch.tensor(True),
        "transition_rna_jepa_pred": pred,
        "transition_rna_jepa_target": target,
        "transition_rna_jepa_available": torch.tensor(True),
        "transition_image_jepa_pred": pred,
        "transition_image_jepa_target": target,
        "transition_image_jepa_available": torch.tensor(True),
        "transition_joint_jepa_pred": pred,
        "transition_joint_jepa_target": target,
        "transition_joint_jepa_available": torch.tensor(True),
        "rna_condition_state": torch.randn(4, 8, requires_grad=True),
        "image_condition_state": torch.randn(4, 8, requires_grad=True),
        "joint_condition_state": torch.randn(4, 8, requires_grad=True),
        "shared_state": torch.randn(4, 8, requires_grad=True),
        "batch_id_for_loss": torch.tensor([0, 0, 1, 1]),
    }
    loss, diagnostics = bioaction_jepa_loss(outputs, BioActionJEPALossWeights(batch_invariance=0.1))
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert "loss/total" in diagnostics
    assert "loss/batch_invariance" in diagnostics
