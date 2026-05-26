from __future__ import annotations

import torch
import torch.nn.functional as F

from perturb_jepa.models.program_bootstrap_jepa import (
    FloorInitializedTransitionHead,
    FloorInitializedTransitionHeadConfig,
    ProgramBootstrapJEPA,
    ProgramBootstrapJEPAConfig,
)


def test_program_bootstrap_jepa_uses_ema_stop_gradient_targets():
    model = ProgramBootstrapJEPA(ProgramBootstrapJEPAConfig(genes=12, image_dim=16, action_dim=3, z_dim=5, hidden_dim=8))
    assert all(not parameter.requires_grad for parameter in model.rna_target.parameters())
    assert all(not parameter.requires_grad for parameter in model.image_target.parameters())
    batch = {
        "control_expression": torch.rand(4, 12),
        "target_expression": torch.rand(4, 12),
        "target_image_flat": torch.rand(4, 16),
        "action": torch.eye(3)[torch.tensor([0, 1, 2, 0])].float(),
        "pca_target": torch.rand(4, 5),
    }
    loss, metrics = model.forward_loss(**batch)
    assert loss.requires_grad
    assert metrics["transition_loss"].ndim == 0
    assert metrics["delta_direction_loss"].ndim == 0
    assert metrics["source_improvement_hinge"].ndim == 0
    before = [parameter.detach().clone() for parameter in model.rna_target.parameters()]
    with torch.no_grad():
        for parameter in model.rna_online.parameters():
            parameter.add_(0.01)
    model.update_targets(decay=0.5)
    after = list(model.rna_target.parameters())
    assert any(not torch.allclose(left, right) for left, right in zip(before, after, strict=True))


def test_program_bootstrap_jepa_predictors_are_query_conditioned_and_program_action_only():
    config = ProgramBootstrapJEPAConfig(genes=10, image_dim=9, action_dim=2, z_dim=4, hidden_dim=8)
    model = ProgramBootstrapJEPA(config)
    control_z = torch.rand(3, config.z_dim)
    action = torch.eye(config.action_dim)[torch.tensor([0, 1, 0])].float()
    transition = model.predict_transition(control_z, action)
    assert transition.shape == (3, config.z_dim)
    assert model.transition_query.requires_grad
    assert model.rna_to_image_query.requires_grad
    assert model.image_to_rna_query.requires_grad


def test_program_bootstrap_jepa_can_enable_delta_direction_repair_losses():
    config = ProgramBootstrapJEPAConfig(
        genes=10,
        image_dim=9,
        action_dim=2,
        z_dim=4,
        hidden_dim=8,
        delta_direction_weight=0.7,
        source_improvement_weight=0.3,
    )
    model = ProgramBootstrapJEPA(config)
    batch = {
        "control_expression": torch.rand(5, 10),
        "target_expression": torch.rand(5, 10),
        "target_image_flat": torch.rand(5, 9),
        "action": torch.eye(2)[torch.tensor([0, 1, 0, 1, 0])].float(),
        "pca_target": torch.rand(5, 4),
    }
    loss, metrics = model.forward_loss(**batch)
    assert loss.requires_grad
    assert torch.isfinite(metrics["delta_direction_loss"])
    assert torch.isfinite(metrics["source_improvement_hinge"])


def test_floor_initialized_transition_head_starts_exactly_at_frozen_floor():
    config = FloorInitializedTransitionHeadConfig(z_dim=3, action_dim=2, hidden_dim=8)
    floor_weight = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 100.0
    floor_bias = torch.tensor([0.1, -0.2, 0.3])
    head = FloorInitializedTransitionHead(config, floor_weight=floor_weight, floor_bias=floor_bias)
    source = torch.rand(4, 3)
    action = torch.rand(4, 2)
    expected = source + F.linear(torch.cat((source, action), dim=-1), floor_weight, floor_bias)
    actual = head(source, action)
    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert not head.floor_weight.requires_grad
    assert not head.floor_bias.requires_grad
    assert head.transition_query.requires_grad
    assert head.residual_scale.requires_grad
