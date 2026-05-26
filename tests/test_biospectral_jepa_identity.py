import torch

from perturb_jepa.models.biospectral_jepa import BioSpectralJEPAConfig, FloorPreservingTransitionHead
from perturb_jepa.training.biospectral_losses import biospectral_loss
from perturb_jepa.training.biospectral_operator import identity_metrics, identity_violation


def test_biospectral_config_exposes_rank_preserving_defaults():
    config = BioSpectralJEPAConfig()

    assert config.bio_dim == 24
    assert config.tech_dim == 8
    assert config.rank_ladder == (4, 8, 12, 24)
    assert config.use_full_ridge_floor
    assert config.freeze_ridge_floor
    assert config.residual_init_scale == 0.0
    assert config.residual_norm_cap == 0.25


def test_identity_report_passes_without_forbidden_shortcuts():
    metrics = identity_metrics(frozen_latent=True, full_jepa=False)

    assert metrics["encoder_path_used"] == 1.0
    assert metrics["pls_raw_linear_main_path_used"] == 0.0
    assert metrics["condition_key_feature_present"] == 0.0
    assert metrics["biological_key_one_hot_present"] == 0.0
    assert metrics["test_target_mean_used_for_fit"] == 0.0
    assert metrics["pooled_train_test_target_used_for_fit"] == 0.0
    assert metrics["teacher_stop_gradient_verified"] == 1.0
    assert not identity_violation(metrics)


def test_identity_report_flags_forbidden_condition_key_features():
    metrics = identity_metrics(frozen_latent=True)
    metrics["condition_key_feature_present"] = 1.0

    assert identity_violation(metrics)


def test_teacher_targets_are_stop_gradient_in_loss():
    source = torch.randn(6, 4, requires_grad=True)
    target = torch.randn(6, 4, requires_grad=True)
    delta = target - source.detach()
    outputs = {
        "z_control_bio": source,
        "z_target_bio_teacher": target,
        "predicted_target_bio": torch.nn.functional.normalize(source + delta.detach(), dim=-1),
        "predicted_delta_bio": delta.detach().clone().requires_grad_(True),
        "delta_teacher": delta,
        "delta_floor": delta.detach(),
        "delta_residual": torch.zeros_like(delta),
    }

    loss, diagnostics = biospectral_loss(outputs)
    loss.backward()

    assert diagnostics["loss/total"].item() >= 0.0
    assert target.grad is None


def test_floor_preserving_head_initialization_contract():
    torch.manual_seed(0)
    source = torch.randn(12, 5)
    action = torch.randn(12, 3)
    delta = torch.randn(12, 5) * 0.1
    head = FloorPreservingTransitionHead(5, 3, residual_rank=4, hidden_dim=8)
    head.fit_floor_and_basis(source, action, delta, alpha=1.0e-2)
    out = head(source, action)

    assert torch.allclose(out["predicted_delta_bio"], out["delta_floor"], atol=1.0e-7)
    assert out["residual_to_floor_norm_ratio"].item() == 0.0
