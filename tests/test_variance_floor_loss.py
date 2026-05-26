from __future__ import annotations

import pytest
import torch

from perturb_jepa.losses import (
    BridgeLossWeights,
    bridge_loss,
    covariance_offdiag_loss,
    cross_correlation_identity_loss,
    variance_floor_loss,
)


def test_variance_floor_loss_penalizes_collapsed_embeddings():
    collapsed = torch.ones(8, 4)
    spread = torch.randn(8, 4) * 2.0
    assert variance_floor_loss(collapsed) > variance_floor_loss(spread)


def test_shared_variance_weight_defaults_to_zero_in_bridge_loss():
    outputs = {
        "rna_shared": torch.ones(8, 4),
        "image_shared": torch.ones(8, 4),
    }
    total, terms = bridge_loss(
        outputs,
        weights=BridgeLossWeights(align=0.0, mmd=0.0, sliced_wasserstein=0.0),
    )
    assert "rna_shared_variance" in terms
    assert "image_shared_variance" in terms
    assert torch.isclose(total, torch.tensor(0.0))


def test_covariance_offdiag_loss_penalizes_redundant_dimensions():
    base = torch.linspace(-1.0, 1.0, 16)
    redundant = torch.stack([base, base, -base], dim=1)
    independent = torch.eye(16, 4)
    assert covariance_offdiag_loss(redundant) > covariance_offdiag_loss(independent)


def test_cross_correlation_identity_loss_prefers_matching_dimensions():
    x = torch.eye(8, 4)
    y = x.clone()
    shuffled = x[:, [1, 0, 2, 3]]
    assert cross_correlation_identity_loss(x, y) < cross_correlation_identity_loss(x, shuffled)


def test_shared_geometry_weights_default_to_zero_in_bridge_loss():
    outputs = {
        "rna_shared": torch.randn(8, 4),
        "image_shared": torch.randn(8, 4),
    }
    total, terms = bridge_loss(
        outputs,
        weights=BridgeLossWeights(align=0.0, mmd=0.0, sliced_wasserstein=0.0),
    )
    assert "rna_shared_covariance" in terms
    assert "image_shared_covariance" in terms
    assert "shared_cross_correlation" in terms
    assert total.item() == pytest.approx(0.0)
