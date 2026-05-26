from pathlib import Path

import numpy as np
import torch

from perturb_jepa.models.biospectral_jepa import (
    FloorPreservingTransitionHead,
    NeuralReducedRankRidgeHead,
    RankLadderTransitionHead,
    RidgeFloorHead,
    SpectralResidualHead,
)
from perturb_jepa.training.biospectral_operator import (
    bundle_features,
    bundle_transition_metrics,
    fit_reduced_rank_ridge_numpy,
    load_latent_bundle,
    predict_reduced_rank_ridge_numpy,
)


PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")


def _bundles():
    train = load_latent_bundle(PHASE4_CACHE / "synth_genetic_anchor_lite_train_latents", "train")
    eval_bundle = load_latent_bundle(PHASE4_CACHE / "synth_genetic_anchor_lite_test_heldout_perturbation_latents", "eval")
    return train, eval_bundle


def _tensor(values):
    return torch.as_tensor(values, dtype=torch.float32)


def test_full_ridge_floor_reproduction_matches_registered_floor():
    train, eval_bundle = _bundles()
    head = RidgeFloorHead(train.source.shape[1], train.action.shape[1])
    head.fit(_tensor(train.source), _tensor(train.action), _tensor(train.delta), alpha=1.0e-2)

    eval_delta = head(_tensor(eval_bundle.source), _tensor(eval_bundle.action)).detach().numpy()
    metrics = bundle_transition_metrics(eval_bundle, eval_delta)

    assert abs(metrics["transition_source_cosine_improvement"] - 0.0057) <= 1.0e-4
    assert abs(metrics["delta_cosine"] - 0.3980) <= 1.0e-4
    assert abs(metrics["transition_to_target_recall@1"] - 0.4815) <= 1.0e-4
    assert abs(metrics["delta_prediction_effective_rank"] - 10.2835) <= 1.0e-3


def test_low_rank_ridge_floor_reproduction_matches_registered_floor():
    train, eval_bundle = _bundles()
    fit = fit_reduced_rank_ridge_numpy(bundle_features(train), train.delta, rank=8, alpha=1.0e-2)
    eval_delta = predict_reduced_rank_ridge_numpy(fit, bundle_features(eval_bundle))
    metrics = bundle_transition_metrics(eval_bundle, eval_delta)

    assert abs(metrics["transition_source_cosine_improvement"] - 0.0046) <= 1.0e-4
    assert abs(metrics["delta_cosine"] - 0.3877) <= 1.0e-4
    assert abs(metrics["transition_to_target_recall@1"] - 0.4074) <= 1.0e-4
    assert abs(metrics["delta_prediction_effective_rank"] - 6.7681) <= 1.0e-3


def test_neural_low_rank_exact_equivalence_to_analytic():
    train, eval_bundle = _bundles()
    analytic = fit_reduced_rank_ridge_numpy(bundle_features(train), train.delta, rank=8, alpha=1.0e-2)
    analytic_eval = predict_reduced_rank_ridge_numpy(analytic, bundle_features(eval_bundle))

    head = NeuralReducedRankRidgeHead(train.source.shape[1], train.action.shape[1], rank=8)
    head.fit(_tensor(train.source), _tensor(train.action), _tensor(train.delta), alpha=1.0e-2)
    neural_eval = head(_tensor(eval_bundle.source), _tensor(eval_bundle.action)).detach().numpy()

    assert np.allclose(neural_eval, analytic_eval, atol=1.0e-5)
    neural_metrics = bundle_transition_metrics(eval_bundle, neural_eval)
    analytic_metrics = bundle_transition_metrics(eval_bundle, analytic_eval)
    assert abs(neural_metrics["transition_source_cosine_improvement"] - analytic_metrics["transition_source_cosine_improvement"]) <= 1.0e-4
    assert abs(neural_metrics["delta_prediction_effective_rank"] - analytic_metrics["delta_prediction_effective_rank"]) <= 1.0e-3


def test_zero_residual_preserves_full_ridge_floor_exactly():
    train, eval_bundle = _bundles()
    head = FloorPreservingTransitionHead(train.source.shape[1], train.action.shape[1], residual_rank=12, hidden_dim=32)
    head.fit_floor_and_basis(_tensor(train.source), _tensor(train.action), _tensor(train.delta), alpha=1.0e-2)

    out = head(_tensor(eval_bundle.source), _tensor(eval_bundle.action))
    delta = out["predicted_delta_bio"].detach().numpy()
    floor = out["delta_floor"].detach().numpy()

    assert np.max(np.abs(delta - floor)) <= 1.0e-7
    assert out["residual_to_floor_norm_ratio"].item() == 0.0


def test_residual_branch_initialized_zero_and_cap_triggers():
    source = torch.randn(5, 4)
    action = torch.randn(5, 2)
    floor_delta = torch.ones(5, 4)
    residual = SpectralResidualHead(4, 2, rank=3, hidden_dim=8, norm_cap=0.25, init_scale=0.0)

    out = residual(source, action, floor_delta=floor_delta)
    assert torch.allclose(out["delta_residual"], torch.zeros_like(out["delta_residual"]))

    with torch.no_grad():
        residual.mlp[-1].bias.fill_(10.0)
    capped = residual(source, action, floor_delta=floor_delta)
    assert capped["residual_cap_hit_fraction"].item() > 0.0
    assert capped["residual_to_floor_norm_ratio"].item() <= 0.25 + 1.0e-6


def test_rank_ladder_includes_full_floor_expert_and_valid_router_probabilities():
    train, eval_bundle = _bundles()
    head = RankLadderTransitionHead(train.source.shape[1], train.action.shape[1], ranks=(4, 8, 12, 24))
    head.fit(_tensor(train.source), _tensor(train.action), _tensor(train.delta), alpha=1.0e-2)
    out = head(_tensor(eval_bundle.source), _tensor(eval_bundle.action))
    weights = out["router_weights"]

    assert "full" in head.expert_names
    assert torch.all(weights >= 0.0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0]))
    assert torch.allclose(out["delta_ladder"], head.full_floor(_tensor(eval_bundle.source), _tensor(eval_bundle.action)))
