from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_biooperator_contract_audit import (
    LatentBundle,
    features,
    fit_ridge,
    load_bundle,
    predict_ridge,
    transition_metrics,
)


PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")


def test_delta_sign_contract_detects_swap_or_negation():
    source = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    target = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
    delta = target - source

    good = F.normalize(source + delta, dim=-1)
    bad = F.normalize(source - delta, dim=-1)

    source_cos = F.cosine_similarity(source, target, dim=-1)
    assert F.cosine_similarity(good, target, dim=-1).item() > source_cos.item()
    assert F.cosine_similarity(bad, target, dim=-1).item() < source_cos.item()


def test_one_step_gradient_contract_raw_and_whitened_delta_mse():
    source = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0]]), dim=-1)
    target = F.normalize(torch.tensor([[0.0, 1.0], [0.2, 0.98], [0.98, 0.2], [1.0, 0.0]]), dim=-1)
    action = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    x = torch.cat((source, action), dim=-1)
    delta = target - source

    for whitened in (False, True):
        torch.manual_seed(0)
        head = torch.nn.Linear(x.shape[1], source.shape[1])
        torch.nn.init.normal_(head.weight, std=1.0e-3)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
        before = F.cosine_similarity(F.normalize(source + head(x), dim=-1), target, dim=-1).mean()
        pred = head(x)
        if whitened:
            scale = delta.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-3)
            loss = F.mse_loss((pred - delta.mean(dim=0, keepdim=True)) / scale, (delta - delta.mean(dim=0, keepdim=True)) / scale)
        else:
            loss = F.mse_loss(pred, delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        after = F.cosine_similarity(F.normalize(source + head(x), dim=-1), target, dim=-1).mean()
        assert after.item() > before.item()


def test_ridge_equivalence_contract_matches_cached_floor():
    train = load_bundle(PHASE4_CACHE / "synth_genetic_anchor_lite_train_latents", "train")
    eval_bundle = load_bundle(PHASE4_CACHE / "synth_genetic_anchor_lite_test_heldout_perturbation_latents", "eval")
    fit = fit_ridge(features(train), train.delta, alpha=1.0e-2)
    x_train = features(train)
    x_eval = features(eval_bundle)
    linear = torch.nn.Linear(x_train.shape[1], train.delta.shape[1])
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(fit.coef.T, dtype=torch.float32))
        linear.bias.copy_(torch.as_tensor((fit.y_mean - fit.x_mean @ fit.coef).reshape(-1), dtype=torch.float32))

    train_delta = linear(torch.as_tensor(x_train, dtype=torch.float32)).detach().numpy()
    eval_delta = linear(torch.as_tensor(x_eval, dtype=torch.float32)).detach().numpy()
    assert np.allclose(train_delta, predict_ridge(fit, x_train), atol=1.0e-5)
    assert np.allclose(eval_delta, predict_ridge(fit, x_eval), atol=1.0e-5)

    train_metrics = transition_metrics(train, train.source + train_delta)
    eval_metrics = transition_metrics(eval_bundle, eval_bundle.source + eval_delta)
    assert train_metrics["transition_source_cosine_improvement"] >= 0.0600
    assert train_metrics["delta_cosine"] >= 0.75
    assert eval_metrics["transition_source_cosine_improvement"] >= 0.0035
    assert eval_metrics["delta_cosine"] >= 0.30
    assert eval_metrics["delta_prediction_effective_rank"] >= 8.0
