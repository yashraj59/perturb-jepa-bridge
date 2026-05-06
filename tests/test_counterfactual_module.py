import torch

from perturb_jepa.models.counterfactual import (
    CounterfactualResponsePredictor,
    PerturbationConditionEncoder,
    counterfactual_cosine_loss,
    counterfactual_mmd_loss,
    gaussian_nll_loss,
)


def test_condition_encoder_and_counterfactual_predictor_shapes():
    torch.manual_seed(0)
    encoder = PerturbationConditionEncoder(num_conditions=5, feature_dim=2, hidden_dim=8, output_dim=6)
    predictor = CounterfactualResponsePredictor(4, 6, hidden_dim=10)
    control = torch.randn(3, 2, 4)
    condition = encoder(torch.tensor([0, 1, 2]), torch.randn(3, 2))

    output = predictor(control, condition)

    assert output.delta_mu.shape == control.shape
    assert output.delta_logvar.shape == control.shape
    assert output.treated_mu.shape == control.shape
    assert output.treated_logvar.shape == control.shape
    assert torch.allclose(output.treated_mu, control + output.delta_mu)
    assert torch.exp(output.treated_logvar).gt(0).all()


def test_counterfactual_helper_losses_are_finite():
    torch.manual_seed(0)
    mu = torch.randn(2, 3, 4)
    logvar = torch.zeros_like(mu)
    target = mu + 0.1 * torch.randn_like(mu)

    assert torch.isfinite(gaussian_nll_loss(mu, logvar, target))
    assert torch.isfinite(counterfactual_mmd_loss(mu, target))
    assert torch.isfinite(counterfactual_cosine_loss(mu, target))


def test_condition_encoder_supports_named_biological_fields():
    encoder = PerturbationConditionEncoder(
        num_perturbations=4,
        num_cell_lines=3,
        num_time_bins=2,
        num_moas=2,
        num_pathways=2,
        hidden_dim=8,
        output_dim=6,
    )

    embedding = encoder(
        perturbation_id=torch.tensor([0, 1]),
        dose=torch.tensor([1.0, 10.0]),
        time=torch.tensor([24.0, 48.0]),
        time_id=torch.tensor([0, 1]),
        cell_line_id=torch.tensor([1, 2]),
        moa_id=torch.tensor([0, 1]),
        pathway_id=torch.tensor([1, 0]),
    )

    assert embedding.shape == (2, 6)
