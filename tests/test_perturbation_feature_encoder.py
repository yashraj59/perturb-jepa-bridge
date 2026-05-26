import pytest
import torch

from perturb_jepa.models.perturbation_encoder import PerturbationEncoder, PerturbationEncoderConfig


def _inputs(batch_size: int = 5):
    torch.manual_seed(0)
    return {
        "perturbation_id": torch.tensor([0, 1, 2, 3, 1]),
        "perturbation_type_id": torch.tensor([0, 1, 0, 1, 0]),
        "cell_line_id": torch.tensor([0, 1, 0, 1, 0]),
        "batch_id": torch.zeros(batch_size, dtype=torch.long),
        "dose": torch.linspace(0.0, 1.0, batch_size),
        "time": torch.linspace(1.0, 2.0, batch_size),
        "descriptor": torch.randn(batch_size, 7),
    }


def test_residual_feature_mode_preserves_lookup_encoder_at_zero_scale():
    config = PerturbationEncoderConfig(
        num_perturbations=4,
        num_types=2,
        num_cell_lines=2,
        num_batches=1,
        dim=8,
        descriptor_dim=7,
        dropout=0.0,
    )
    feature_config = PerturbationEncoderConfig(
        num_perturbations=4,
        num_types=2,
        num_cell_lines=2,
        num_batches=1,
        dim=8,
        descriptor_dim=7,
        perturbation_feature_mode="residual",
        perturbation_feature_scale_init=0.0,
        dropout=0.0,
    )
    lookup = PerturbationEncoder(config)
    residual = PerturbationEncoder(feature_config)
    residual.load_state_dict(lookup.state_dict(), strict=False)
    lookup.eval()
    residual.eval()

    inputs = _inputs()
    with torch.no_grad():
        expected = lookup(**inputs)
        actual = residual(**inputs)

    torch.testing.assert_close(actual, expected)


def test_feature_only_mode_requires_descriptors():
    with pytest.raises(ValueError, match="descriptor_dim"):
        PerturbationEncoder(
            PerturbationEncoderConfig(
                num_perturbations=4,
                num_types=2,
                num_cell_lines=2,
                num_batches=1,
                dim=8,
                perturbation_feature_mode="feature_only",
            )
        )
