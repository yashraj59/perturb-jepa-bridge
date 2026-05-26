from dataclasses import replace

import torch

from perturb_jepa.config import ExperimentConfig, default_bridge_config
from perturb_jepa.models.bridge import PerturbJEPABridge
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import forward_batch


def test_pseudobulk_readout_replaces_rna_shared_embedding():
    torch.manual_seed(0)
    base_config = default_bridge_config()
    encoder_model = PerturbJEPABridge(replace(base_config, rna_condition_readout="encoder"))
    pseudobulk_model = PerturbJEPABridge(replace(base_config, rna_condition_readout="pseudobulk"))
    pseudobulk_model.load_state_dict(encoder_model.state_dict())
    encoder_model.eval()
    pseudobulk_model.eval()
    batch = make_synthetic_bridge_batch(batch_size=2)

    encoder_outputs = forward_batch(encoder_model, batch)
    pseudobulk_outputs = forward_batch(pseudobulk_model, batch)

    assert "rna_pseudobulk_shared" in pseudobulk_outputs
    torch.testing.assert_close(pseudobulk_outputs["rna_shared"], pseudobulk_outputs["rna_pseudobulk_shared"])
    assert not torch.allclose(encoder_outputs["rna_shared"], pseudobulk_outputs["rna_shared"])


def test_raw_pseudobulk_readout_replaces_rna_shared_embedding():
    torch.manual_seed(0)
    base_config = default_bridge_config()
    encoder_model = PerturbJEPABridge(replace(base_config, rna_condition_readout="encoder"))
    raw_model = PerturbJEPABridge(replace(base_config, rna_condition_readout="raw_pseudobulk"))
    raw_model.load_state_dict(encoder_model.state_dict())
    encoder_model.eval()
    raw_model.eval()
    batch = make_synthetic_bridge_batch(batch_size=2)

    encoder_outputs = forward_batch(encoder_model, batch)
    raw_outputs = forward_batch(raw_model, batch)

    assert "rna_raw_pseudobulk_shared" in raw_outputs
    torch.testing.assert_close(raw_outputs["rna_shared"], raw_outputs["rna_raw_pseudobulk_shared"])
    assert not torch.allclose(encoder_outputs["rna_shared"], raw_outputs["rna_shared"])


def test_raw_image_readout_replaces_image_shared_embedding():
    torch.manual_seed(0)
    base_config = default_bridge_config()
    encoder_model = PerturbJEPABridge(replace(base_config, image_condition_readout="encoder"))
    raw_model = PerturbJEPABridge(replace(base_config, image_condition_readout="raw_pooled"))
    raw_model.load_state_dict(encoder_model.state_dict())
    encoder_model.eval()
    raw_model.eval()
    batch = make_synthetic_bridge_batch(batch_size=2)

    encoder_outputs = forward_batch(encoder_model, batch)
    raw_outputs = forward_batch(raw_model, batch)

    assert "image_raw_shared" in raw_outputs
    torch.testing.assert_close(raw_outputs["image_shared"], raw_outputs["image_raw_shared"])
    assert not torch.allclose(encoder_outputs["image_shared"], raw_outputs["image_shared"])


def test_raw_linear_readouts_can_be_selected():
    torch.manual_seed(0)
    base_config = default_bridge_config()
    model = PerturbJEPABridge(
        replace(
            base_config,
            rna_condition_readout="raw_linear_pseudobulk",
            rna_pseudobulk_normalize=False,
            image_condition_readout="raw_linear_pooled",
            image_raw_normalize=False,
        )
    )
    model.eval()
    batch = make_synthetic_bridge_batch(batch_size=2)

    outputs = forward_batch(model, batch)

    assert "rna_raw_linear_shared" in outputs
    assert "image_raw_linear_shared" in outputs
    torch.testing.assert_close(outputs["rna_shared"], outputs["rna_raw_linear_shared"])
    torch.testing.assert_close(outputs["image_shared"], outputs["image_raw_linear_shared"])


def test_experiment_config_loads_rna_condition_readout():
    payload = {
        "model": {
            "rna_condition_readout": "encoder_plus_pseudobulk",
            "rna_pseudobulk_normalize": False,
            "image_condition_readout": "encoder_plus_raw_pooled",
            "image_raw_normalize": False,
            "rna": {"pooling": "mean_tokens"},
            "image": {"pooling": "mean_patches"},
        }
    }

    config = ExperimentConfig.from_dict(payload)

    assert config.model.rna_condition_readout == "encoder_plus_pseudobulk"
    assert config.model.rna_pseudobulk_normalize is False
    assert config.model.image_condition_readout == "encoder_plus_raw_pooled"
    assert config.model.image_raw_normalize is False
    assert config.model.rna.pooling == "mean_tokens"
    assert config.model.image.pooling == "mean_patches"
