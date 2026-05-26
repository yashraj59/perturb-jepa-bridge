from __future__ import annotations

import torch

from perturb_jepa.config import default_bridge_config
from perturb_jepa.models.bridge import PerturbJEPABridge
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import _alignment_bio_keys, loss_for_batch


def test_alignment_bio_keys_are_condition_and_perturbation_labels():
    batch = make_synthetic_bridge_batch(batch_size=4, genes=8)
    keys = _alignment_bio_keys(batch)
    assert set(keys) == {"condition", "perturbation"}
    assert keys["condition"].shape == (4,)
    assert torch.equal(keys["perturbation"], batch.perturbation_id)


def test_loss_for_batch_accepts_multi_positive_alignment():
    model = PerturbJEPABridge(default_bridge_config())
    batch = make_synthetic_bridge_batch()
    total, terms = loss_for_batch(model, batch, multi_positive_alignment=True)
    assert torch.isfinite(total)
    assert "align" in terms

