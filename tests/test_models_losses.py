import torch

from scripts.train_smoke import build_smoke_model
from perturb_jepa.losses import bridge_loss
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import forward_batch, train_step


def test_bridge_forward_and_loss_are_finite():
    torch.manual_seed(0)
    model = build_smoke_model()
    batch = make_synthetic_bridge_batch(batch_size=2)
    outputs = forward_batch(model, batch)
    assert outputs["rna_shared"].shape == outputs["image_shared"].shape
    assert outputs["rna_perturbation_logits"].shape[0] == batch.gene_ids.shape[0]
    assert outputs["image_batch_logits"].shape[0] == batch.images.shape[0]
    image_patches = patchify(batch.images, model.config.image.patch_size)
    total, terms = bridge_loss(
        outputs,
        rna_values=batch.expression_values,
        rna_mask=batch.rna_token_mask,
        image_patches=image_patches,
        image_patch_mask=batch.image_patch_mask,
        perturbation_id=batch.perturbation_id,
        batch_id=batch.batch_id,
    )
    assert torch.isfinite(total)
    assert "align" in terms
    assert "rna_perturbation_cls" in terms
    assert "image_batch_adv" in terms


def test_train_step_updates_without_error():
    torch.manual_seed(0)
    model = build_smoke_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_synthetic_bridge_batch(batch_size=2)
    terms = train_step(model, optimizer, batch)
    assert terms["total"] > 0
