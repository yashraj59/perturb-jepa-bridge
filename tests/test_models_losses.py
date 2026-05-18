import torch

from scripts.train_smoke import build_smoke_model
from perturb_jepa.losses import bridge_loss, masked_cosine_jepa_loss, multi_resolution_info_nce_loss, sliced_wasserstein_loss
from perturb_jepa.models.image_encoder import patchify
from perturb_jepa.training.synthetic import make_synthetic_bridge_batch
from perturb_jepa.training.trainer import forward_batch, train_step


def test_bridge_forward_and_loss_are_finite():
    torch.manual_seed(0)
    model = build_smoke_model()
    batch = make_synthetic_bridge_batch(batch_size=2)
    outputs = forward_batch(model, batch)
    assert outputs["rna_shared"].shape == outputs["image_shared"].shape
    assert torch.equal(outputs["rna_retrieval"], outputs["rna_shared"])
    assert torch.equal(outputs["image_retrieval"], outputs["image_shared"])
    assert outputs["rna_perturbation_logits"].shape[0] == batch.gene_ids.shape[0]
    assert outputs["rna_state_perturbation_logits"].shape[0] == batch.gene_ids.shape[0]
    assert outputs["image_batch_logits"].shape[0] == batch.images.shape[0]
    assert outputs["counterfactual_gate"].shape == outputs["counterfactual_delta"].shape
    assert outputs["cycle_reconstruction"].shape == outputs["z_state"].shape
    image_patches = patchify(batch.images, model.config.image.patch_size)
    bag_labels = torch.tensor([0, 0], dtype=torch.long)
    total, terms = bridge_loss(
        outputs,
        rna_values=batch.expression_values,
        rna_mask=batch.rna_token_mask,
        image_patches=image_patches,
        image_patch_mask=batch.image_patch_mask,
        perturbation_id=batch.perturbation_id,
        batch_id=batch.batch_id,
        bag_labels=bag_labels,
        hierarchy_labels=[bag_labels, batch.perturbation_id],
    )
    assert torch.isfinite(total)
    assert "align" in terms
    assert "sliced_wasserstein" in terms
    assert "cycle" in terms
    assert "response_bottleneck" in terms
    assert "rna_perturbation_cls" in terms
    assert "rna_state_perturbation_adv" in terms
    assert "image_batch_adv" in terms
    assert "counterfactual_image" not in terms

    _, explicit_terms = bridge_loss(
        outputs,
        counterfactual_image_target=outputs["counterfactual_image"].detach(),
        counterfactual_control_image=outputs["image_shared"].detach(),
    )
    assert "counterfactual_image" in explicit_terms


def test_train_step_updates_without_error():
    torch.manual_seed(0)
    model = build_smoke_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_synthetic_bridge_batch(batch_size=2)
    terms = train_step(model, optimizer, batch)
    assert terms["total"] > 0


def test_distributional_alignment_losses_are_finite():
    torch.manual_seed(0)
    x = torch.randn(6, 8)
    y = x + 0.1 * torch.randn(6, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    assert torch.isfinite(sliced_wasserstein_loss(x, y, num_projections=8))
    assert torch.isfinite(multi_resolution_info_nce_loss(x, y, [labels, labels], weights=[0.3, 0.7]))


def test_ema_teacher_modules_are_frozen_updated_and_eval_during_forward():
    torch.manual_seed(0)
    model = build_smoke_model()

    teacher_parameters = [
        *model.rna_teacher.parameters(),
        *model.image_teacher.parameters(),
        *model.rna_teacher_projection.parameters(),
        *model.image_teacher_projection.parameters(),
        *model.rna_teacher_bag_aggregator.parameters(),
        *model.image_teacher_bag_aggregator.parameters(),
    ]
    assert teacher_parameters
    assert all(not parameter.requires_grad for parameter in teacher_parameters)

    before = next(model.rna_teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(model.rna_encoder.parameters()).add_(1.0)
    model.update_teachers(decay=0.5)
    after = next(model.rna_teacher.parameters()).detach()
    assert not torch.allclose(before, after)

    training_states: list[bool] = []

    def hook(module, inputs, output):
        training_states.append(module.training)

    handle = model.rna_teacher.register_forward_hook(hook)
    try:
        model.train()
        forward_batch(model, make_synthetic_bridge_batch(batch_size=2))
    finally:
        handle.remove()
    assert training_states == [False]


def test_masked_jepa_loss_uses_only_masked_positions():
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[True, False]])

    masked = masked_cosine_jepa_loss(student, teacher, mask)
    unmasked = masked_cosine_jepa_loss(student, teacher, None)
    zero = masked_cosine_jepa_loss(student, teacher, torch.zeros_like(mask))

    assert masked.item() == 0.0
    assert unmasked.item() > masked.item()
    assert zero.item() == 0.0


def test_batch_adversary_uses_retrieval_embedding(monkeypatch):
    torch.manual_seed(0)
    model = build_smoke_model()
    recorded: list[torch.Tensor] = []
    original_forward = model.batch_adversary.forward

    def spy(x, *, scale=None):
        recorded.append(x)
        return original_forward(x, scale=scale)

    monkeypatch.setattr(model.batch_adversary, "forward", spy)
    outputs = forward_batch(model, make_synthetic_bridge_batch(batch_size=2))

    assert len(recorded) == 2
    assert torch.equal(recorded[0], outputs["rna_retrieval"])
    assert torch.equal(recorded[1], outputs["image_retrieval"])


def test_jepa_teacher_targets_are_unmasked_and_detached():
    torch.manual_seed(0)
    model = build_smoke_model()
    model.eval()
    batch = make_synthetic_bridge_batch(batch_size=2)

    masked_outputs = model(
        gene_ids=batch.gene_ids,
        expression_values=batch.expression_values,
        rna_token_mask=torch.ones_like(batch.rna_token_mask, dtype=torch.bool),
        images=batch.images,
        image_patch_mask=torch.ones_like(batch.image_patch_mask, dtype=torch.bool),
        perturbation_id=batch.perturbation_id,
        perturbation_type_id=batch.perturbation_type_id,
        cell_line_id=batch.cell_line_id,
        batch_id=batch.batch_id,
        dose=batch.dose,
        time=batch.time,
    )
    unmasked_outputs = model(
        gene_ids=batch.gene_ids,
        expression_values=batch.expression_values,
        rna_token_mask=torch.zeros_like(batch.rna_token_mask, dtype=torch.bool),
        images=batch.images,
        image_patch_mask=torch.zeros_like(batch.image_patch_mask, dtype=torch.bool),
        perturbation_id=batch.perturbation_id,
        perturbation_type_id=batch.perturbation_type_id,
        cell_line_id=batch.cell_line_id,
        batch_id=batch.batch_id,
        dose=batch.dose,
        time=batch.time,
    )

    assert not masked_outputs["rna_teacher_tokens"].requires_grad
    assert not masked_outputs["image_teacher_patches"].requires_grad
    assert torch.allclose(masked_outputs["rna_teacher_tokens"], unmasked_outputs["rna_teacher_tokens"])
    assert torch.allclose(masked_outputs["image_teacher_patches"], unmasked_outputs["image_teacher_patches"])
    assert not torch.allclose(masked_outputs["rna_tokens"], unmasked_outputs["rna_tokens"])
    assert not torch.allclose(masked_outputs["image_patches"], unmasked_outputs["image_patches"])
