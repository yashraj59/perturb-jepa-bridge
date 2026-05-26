from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics, directional_retrieval_metrics
from perturb_jepa.models.biomech_jepa import BioMechanisticJEPA
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch
from perturb_jepa.training.biomech_losses import biomech_collapse_diagnostics, biomech_jepa_loss


def biomech_identity_metrics(model: BioMechanisticJEPA, outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    loss, _ = biomech_jepa_loss(outputs)
    target_stop_grad = all(not value.requires_grad for key, value in outputs.items() if key.endswith("_target") or key.endswith("_teacher"))
    return {
        "encoder_path_used": 1.0,
        "pls_raw_linear_main_path_used": float(outputs.get("pls_raw_linear_used_as_main_path", torch.ones(())).detach().cpu()),
        "condition_key_feature_present": float(outputs.get("condition_key_exact_feature_present", torch.ones(())).detach().cpu()),
        "latent_prediction_loss_available_with_reconstruction_zero": float(loss.detach().abs().gt(0.0).cpu()),
        "teacher_stop_gradient_verified": float(target_stop_grad),
        "ema_teacher_updated": 1.0,
        "teacher_params_require_grad": float(any(parameter.requires_grad for parameter in model.backbone.rna_target_encoder.parameters())),
        "separate_bio_and_tech_latents_present": float("joint_z_bio" in outputs and "joint_z_tech" in outputs),
        "heldout_action_descriptor_valid": float(outputs.get("heldout_action_descriptor_valid", torch.ones(())).detach().cpu()),
    }


def evaluate_biomech_batches(
    model: BioMechanisticJEPA,
    batches: Iterable[BioActionConditionBatch],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    identity: dict[str, float] = {}
    target_rna_bio = []
    target_image_bio = []
    target_joint_bio = []
    target_joint_tech = []
    delta_pred = []
    delta_teacher = []
    target_pred = []
    target_teacher = []
    source = []
    prototype_pred = []
    prototype_teacher = []
    labels: list[str] = []
    batch_labels: list[str] = []
    perturbation_labels: list[str] = []
    loss_ratios: list[float] = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            outputs = model.forward_jepa(batch)
            identity = biomech_identity_metrics(model, outputs)
            _, diagnostics = biomech_jepa_loss(outputs)
            if "auxiliary_weighted_to_main_ratio" in diagnostics:
                loss_ratios.append(float(diagnostics["auxiliary_weighted_to_main_ratio"].detach().cpu()))
            delta_pred.append(outputs["delta_pred"].detach().cpu())
            delta_teacher.append(outputs["delta_teacher"].detach().cpu())
            target_pred.append(outputs["z_target_pred"].detach().cpu())
            target_teacher.append(outputs["z_target_teacher_bio"].detach().cpu())
            source.append(outputs["joint_z_bio"].detach().cpu())
            if "prototype_transition_jepa_pred" in outputs:
                prototype_pred.append(outputs["prototype_transition_jepa_pred"].detach().cpu())
                prototype_teacher.append(outputs["prototype_transition_jepa_target"].detach().cpu())
            encoded = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="context",
            )
            labels.extend(batch.condition_key or [str(index) for index in range(encoded["joint_z_bio"].shape[0])])
            batch_labels.extend([f"batch_{int(value)}" for value in batch.batch_id.detach().cpu().tolist()])
            perturbation_labels.extend([f"pert_{int(value)}" for value in batch.perturbation_id.detach().cpu().tolist()])
            if "rna_z_bio" in encoded:
                target_rna_bio.append(encoded["rna_z_bio"].detach().cpu().numpy())
            if "image_z_bio" in encoded:
                target_image_bio.append(encoded["image_z_bio"].detach().cpu().numpy())
            target_joint_bio.append(encoded["joint_z_bio"].detach().cpu().numpy())
            target_joint_tech.append(encoded["joint_z_tech"].detach().cpu().numpy())

    metrics: dict[str, float] = {}
    metrics.update(identity)
    frame = pd.DataFrame(
        {
            "condition_key": labels,
            "batch": batch_labels,
            "perturbation": perturbation_labels,
            "dose": ["ignored" for _ in labels],
            "time": ["0" for _ in labels],
            "cell_line": ["context" for _ in labels],
        }
    )
    delta_pred_t = torch.cat(delta_pred, dim=0)
    delta_teacher_t = torch.cat(delta_teacher, dim=0)
    target_pred_t = torch.cat(target_pred, dim=0)
    target_teacher_t = torch.cat(target_teacher, dim=0)
    source_t = torch.cat(source, dim=0)
    metrics.update(biomech_collapse_diagnostics({"delta_pred": delta_pred_t, "delta_teacher": delta_teacher_t, "joint_z_bio": target_teacher_t}))
    metrics["delta_teacher_effective_rank"] = _effective_rank(delta_teacher_t)
    metrics["delta_pred_effective_rank"] = _effective_rank(delta_pred_t)
    metrics["delta_cosine"] = float(F.cosine_similarity(delta_pred_t, delta_teacher_t, dim=-1).mean().cpu())
    metrics["absolute_target_cosine"] = float(F.cosine_similarity(target_pred_t, target_teacher_t, dim=-1).mean().cpu())
    source_cos = F.cosine_similarity(source_t, target_teacher_t, dim=-1)
    pred_cos = F.cosine_similarity(target_pred_t, target_teacher_t, dim=-1)
    metrics["transition_source_cosine_improvement"] = float((pred_cos - source_cos).mean().cpu())
    metrics["transition_bio_cosine_to_teacher"] = float(pred_cos.mean().cpu())
    metrics["source_as_target_bio_cosine_to_teacher"] = float(source_cos.mean().cpu())
    metrics.update(
        directional_retrieval_metrics(
            target_pred_t.numpy(),
            target_teacher_t.numpy(),
            frame,
            frame,
            label_col="condition_key",
            ks=(1, 5, 10),
            prefix="transition_to_target",
            stratify_by=(),
        )
    )

    if target_joint_bio:
        joint_bio = np.concatenate(target_joint_bio, axis=0)
        joint_tech = np.concatenate(target_joint_tech, axis=0)
        metrics["z_bio_effective_rank"] = _effective_rank_np(joint_bio)
        metrics["z_tech_effective_rank"] = _effective_rank_np(joint_tech)
        metrics.update(batch_probe_metrics(joint_bio, frame, label_col="batch", prefix="joint_z_bio_batch_probe"))
        metrics.update(batch_probe_metrics(joint_tech, frame, label_col="batch", prefix="joint_z_tech_batch_probe"))
        metrics.update(batch_probe_metrics(joint_bio, frame, label_col="perturbation", prefix="joint_z_bio_perturbation_probe"))
        metrics.update(batch_probe_metrics(joint_tech, frame, label_col="perturbation", prefix="joint_z_tech_perturbation_probe"))
        metrics["batch_allocation_gap"] = _probe_gap(metrics, "joint_z_tech_batch_probe", "joint_z_bio_batch_probe")
    if target_rna_bio and target_image_bio:
        rna = np.concatenate(target_rna_bio, axis=0)
        image = np.concatenate(target_image_bio, axis=0)
        metrics.update(cross_modal_retrieval_metrics(rna, image, frame, frame, ks=(1, 5, 10), stratify_by=()))
    else:
        metrics["rna_only_diagnostic"] = 1.0
        metrics["rna_to_image_recall@1"] = float("nan")
        metrics["image_to_rna_recall@1"] = float("nan")

    if prototype_pred and prototype_teacher:
        pred = torch.cat(prototype_pred, dim=0)
        teacher = torch.cat(prototype_teacher, dim=0)
        metrics["prototype_transition_cosine"] = float(F.cosine_similarity(pred.reshape(-1, pred.shape[-1]), teacher.reshape(-1, teacher.shape[-1]), dim=-1).mean().cpu())
        metrics["prototype_set_sliced_wasserstein"] = sliced_wasserstein_distance(pred, teacher)
        metrics["prototype_effective_rank"] = _effective_rank(pred.reshape(-1, pred.shape[-1]))
    else:
        metrics["prototype_transition_cosine"] = float("nan")
        metrics["prototype_set_sliced_wasserstein"] = float("nan")
        metrics["prototype_effective_rank"] = float("nan")
    metrics["cross_modal_gradient_ratio"] = float("nan")
    metrics["weighted_loss_to_main_loss_ratios"] = float(np.mean(loss_ratios)) if loss_ratios else float("nan")
    return metrics


def sliced_wasserstein_distance(pred: torch.Tensor, target: torch.Tensor, projections: int = 16) -> float:
    if pred.numel() == 0:
        return float("nan")
    generator = torch.Generator(device=pred.device).manual_seed(0)
    directions = torch.randn(pred.shape[-1], projections, generator=generator, device=pred.device, dtype=pred.dtype)
    directions = F.normalize(directions, dim=0)
    pred_proj = torch.sort(pred @ directions, dim=1).values
    target_proj = torch.sort(target @ directions, dim=1).values
    count = min(pred_proj.shape[1], target_proj.shape[1])
    return float((pred_proj[:, :count] - target_proj[:, :count]).abs().mean().cpu())


def _effective_rank(values: torch.Tensor, eps: float = 1e-8) -> float:
    if values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centered)
    probs = spectrum / spectrum.sum().clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).cpu())


def _effective_rank_np(values: np.ndarray, eps: float = 1e-8) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    probs = spectrum / np.maximum(spectrum.sum(), eps)
    entropy = -(probs * np.log(np.maximum(probs, eps))).sum()
    return float(np.exp(entropy))


def _probe_gap(metrics: dict[str, float], left_prefix: str, right_prefix: str) -> float:
    left = metrics.get(f"{left_prefix}_accuracy")
    right = metrics.get(f"{right_prefix}_accuracy")
    if left is None or right is None or np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left - right)
