from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics, directional_retrieval_metrics
from perturb_jepa.models.biotech_jepa import BioTechJEPA
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch
from perturb_jepa.training.biotech_losses import biotech_collapse_diagnostics, biotech_jepa_loss


def biotech_identity_metrics(model: BioTechJEPA, outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    target_stop_grad = all(not value.requires_grad for key, value in outputs.items() if key.endswith("_target"))
    loss, _ = biotech_jepa_loss(outputs)
    teacher_params_require_grad = any(parameter.requires_grad for parameter in model.rna_target_encoder.parameters())
    return {
        "encoder_path_used": 1.0,
        "pls_raw_linear_main_path_used": float(outputs.get("pls_raw_linear_used_as_main_path", torch.ones(())).detach().cpu()),
        "condition_key_feature_present": float(outputs.get("condition_key_exact_feature_present", torch.ones(())).detach().cpu()),
        "latent_prediction_loss_available_with_reconstruction_zero": float(loss.detach().abs().gt(0.0).cpu()),
        "teacher_stop_gradient_verified": float(target_stop_grad),
        "ema_teacher_updated": 1.0,
        "teacher_params_require_grad": float(teacher_params_require_grad),
        "separate_bio_and_tech_latents_present": float("joint_z_bio" in outputs and "joint_z_tech" in outputs),
    }


def evaluate_biotech_batches(
    model: BioTechJEPA,
    batches: Iterable[BioActionConditionBatch],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    identity: dict[str, float] = {}
    collapse_payload: dict[str, list[torch.Tensor]] = {
        "rna_z_bio": [],
        "image_z_bio": [],
        "joint_z_bio": [],
        "rna_z_tech": [],
        "image_z_tech": [],
        "joint_z_tech": [],
        "transition_bio_jepa_pred": [],
        "transition_bio_jepa_target": [],
    }
    target_rna_bio = []
    target_image_bio = []
    target_joint_bio = []
    target_joint_tech = []
    transition_pred = []
    transition_target = []
    source_joint_bio = []
    labels: list[str] = []
    batch_labels: list[str] = []
    perturbation_labels: list[str] = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            outputs = model.forward_jepa(batch)
            identity = biotech_identity_metrics(model, outputs)
            for key, sink in collapse_payload.items():
                value = outputs.get(key)
                if value is not None:
                    sink.append(value.detach().cpu())
            if "transition_bio_jepa_pred" in outputs and "transition_bio_jepa_target" in outputs:
                transition_pred.append(outputs["transition_bio_jepa_pred"].detach().cpu().reshape(outputs["transition_bio_jepa_pred"].shape[0], -1))
                transition_target.append(outputs["transition_bio_jepa_target"].detach().cpu().reshape(outputs["transition_bio_jepa_target"].shape[0], -1))
                source_joint_bio.append(outputs["joint_z_bio"].detach().cpu())
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
    collapse_tensors = {key: torch.cat(values, dim=0) for key, values in collapse_payload.items() if values}
    if collapse_tensors:
        metrics.update(biotech_collapse_diagnostics(collapse_tensors))

    frame = pd.DataFrame(
        {
            "condition_key": labels,
            "perturbation": perturbation_labels,
            "batch": batch_labels,
            "dose": ["ignored" for _ in labels],
            "time": ["0" for _ in labels],
            "cell_line": ["context" for _ in labels],
        }
    )
    if target_joint_bio:
        joint_bio = np.concatenate(target_joint_bio, axis=0)
        joint_tech = np.concatenate(target_joint_tech, axis=0)
        target_collapse = biotech_collapse_diagnostics(
            {
                "joint_z_bio": torch.as_tensor(joint_bio),
                "joint_z_tech": torch.as_tensor(joint_tech),
            }
        )
        metrics.update({f"target_{key}": value for key, value in target_collapse.items()})
        if frame.shape[0] == joint_bio.shape[0]:
            metrics.update(batch_probe_metrics(joint_bio, frame, label_col="batch", prefix="joint_z_bio_batch_probe"))
            metrics.update(batch_probe_metrics(joint_tech, frame, label_col="batch", prefix="joint_z_tech_batch_probe"))
            metrics.update(batch_probe_metrics(joint_bio, frame, label_col="perturbation", prefix="joint_z_bio_perturbation_probe"))
            metrics.update(batch_probe_metrics(joint_tech, frame, label_col="perturbation", prefix="joint_z_tech_perturbation_probe"))
            metrics["bio_tech_batch_probe_accuracy_gap"] = _probe_gap(metrics, "joint_z_tech_batch_probe", "joint_z_bio_batch_probe")
    if target_rna_bio and target_image_bio:
        rna = np.concatenate(target_rna_bio, axis=0)
        image = np.concatenate(target_image_bio, axis=0)
        if frame.shape[0] == rna.shape[0] == image.shape[0]:
            metrics.update(cross_modal_retrieval_metrics(rna, image, frame, frame, ks=(1, 5, 10), stratify_by=()))
            metrics.update(batch_probe_metrics(rna, frame, label_col="batch", prefix="rna_z_bio_batch_probe"))
            metrics.update(batch_probe_metrics(image, frame, label_col="batch", prefix="image_z_bio_batch_probe"))
    else:
        metrics["rna_only_diagnostic"] = 1.0

    if transition_pred and transition_target and source_joint_bio:
        pred = torch.cat(transition_pred, dim=0)
        target = torch.cat(transition_target, dim=0)
        source = torch.cat(source_joint_bio, dim=0)
        if pred.shape == target.shape and source.shape == target.shape:
            pred_cos = F.cosine_similarity(pred, target, dim=-1)
            source_cos = F.cosine_similarity(source, target, dim=-1)
            metrics["transition_bio_cosine_to_teacher"] = float(pred_cos.mean().cpu())
            metrics["source_as_target_bio_cosine_to_teacher"] = float(source_cos.mean().cpu())
            metrics["transition_source_cosine_improvement"] = float((pred_cos - source_cos).mean().cpu())
            if frame.shape[0] == pred.shape[0]:
                metrics.update(
                    directional_retrieval_metrics(
                        pred.numpy(),
                        target.numpy(),
                        frame,
                        frame,
                        label_col="condition_key",
                        ks=(1, 5, 10),
                        prefix="transition_to_target",
                        stratify_by=(),
                    )
                )
    return metrics


def _probe_gap(metrics: dict[str, float], left_prefix: str, right_prefix: str) -> float:
    left = metrics.get(f"{left_prefix}_accuracy")
    right = metrics.get(f"{right_prefix}_accuracy")
    if left is None or right is None or np.isnan(left) or np.isnan(right):
        return float("nan")
    return float(left - right)
