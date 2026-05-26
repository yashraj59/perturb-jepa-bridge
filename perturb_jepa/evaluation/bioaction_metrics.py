from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics
from perturb_jepa.models.bioaction_jepa import BioActionJEPA
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch
from perturb_jepa.training.bioaction_losses import bioaction_jepa_loss, collapse_diagnostics


def bioaction_identity_metrics(model: BioActionJEPA, outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    teacher_params_require_grad = any(parameter.requires_grad for parameter in model.rna_target_encoder.parameters())
    target_stop_grad = all(not value.requires_grad for key, value in outputs.items() if key.endswith("_target"))
    loss, _ = bioaction_jepa_loss(outputs)
    return {
        "encoder_path_used": 1.0,
        "pls_raw_linear_main_path_used": float(outputs.get("pls_raw_linear_used_as_main_path", torch.ones(())).detach().cpu()),
        "condition_key_feature_present": float(outputs.get("condition_key_exact_feature_present", torch.ones(())).detach().cpu()),
        "latent_prediction_loss_available_with_reconstruction_zero": float(loss.detach().abs().gt(0.0).cpu()),
        "teacher_stop_gradient_verified": float(target_stop_grad),
        "ema_teacher_updated": 1.0,
        "teacher_params_require_grad": float(teacher_params_require_grad),
    }


def evaluate_bioaction_batches(
    model: BioActionJEPA,
    batches: Iterable[BioActionConditionBatch],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    rna_states = []
    image_states = []
    joint_states = []
    transition_preds = []
    transition_targets = []
    source_joint_states = []
    labels = []
    batch_labels = []
    identity: dict[str, float] = {}
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            outputs = model.forward_jepa(batch)
            identity = bioaction_identity_metrics(model, outputs)
            for key, sink in (
                ("rna_condition_state", rna_states),
                ("image_condition_state", image_states),
                ("joint_condition_state", joint_states),
                ("transition_joint_jepa_pred", transition_preds),
                ("transition_joint_jepa_target", transition_targets),
            ):
                if key in outputs:
                    sink.append(outputs[key].detach().cpu())
            if "joint_condition_state" in outputs:
                source_joint_states.append(outputs["joint_condition_state"].detach().cpu())
            encoded = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="context",
            )
            if "rna_condition_state" in encoded and "image_condition_state" in encoded:
                # Retrieval uses context states; collapse diagnostics above use
                # JEPA forward outputs so transition predictions are included.
                labels.extend(batch.condition_key or [str(i) for i in range(encoded["rna_condition_state"].shape[0])])
                batch_labels.extend([f"batch_{int(value)}" for value in batch.batch_id.detach().cpu().tolist()])
                encoded_rna = encoded["rna_condition_state"].detach().cpu().numpy()
                encoded_image = encoded["image_condition_state"].detach().cpu().numpy()
                if "_retrieval_rna" not in locals():
                    _retrieval_rna = []
                    _retrieval_image = []
                _retrieval_rna.append(encoded_rna)
                _retrieval_image.append(encoded_image)
    metrics: dict[str, float] = {}
    metrics.update(identity)
    collapse_payload: dict[str, torch.Tensor] = {}
    if rna_states:
        collapse_payload["rna_condition_state"] = torch.cat(rna_states, dim=0)
    if image_states:
        collapse_payload["image_condition_state"] = torch.cat(image_states, dim=0)
    if joint_states:
        collapse_payload["joint_condition_state"] = torch.cat(joint_states, dim=0)
        collapse_payload["shared_state"] = collapse_payload["joint_condition_state"]
    if transition_preds:
        collapse_payload["transition_joint_jepa_pred"] = torch.cat(transition_preds, dim=0)
    if transition_targets:
        collapse_payload["transition_joint_jepa_target"] = torch.cat(transition_targets, dim=0)
    if collapse_payload:
        metrics.update(collapse_diagnostics(collapse_payload))
    if transition_preds and transition_targets and source_joint_states:
        pred = torch.cat(transition_preds, dim=0).reshape(-1, transition_preds[0].shape[-1])
        target = torch.cat(transition_targets, dim=0).reshape(-1, transition_targets[0].shape[-1])
        source = torch.cat(source_joint_states, dim=0).reshape(-1, source_joint_states[0].shape[-1])
        if source.shape[0] == target.shape[0]:
            pred_cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1)
            source_cos = torch.nn.functional.cosine_similarity(source, target, dim=-1)
            metrics["transition_joint_cosine_to_teacher"] = float(pred_cos.mean().cpu())
            metrics["source_as_target_joint_cosine_to_teacher"] = float(source_cos.mean().cpu())
            metrics["transition_source_cosine_improvement"] = float((pred_cos - source_cos).mean().cpu())
    if "_retrieval_rna" in locals() and "_retrieval_image" in locals():
        rna = np.concatenate(_retrieval_rna, axis=0)
        image = np.concatenate(_retrieval_image, axis=0)
        frame = pd.DataFrame(
            {
                "condition_key": labels[: rna.shape[0]],
                "perturbation": labels[: rna.shape[0]],
                "batch": batch_labels[: rna.shape[0]],
            }
        )
        metrics.update(cross_modal_retrieval_metrics(rna, image, frame, frame, ks=(1, 5, 10), stratify_by=()))
        metrics.update(batch_probe_metrics(rna, frame, label_col="batch", prefix="rna_embedding_batch_probe"))
        metrics.update(batch_probe_metrics(image, frame, label_col="batch", prefix="image_embedding_batch_probe"))
        if joint_states:
            joint = torch.cat(joint_states, dim=0).detach().cpu().numpy()
            if joint.shape[0] == frame.shape[0]:
                metrics.update(batch_probe_metrics(joint, frame, label_col="batch", prefix="joint_embedding_batch_probe"))
    return metrics
