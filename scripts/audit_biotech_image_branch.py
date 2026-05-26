from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics, retrieval_ranks
from perturb_jepa.models.biotech_jepa import BioTechJEPA
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.biotech_losses import biotech_collapse_diagnostics, biotech_jepa_loss
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.evaluate_biotech_jepa import _config_from_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit BioTech-JEPA image branch health before Phase 3 architecture work.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt"))
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bag-size", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/autoresearch_biomech_jepa_phase3/diagnostics/image_branch_health"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    payload = torch.load(args.checkpoint, map_location=device)
    model = BioTechJEPA(_config_from_payload(payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    random_model = BioTechJEPA(_config_from_payload(payload["config"])).to(device)

    eval_batches = list(
        iter_bioaction_condition_batches(
            dataset,
            split=args.eval_split,
            batch_size=args.batch_size,
            bag_size=args.bag_size,
            steps=args.eval_steps,
            seed=args.seed + 1000,
            device=device,
        )
    )
    trained_metrics = _collect_image_metrics(model, eval_batches, device=device, prefix="trained")
    random_metrics = _collect_image_metrics(random_model, eval_batches, device=device, prefix="random")
    gradient_metrics = _gradient_metrics(model, dataset, args, device=device)
    metrics = {**trained_metrics, **random_metrics, **gradient_metrics}
    metrics["decision_label"] = _decision_label(metrics)
    metrics["dataset"] = args.dataset
    metrics["eval_split"] = args.eval_split
    metrics["checkpoint"] = str(args.checkpoint)
    (args.output_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(args.output_dir / "REPORT.md", metrics)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _collect_image_metrics(model: BioTechJEPA, batches, *, device: torch.device, prefix: str) -> dict[str, Any]:
    model.eval()
    online_rna = []
    online_image = []
    teacher_rna = []
    teacher_image = []
    teacher_image_tech = []
    image_region_tokens = []
    labels: list[str] = []
    batch_labels: list[str] = []
    image_pixel_var = []
    with torch.no_grad():
        for batch in batches:
            batch = batch.to_device(device)
            encoded_online = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="context",
            )
            encoded_teacher = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="target",
            )
            online_rna.append(encoded_online["rna_z_bio"].detach().cpu().numpy())
            online_image.append(encoded_online["image_z_bio"].detach().cpu().numpy())
            teacher_rna.append(encoded_teacher["rna_z_bio"].detach().cpu().numpy())
            teacher_image.append(encoded_teacher["image_z_bio"].detach().cpu().numpy())
            teacher_image_tech.append(encoded_teacher["image_z_tech"].detach().cpu().numpy())
            image_region_tokens.append(encoded_teacher["image_region_bio_tokens"].detach().cpu())
            labels.extend(batch.condition_key or [str(idx) for idx in range(encoded_online["rna_z_bio"].shape[0])])
            batch_labels.extend([f"batch_{int(value)}" for value in batch.batch_id.detach().cpu().tolist()])
            if batch.target_images is not None:
                image_pixel_var.append(float(batch.target_images.detach().float().var().cpu()))
    frame = pd.DataFrame({"condition_key": labels, "batch": batch_labels, "perturbation": labels})
    metrics: dict[str, Any] = {}
    for tag, rna_values, image_values in (
        ("online", np.concatenate(online_rna, axis=0), np.concatenate(online_image, axis=0)),
        ("teacher", np.concatenate(teacher_rna, axis=0), np.concatenate(teacher_image, axis=0)),
    ):
        retrieval = cross_modal_retrieval_metrics(rna_values, image_values, frame, frame, ks=(1, 5, 10), stratify_by=())
        metrics.update({f"{prefix}/{tag}/{key}": value for key, value in retrieval.items()})
        metrics[f"{prefix}/{tag}/image_to_rna_rank_median"] = float(
            np.median(retrieval_ranks(image_values, rna_values, frame["condition_key"].tolist(), frame["condition_key"].tolist()))
        )
        metrics[f"{prefix}/{tag}/rna_to_image_rank_median"] = float(
            np.median(retrieval_ranks(rna_values, image_values, frame["condition_key"].tolist(), frame["condition_key"].tolist()))
        )
    teacher_image_array = np.concatenate(teacher_image, axis=0)
    teacher_image_tech_array = np.concatenate(teacher_image_tech, axis=0)
    metrics[f"{prefix}/teacher/image_z_bio_effective_rank"] = _effective_rank_np(teacher_image_array)
    metrics[f"{prefix}/teacher/image_z_tech_effective_rank"] = _effective_rank_np(teacher_image_tech_array)
    metrics[f"{prefix}/teacher/image_z_bio_std_mean"] = float(np.std(teacher_image_array, axis=0).mean())
    metrics[f"{prefix}/teacher/image_z_tech_std_mean"] = float(np.std(teacher_image_tech_array, axis=0).mean())
    region = torch.cat(image_region_tokens, dim=0)
    metrics[f"{prefix}/teacher/image_region_target_variance"] = float(region.reshape(-1, region.shape[-1]).var(dim=0, unbiased=False).mean().cpu())
    collapse = biotech_collapse_diagnostics(
        {
            "image_z_bio": torch.as_tensor(teacher_image_array),
            "image_z_tech": torch.as_tensor(teacher_image_tech_array),
        }
    )
    metrics.update({f"{prefix}/teacher/{key}": value for key, value in collapse.items()})
    metrics[f"{prefix}/image_pixel_variance_mean"] = float(np.mean(image_pixel_var)) if image_pixel_var else float("nan")
    return metrics


def _gradient_metrics(model: BioTechJEPA, dataset, args: argparse.Namespace, *, device: torch.device) -> dict[str, float]:
    model.train()
    batch = next(
        iter_bioaction_condition_batches(
            dataset,
            split="train",
            batch_size=2,
            bag_size=args.bag_size,
            steps=1,
            seed=args.seed + 17,
            device=device,
        )
    )
    model.zero_grad(set_to_none=True)
    outputs = model.forward_jepa(batch)
    loss, diagnostics = biotech_jepa_loss(outputs)
    loss.backward()
    groups = {
        "rna_branch": ("rna_context_encoder", "rna_context_projector", "rna_bio_head"),
        "image_branch": ("image_context_encoder", "image_context_projector", "image_bio_head"),
        "cross_modal_predictor": ("cross_modal_predictor",),
        "transition_predictor": ("transition_predictor",),
    }
    metrics: dict[str, float] = {}
    for group, prefixes in groups.items():
        total = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if any(name.startswith(prefix) for prefix in prefixes):
                total += float(parameter.grad.detach().norm().cpu())
        metrics[f"gradient/{group}_norm"] = total
    metrics["gradient/image_to_rna_branch_ratio"] = metrics["gradient/image_branch_norm"] / max(metrics["gradient/rna_branch_norm"], 1e-8)
    weighted_cross = float(diagnostics.get("weighted/rna_to_image_jepa", torch.zeros(())).cpu()) + float(
        diagnostics.get("weighted/image_to_rna_jepa", torch.zeros(())).cpu()
    )
    weighted_transition = float(diagnostics.get("weighted/transition_bio_jepa", torch.ones(())).cpu())
    weighted_total = float(diagnostics.get("loss/total", torch.ones(())).cpu())
    metrics["loss/cross_modal_weighted_to_transition_ratio"] = weighted_cross / max(weighted_transition, 1e-8)
    metrics["loss/cross_modal_weighted_to_total_ratio"] = weighted_cross / max(weighted_total, 1e-8)
    return metrics


def _decision_label(metrics: dict[str, Any]) -> str:
    if metrics.get("trained/image_pixel_variance_mean", 0.0) < 1e-5:
        return "IMAGE_BRANCH_AUDIT_DATA_OR_LOADER_ISSUE"
    if metrics.get("trained/teacher/image_z_bio_effective_rank", 0.0) < 2.0 or metrics.get("trained/teacher/image_z_bio_std_mean", 0.0) < 0.02:
        return "IMAGE_BRANCH_AUDIT_COLLAPSE"
    if metrics.get("gradient/image_to_rna_branch_ratio", 0.0) < 0.25 or metrics.get("loss/cross_modal_weighted_to_total_ratio", 0.0) < 0.15:
        return "IMAGE_BRANCH_AUDIT_LOSS_IMBALANCE"
    return "IMAGE_BRANCH_AUDIT_HEALTHY"


def _effective_rank_np(values: np.ndarray, eps: float = 1e-8) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    probs = spectrum / np.maximum(spectrum.sum(), eps)
    entropy = -(probs * np.log(np.maximum(probs, eps))).sum()
    return float(np.exp(entropy))


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Image Branch Health Audit",
        "",
        f"Decision label: `{metrics['decision_label']}`",
        "",
        "## Key Metrics",
        "",
        f"- Trained online image->RNA recall@1: `{metrics.get('trained/online/image_to_rna_recall@1', float('nan')):.4f}`",
        f"- Trained teacher image->RNA recall@1: `{metrics.get('trained/teacher/image_to_rna_recall@1', float('nan')):.4f}`",
        f"- Trained online RNA->image recall@1: `{metrics.get('trained/online/rna_to_image_recall@1', float('nan')):.4f}`",
        f"- Random online image->RNA recall@1: `{metrics.get('random/online/image_to_rna_recall@1', float('nan')):.4f}`",
        f"- Image teacher `z_bio` effective rank: `{metrics.get('trained/teacher/image_z_bio_effective_rank', float('nan')):.4f}`",
        f"- Image teacher `z_tech` effective rank: `{metrics.get('trained/teacher/image_z_tech_effective_rank', float('nan')):.4f}`",
        f"- Image region target variance: `{metrics.get('trained/teacher/image_region_target_variance', float('nan')):.6f}`",
        f"- Image/RNA branch gradient norm ratio: `{metrics.get('gradient/image_to_rna_branch_ratio', float('nan')):.4f}`",
        f"- Cross-modal weighted/total loss ratio: `{metrics.get('loss/cross_modal_weighted_to_total_ratio', float('nan')):.4f}`",
        "",
        "## Interpretation",
        "",
        "This audit distinguishes loader/data failure, latent collapse, and loss/gradient imbalance before any Phase 3 architecture change.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
