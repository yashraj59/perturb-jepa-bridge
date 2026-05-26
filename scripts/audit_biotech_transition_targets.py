from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.batch_probe import batch_probe_metrics
from perturb_jepa.evaluation.retrieval import ranks_from_scores
from perturb_jepa.models.biotech_jepa import BioTechJEPA
from perturb_jepa.training.bioaction_batches import iter_bioaction_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from scripts.evaluate_biotech_jepa import _config_from_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit absolute versus delta transition targets for BioTech-JEPA.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt"))
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bag-size", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/autoresearch_biomech_jepa_phase3/diagnostics/transition_target_audit"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    payload = torch.load(args.checkpoint, map_location=device)
    model = BioTechJEPA(_config_from_payload(payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    metrics: dict[str, Any] = {}
    for split, seed_offset in (("train", 20), (args.eval_split, 1000)):
        batches = iter_bioaction_condition_batches(
            dataset,
            split=split,
            batch_size=args.batch_size,
            bag_size=args.bag_size,
            steps=args.steps,
            seed=args.seed + seed_offset,
            device=device,
        )
        metrics.update({f"{split}/{key}": value for key, value in _collect_split_metrics(model, batches).items()})
    metrics["decision_label"] = _decision_label(metrics, args.eval_split)
    metrics["dataset"] = args.dataset
    metrics["eval_split"] = args.eval_split
    metrics["checkpoint"] = str(args.checkpoint)
    (args.output_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(args.output_dir / "REPORT.md", metrics, eval_split=args.eval_split)
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _collect_split_metrics(model: BioTechJEPA, batches) -> dict[str, float]:
    controls = []
    targets = []
    deltas = []
    labels: list[str] = []
    batch_labels: list[str] = []
    with torch.no_grad():
        for batch in batches:
            control = model.encode_condition(
                gene_ids=batch.control_gene_ids,
                expression_values=batch.control_expression_values,
                images=batch.control_images,
                rna_bag_mask=batch.control_rna_bag_mask,
                image_bag_mask=batch.control_image_bag_mask,
                mode="target",
            )
            target = model.encode_condition(
                gene_ids=batch.target_gene_ids,
                expression_values=batch.target_expression_values,
                images=batch.target_images,
                rna_bag_mask=batch.target_rna_bag_mask,
                image_bag_mask=batch.target_image_bag_mask,
                mode="target",
            )
            control_z = control["joint_z_bio"].detach().cpu()
            target_z = target["joint_z_bio"].detach().cpu()
            controls.append(control_z)
            targets.append(target_z)
            deltas.append(target_z - control_z)
            labels.extend(batch.condition_key or [str(idx) for idx in range(target_z.shape[0])])
            batch_labels.extend([f"batch_{int(value)}" for value in batch.batch_id.detach().cpu().tolist()])
    control_array = torch.cat(controls, dim=0)
    target_array = torch.cat(targets, dim=0)
    delta_array = torch.cat(deltas, dim=0)
    label_array = np.asarray(labels, dtype=object)
    frame = pd.DataFrame({"condition_key": labels, "batch": batch_labels})

    source_target_cos = F.cosine_similarity(control_array, target_array, dim=-1)
    source_delta_cos = F.cosine_similarity(control_array, delta_array, dim=-1)
    delta_target_cos = F.cosine_similarity(delta_array, target_array, dim=-1)
    absolute_scores = _normalize_np(control_array.numpy()) @ _normalize_np(target_array.numpy()).T
    delta_scores = _normalize_np(delta_array.numpy()) @ _normalize_np(delta_array.numpy()).T
    absolute_ranks = ranks_from_scores(absolute_scores, label_array, label_array)
    delta_ranks = ranks_from_scores(delta_scores, label_array, label_array)
    absolute_probe = batch_probe_metrics(target_array.numpy(), frame, label_col="batch", prefix="absolute_target_batch_probe")
    delta_probe = batch_probe_metrics(delta_array.numpy(), frame, label_col="batch", prefix="delta_target_batch_probe")
    metrics = {
        "source_to_target_cosine_mean": float(source_target_cos.mean().cpu()),
        "source_to_delta_cosine_mean": float(source_delta_cos.mean().cpu()),
        "delta_to_target_cosine_mean": float(delta_target_cos.mean().cpu()),
        "absolute_target_nn_median_rank": float(np.median(absolute_ranks)),
        "absolute_target_nn_recall@1": float(np.mean(absolute_ranks <= 1)),
        "delta_target_nn_median_rank": float(np.median(delta_ranks)),
        "delta_target_nn_recall@1": float(np.mean(delta_ranks <= 1)),
        "absolute_target_effective_rank": _effective_rank_torch(target_array),
        "delta_teacher_effective_rank": _effective_rank_torch(delta_array),
        "absolute_target_std_mean": float(target_array.std(dim=0, unbiased=False).mean().cpu()),
        "delta_teacher_std_mean": float(delta_array.std(dim=0, unbiased=False).mean().cpu()),
        "delta_teacher_norm_mean": float(delta_array.norm(dim=-1).mean().cpu()),
    }
    metrics.update(absolute_probe)
    metrics.update(delta_probe)
    return metrics


def _decision_label(metrics: dict[str, Any], eval_split: str) -> str:
    prefix = f"{eval_split}/"
    delta_rank = float(metrics.get(prefix + "delta_teacher_effective_rank", 0.0))
    delta_std = float(metrics.get(prefix + "delta_teacher_std_mean", 0.0))
    if delta_rank < 4.0 or delta_std < 0.01:
        return "TRANSITION_TARGET_COLLAPSED"
    batch_acc = metrics.get(prefix + "delta_target_batch_probe_accuracy")
    majority = metrics.get(prefix + "delta_target_batch_probe_majority_accuracy")
    if batch_acc is not None and majority is not None and not np.isnan(batch_acc) and batch_acc > majority + 0.25:
        return "TRANSITION_TARGET_BATCH_CONTAMINATED"
    if (
        float(metrics.get(prefix + "delta_target_nn_recall@1", 0.0))
        >= float(metrics.get(prefix + "absolute_target_nn_recall@1", 0.0))
        or delta_rank >= 6.0
    ):
        return "DELTA_TARGET_HAS_HEADROOM"
    return "ABSOLUTE_TARGET_PREFERRED"


def _normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def _effective_rank_torch(values: torch.Tensor, eps: float = 1e-8) -> float:
    if values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centered)
    probs = spectrum / spectrum.sum().clamp_min(eps)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float(torch.exp(entropy).cpu())


def _write_report(path: Path, metrics: dict[str, Any], *, eval_split: str) -> None:
    prefix = f"{eval_split}/"
    lines = [
        "# Transition Target Audit",
        "",
        f"Decision label: `{metrics['decision_label']}`",
        "",
        "## Eval Split Metrics",
        "",
        f"- Source->target cosine mean: `{metrics.get(prefix + 'source_to_target_cosine_mean', float('nan')):.4f}`",
        f"- Source->delta cosine mean: `{metrics.get(prefix + 'source_to_delta_cosine_mean', float('nan')):.4f}`",
        f"- Delta teacher effective rank: `{metrics.get(prefix + 'delta_teacher_effective_rank', float('nan')):.4f}`",
        f"- Delta teacher std mean: `{metrics.get(prefix + 'delta_teacher_std_mean', float('nan')):.6f}`",
        f"- Absolute target NN recall@1: `{metrics.get(prefix + 'absolute_target_nn_recall@1', float('nan')):.4f}`",
        f"- Delta target NN recall@1: `{metrics.get(prefix + 'delta_target_nn_recall@1', float('nan')):.4f}`",
        f"- Absolute target batch probe accuracy: `{metrics.get(prefix + 'absolute_target_batch_probe_accuracy', float('nan')):.4f}`",
        f"- Delta target batch probe accuracy: `{metrics.get(prefix + 'delta_target_batch_probe_accuracy', float('nan')):.4f}`",
        "",
        "## Interpretation",
        "",
        "If delta targets have non-collapsed rank and discriminative structure, Phase 3 may implement delta-state JEPA.",
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
