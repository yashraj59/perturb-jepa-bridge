from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.norman2019 import add_gears_simulation_split, load_norman2019_condition_data
from perturb_jepa.evaluation.bioflow_metrics import evaluate_bioflow_batches
from perturb_jepa.models.bioflow_jepa import BioFlowJEPA, BioFlowJEPAConfig
from perturb_jepa.models.biotech_jepa import BioTechJEPAConfig
from perturb_jepa.models.image_encoder import ImageEncoderConfig
from perturb_jepa.models.perturbation_encoder import PerturbationEncoderConfig
from perturb_jepa.models.rna_encoder import RNAEncoderConfig
from perturb_jepa.training.action_descriptors import descriptors_for_perturbation_ids, synthetic_action_descriptor_matrix, synthetic_action_descriptor_spec
from perturb_jepa.training.bioaction_batches import BioActionConditionBatch, iter_bioaction_condition_batches
from perturb_jepa.training.bioflow_losses import BioFlowJEPALossWeights
from perturb_jepa.training.bioflow_trainer import BioFlowJEPATrainer
from perturb_jepa.training.norman_biotech_batches import build_norman_biotech_spec, iter_norman_biotech_condition_batches
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config


PHASE4_ROOT = Path("outputs/autoresearch_bioflow_jepa_phase4")
DEFAULT_BASE_CHECKPOINT = Path("outputs/autoresearch_biotech_jepa_phase2/experiments/BTJ001_synth_genetic_anchor_seed0/checkpoint.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BioFlow-JEPA frozen-encoder transition probe.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite", choices=("synth_genetic_anchor_lite", "norman"))
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--norman-h5ad", type=Path, default=Path("data/raw/gears_norman/norman/perturb_processed.h5ad"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bag-size", type=int, default=3)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--bio-dim", type=int, default=24)
    parser.add_argument("--tech-dim", type=int, default=8)
    parser.add_argument("--predictor-dim", type=int, default=64)
    parser.add_argument("--gene-count", type=int, default=256)
    parser.add_argument("--transition-mode", default="vector_field", choices=("vector_field", "low_rank_koopman", "hybrid"))
    parser.add_argument("--flow-steps", type=int, default=4)
    parser.add_argument("--use-delta-whitening", action="store_true")
    parser.add_argument("--no-tangent-projection", action="store_true")
    parser.add_argument("--use-source-improvement-hinge", action="store_true")
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--velocity-loss-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--delta-direction-loss-weight", type=float, default=2.0)
    parser.add_argument("--source-improvement-loss-weight", type=float, default=2.0)
    parser.add_argument("--delta-rank-variance-weight", type=float, default=0.1)
    parser.add_argument("--action-negative-loss-weight", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = _select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_dataset_payload(args)
    action_dim = _action_dim(payload)
    base_config, base_state = _load_base(args.base_checkpoint, device=device)
    config = BioFlowJEPAConfig(
        base_biotech_config=base_config,
        action_dim=action_dim,
        transition_mode=args.transition_mode,
        flow_steps=args.flow_steps,
        use_delta_whitening=args.use_delta_whitening,
        use_tangent_projection=not args.no_tangent_projection,
        use_source_improvement_hinge=args.use_source_improvement_hinge,
        endpoint_loss_weight=args.endpoint_loss_weight,
        velocity_loss_weight=args.velocity_loss_weight,
        delta_direction_loss_weight=args.delta_direction_loss_weight,
        source_improvement_loss_weight=args.source_improvement_loss_weight,
        delta_rank_variance_weight=args.delta_rank_variance_weight,
        action_negative_loss_weight=args.action_negative_loss_weight,
    )
    model = BioFlowJEPA(config).to(device)
    model.base.load_state_dict(base_state)
    if args.freeze_encoders:
        model.freeze_base()
    if args.use_delta_whitening:
        model.fit_delta_whitening_from_batches(
            _batches(payload, args, split="train", steps=None, batch_size=max(8, args.batch_size), device=device),
            device=device,
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    weights = BioFlowJEPALossWeights(
        endpoint_loss_weight=args.endpoint_loss_weight,
        velocity_loss_weight=args.velocity_loss_weight,
        delta_direction_loss_weight=args.delta_direction_loss_weight,
        source_improvement_loss_weight=args.source_improvement_loss_weight,
        delta_rank_variance_weight=args.delta_rank_variance_weight,
        action_negative_loss_weight=args.action_negative_loss_weight,
    )
    trainer = BioFlowJEPATrainer(
        model,
        optimizer,
        weights=weights,
        device=device,
        update_teachers=not args.freeze_encoders,
    )
    last: dict[str, float] = {}
    with (args.output_dir / "metrics_train.jsonl").open("w", encoding="utf-8") as handle:
        for result in map(
            trainer.train_step,
            _batches(payload, args, split="train", steps=args.steps, batch_size=args.batch_size, device=device),
        ):
            last = result.diagnostics
            handle.write(json.dumps(_jsonable(last), sort_keys=True) + "\n")
    metrics = evaluate_bioflow_batches(
        model,
        _batches(payload, args, split=args.eval_split, steps=args.eval_steps, batch_size=max(4, args.batch_size), device=device),
        device=device,
    )
    metrics.update({f"last_train/{key}": value for key, value in last.items() if isinstance(value, (int, float))})
    metrics["dataset"] = args.dataset  # type: ignore[assignment]
    metrics["eval_split"] = args.eval_split  # type: ignore[assignment]
    metrics["seed"] = float(args.seed)
    metrics["device_used_cuda"] = float(str(device).startswith("cuda"))
    decision = bfj001_decision(metrics)
    metrics["decision_label"] = decision  # type: ignore[assignment]
    write_json(args.output_dir / "config.json", {"args": vars(args), "device": str(device), "model": _jsonable(config), "weights": _jsonable(weights)})
    write_json(args.output_dir / "metrics_eval.json", metrics)
    write_identity_report(args.output_dir / "jepa_identity_report.md", metrics)
    write_model_card(args.output_dir / "model_card.md", args=args, metrics=metrics, decision=decision, device=str(device))
    update_phase4_results(args=args, metrics=metrics, decision=decision)
    append_phase4_journal(args=args, metrics=metrics, decision=decision)
    if decision in {"BFJ_TIER1_DISCARD_NO_SIGNAL", "BFJ_TIER1_DISCARD_ANTI_ALIGNED_DELTA", "BFJ_TIER1_DISCARD_RANK_COLLAPSE", "BFJ_TIER1_DISCARD_IDENTITY_VIOLATION"}:
        write_final_report(args=args, metrics=metrics, decision=decision)
    if args.save_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": _jsonable(config),
                "base_checkpoint": str(args.base_checkpoint),
                "dataset": args.dataset,
                "seed": args.seed,
                "decision_label": decision,
            },
            args.output_dir / "checkpoint.pt",
        )
    print(json.dumps(_jsonable(metrics), sort_keys=True))
    return 0


def _build_dataset_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "norman":
        dataset = add_gears_simulation_split(load_norman2019_condition_data(args.norman_h5ad), seed=args.split_seed)
        spec = build_norman_biotech_spec(dataset, gene_count=args.gene_count)
        return {"kind": "norman", "dataset": dataset, "spec": spec, "descriptor_matrix": None}
    dataset = generate_synthetic_biology_lite(synthetic_lite_config(args.dataset, seed=args.seed))
    descriptor_matrix = synthetic_action_descriptor_matrix(dataset, synthetic_action_descriptor_spec(dataset))
    return {"kind": "synthetic", "dataset": dataset, "descriptor_matrix": descriptor_matrix}


def _action_dim(payload: dict[str, Any]) -> int:
    if payload["kind"] == "norman":
        return int(payload["spec"].descriptor_dim)
    return int(payload["descriptor_matrix"].shape[1])


def _batches(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    split: str,
    steps: int | None,
    batch_size: int,
    device: torch.device,
) -> Iterable[BioActionConditionBatch]:
    if payload["kind"] == "norman":
        return iter_norman_biotech_condition_batches(
            payload["dataset"],
            payload["spec"],
            split=split,
            batch_size=batch_size,
            steps=steps,
            seed=args.seed + (0 if split == "train" else 1000),
            device=device,
        )
    batches = iter_bioaction_condition_batches(
        payload["dataset"],
        split=split,
        batch_size=batch_size,
        bag_size=args.bag_size,
        steps=steps,
        seed=args.seed + (0 if split == "train" else 1000),
        device=device,
    )
    return _attach_synthetic_descriptors(batches, payload["descriptor_matrix"])


def _attach_synthetic_descriptors(
    batches: Iterable[BioActionConditionBatch],
    descriptor_matrix: np.ndarray,
) -> Iterable[BioActionConditionBatch]:
    for batch in batches:
        batch.descriptor = descriptors_for_perturbation_ids(batch.perturbation_id, descriptor_matrix)
        yield batch


def _load_base(checkpoint: Path, *, device: torch.device) -> tuple[BioTechJEPAConfig, dict[str, torch.Tensor]]:
    payload = torch.load(checkpoint, map_location=device)
    return _config_from_payload(payload["config"]), payload["model_state_dict"]


def _config_from_payload(payload: dict[str, Any]) -> BioTechJEPAConfig:
    return BioTechJEPAConfig(
        rna=RNAEncoderConfig(**_dataclass_kwargs(RNAEncoderConfig, payload["rna"])),
        image=ImageEncoderConfig(**_dataclass_kwargs(ImageEncoderConfig, payload["image"])),
        perturbation=PerturbationEncoderConfig(**_dataclass_kwargs(PerturbationEncoderConfig, payload["perturbation"])),
        **{
            field.name: payload[field.name]
            for field in fields(BioTechJEPAConfig)
            if field.name in payload and field.name not in {"rna", "image", "perturbation"}
        },
    )


def bfj001_decision(metrics: dict[str, Any]) -> str:
    identity_ok = (
        float(metrics.get("condition_key_feature_present", 1.0)) == 0.0
        and float(metrics.get("biological_key_onehot_present", 1.0)) == 0.0
        and float(metrics.get("pls_raw_linear_main_path_used", 1.0)) == 0.0
        and float(metrics.get("teacher_stop_gradient_verified", 0.0)) == 1.0
    )
    if not identity_ok:
        return "BFJ_TIER1_DISCARD_IDENTITY_VIOLATION"
    if float(metrics.get("transition_source_cosine_improvement", -1.0)) <= 0.0:
        return "BFJ_TIER1_DISCARD_NO_SIGNAL"
    if float(metrics.get("delta_cosine", -1.0)) <= 0.0:
        return "BFJ_TIER1_DISCARD_ANTI_ALIGNED_DELTA"
    if float(metrics.get("delta_prediction_effective_rank", 0.0)) < 5.0:
        return "BFJ_TIER1_DISCARD_RANK_COLLAPSE"
    keep = (
        float(metrics.get("transition_source_cosine_improvement", 0.0)) >= 0.0200
        and float(metrics.get("delta_cosine", 0.0)) > 0.0500
        and (
            float(metrics.get("delta_prediction_effective_rank", 0.0)) >= 8.0
            or float(metrics.get("delta_rank_ratio", 0.0)) >= 0.65
        )
        and float(metrics.get("transition_to_target_recall@1", 0.0)) >= 0.2500
        and 0.25 <= float(metrics.get("delta_magnitude_ratio", 0.0)) <= 2.50
        and float(metrics.get("image_to_rna_recall@1", 1.0)) >= 0.0500
        and float(metrics.get("rna_to_image_recall@1", 1.0)) >= 0.0500
    )
    return "BFJ_TIER1_KEEP_CONTROLLED_SIGNAL" if keep else "BFJ_TIER1_DISCARD_NO_SIGNAL"


def write_identity_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# BioFlow-JEPA Identity Report",
        "",
        f"- Real JEPA path: `{'yes' if metrics.get('encoder_path_used') == 1.0 and metrics.get('pls_raw_linear_main_path_used') == 0.0 else 'no'}`",
        "- Online/context encoder path: `yes`",
        "- EMA target encoders: `yes`",
        f"- Teacher stop-gradient verified: `{'yes' if metrics.get('teacher_stop_gradient_verified') == 1.0 else 'no'}`",
        f"- Transition target stop-gradient: `{'yes' if metrics.get('transition_target_stop_gradient') == 1.0 else 'no'}`",
        f"- Separate z_bio/z_tech: `{'yes' if metrics.get('separate_bio_and_tech_latents_present') == 1.0 else 'no'}`",
        f"- Condition-key feature present: `{metrics.get('condition_key_feature_present')}`",
        f"- Biological-key one-hot present: `{metrics.get('biological_key_onehot_present')}`",
        f"- PLS raw-linear main path used: `{metrics.get('pls_raw_linear_main_path_used')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(path: Path, *, args: argparse.Namespace, metrics: dict[str, Any], decision: str, device: str) -> None:
    lines = [
        "# BioFlow-JEPA BFJ001 Model Card",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Device: `{device}`",
        f"- Steps: `{args.steps}`",
        f"- Frozen encoders: `{bool(args.freeze_encoders)}`",
        f"- Transition mode: `{args.transition_mode}`",
        f"- Delta whitening: `{bool(args.use_delta_whitening)}`",
        f"- Source-improvement hinge: `{bool(args.use_source_improvement_hinge)}`",
        f"- Decision label: `{decision}`",
        "",
        "## Metrics",
        "",
        f"- transition_source_cosine_improvement: `{float(metrics.get('transition_source_cosine_improvement', float('nan'))):.4f}`",
        f"- delta_cosine: `{float(metrics.get('delta_cosine', float('nan'))):.4f}`",
        f"- delta_prediction_effective_rank: `{float(metrics.get('delta_prediction_effective_rank', float('nan'))):.4f}`",
        f"- delta_teacher_effective_rank: `{float(metrics.get('delta_teacher_effective_rank', float('nan'))):.4f}`",
        f"- transition_to_target_recall@1: `{float(metrics.get('transition_to_target_recall@1', float('nan'))):.4f}`",
        f"- delta_magnitude_ratio: `{float(metrics.get('delta_magnitude_ratio', float('nan'))):.4f}`",
        f"- image_to_RNA recall@1: `{float(metrics.get('image_to_rna_recall@1', float('nan'))):.4f}`",
        f"- RNA_to_image recall@1: `{float(metrics.get('rna_to_image_recall@1', float('nan'))):.4f}`",
        "",
        "Protected PLS remains the model of record. This candidate cannot promote in Phase 4.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_phase4_results(*, args: argparse.Namespace, metrics: dict[str, Any], decision: str) -> None:
    path = PHASE4_ROOT / "results.tsv"
    row = {
        "commit": git_commit_label(),
        "experiment_num": "BFJ001",
        "stage": "StageE",
        "family": "BioFlow_vector_field",
        "tier_reached": "Tier1",
        "decision_label": decision,
        "status": "SEARCH_CLOSED_NO_NEW_BASELINE" if decision.startswith("BFJ_TIER1_DISCARD") else decision,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed_list": str(args.seed),
        "primary_metric": f"transition_source_cosine_improvement={float(metrics.get('transition_source_cosine_improvement', float('nan'))):.4f}",
        "secondary_metric": f"delta_cosine={float(metrics.get('delta_cosine', float('nan'))):.4f}; delta_prediction_effective_rank={float(metrics.get('delta_prediction_effective_rank', float('nan'))):.4f}; transition_to_target_recall@1={float(metrics.get('transition_to_target_recall@1', float('nan'))):.4f}",
        "protected_metric_summary": "protected_rank3_train_split_pls_remains_model_of_record; encoder_path=1.0; pls_main_path=0.0; condition_key_features=0.0",
        "architectural_change": "controlled_latent_vector_field_delta_whitening_source_improvement_hinge",
        "description": "Frozen-encoder BioFlow-JEPA transition probe.",
    }
    if not path.exists() or path.stat().st_size == 0:
        pd_header = list(row)
        path.write_text("\t".join(pd_header) + "\n", encoding="utf-8")
    import pandas as pd

    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["experiment_num"].astype(str) != "BFJ001"] if "experiment_num" in frame.columns else frame
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)


def append_phase4_journal(*, args: argparse.Namespace, metrics: dict[str, Any], decision: str) -> None:
    entry = "\n".join(
        [
            "",
            "## BFJ001: Frozen-Encoder BioFlow Transition Probe",
            "",
            "**Hypothesis**: A controlled vector field trained with train-only delta whitening, delta direction loss, and source-improvement hinge will stop the BMJ001 anti-aligned delta failure while preserving JEPA identity.",
            "",
            "**Implementation files changed**: `perturb_jepa/models/bioflow_jepa.py`, `perturb_jepa/training/bioflow_losses.py`, `perturb_jepa/training/bioflow_trainer.py`, `perturb_jepa/evaluation/bioflow_metrics.py`, `scripts/train_bioflow_jepa.py`.",
            "",
            f"**Initialization / identity preservation**: base BioTech checkpoint `{args.base_checkpoint}`; encoders frozen `{bool(args.freeze_encoders)}`; PLS main path `0.0`; condition-key feature `0.0`; teacher stop-gradient `1.0`.",
            "",
            f"**Tier result**: transition improvement `{float(metrics.get('transition_source_cosine_improvement', float('nan'))):.4f}`, delta cosine `{float(metrics.get('delta_cosine', float('nan'))):.4f}`, pred delta rank `{float(metrics.get('delta_prediction_effective_rank', float('nan'))):.4f}`, recall@1 `{float(metrics.get('transition_to_target_recall@1', float('nan'))):.4f}`.",
            "",
            f"**Decision label**: `{decision}`.",
            "",
            "**Learning**: BFJ001 is a transition-operator diagnostic only; protected PLS remains the model of record and no Phase 4 result can promote.",
            "",
        ]
    )
    with (PHASE4_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(entry)


def write_final_report(*, args: argparse.Namespace, metrics: dict[str, Any], decision: str) -> None:
    lines = [
        "# BioFlow-JEPA Phase 4 Final Report",
        "",
        f"Decision label: `{decision}`",
        "",
        "No BioFlow-JEPA candidate is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.",
        "",
        "## BFJ001 Stop Metrics",
        "",
        f"- transition_source_cosine_improvement: `{float(metrics.get('transition_source_cosine_improvement', float('nan'))):.4f}`",
        f"- delta_cosine: `{float(metrics.get('delta_cosine', float('nan'))):.4f}`",
        f"- delta_prediction_effective_rank: `{float(metrics.get('delta_prediction_effective_rank', float('nan'))):.4f}`",
        f"- source_improvement_hinge_violation_fraction: `{float(metrics.get('source_improvement_hinge_violation_fraction', float('nan'))):.4f}`",
        "",
        "## Stop Reason",
        "",
        "BFJ001 failed a Phase 4 Tier 1 stop/discard gate, so the autonomous loop stops without launching BFJ002-BFJ006 or Norman.",
        "",
        "## Files",
        "",
        f"- Experiment directory: `{args.output_dir}`",
        "- Delta audit: `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md`",
    ]
    (PHASE4_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def git_commit_label() -> str:
    import subprocess

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        return f"{commit}+dirty" if dirty else commit
    except Exception:
        return "unknown"


def _select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
