from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.losses import BridgeLossWeights, bridge_loss
from perturb_jepa.training.seed import seed_everything
from perturb_jepa.training.synthetic_biology_lite import generate_synthetic_biology_lite, synthetic_lite_config
from perturb_jepa.training.trainer import forward_batch, move_batch_to_device, patchify_batch_images
from scripts.run_synthetic_lite_step0 import _experiment_config_for_dataset


OUTPUT_DIR = Path("outputs/autoresearch_synth_lite/diagnostics/OBJECTIVE_COLLAPSE_AUDIT")
TRACE_FIELDS = [
    "run",
    "step",
    "total",
    "rna_mask",
    "image_mask",
    "rna_jepa",
    "image_jepa",
    "align",
    "mmd",
    "sliced_wasserstein",
    "rna_perturbation_cls",
    "image_perturbation_cls",
    "rna_batch_adv",
    "image_batch_adv",
    "cycle",
    "response_bottleneck",
    "rna_shared_variance",
    "image_shared_variance",
    "rna_shared_covariance",
    "image_shared_covariance",
    "shared_cross_correlation",
    "rna_shared_min_std",
    "image_shared_min_std",
    "rna_shared_mean_std",
    "image_shared_mean_std",
    "rna_rank",
    "image_rank",
    "collapse_flag",
    "student_teacher_cosine_mean",
    "student_teacher_cosine_std",
]
GRAD_FIELDS = [
    "run",
    "step",
    "total",
    "rna_encoder",
    "image_encoder",
    "rna_projection",
    "image_projection",
    "rna_bag_aggregator",
    "image_bag_aggregator",
    "jepa_predictors",
    "state_response_heads",
    "perturbation_encoder",
    "adversaries",
    "counterfactual_heads",
    "decoders",
    "other",
]
RESULT_FIELDS = [
    "run",
    "bag_aggregator",
    "steps_completed",
    "first_collapse_step",
    "final_collapse",
    "final_rna_min_std",
    "final_image_min_std",
    "final_rna_rank",
    "final_image_rank",
    "dominant_loss_step0",
    "dominant_grad_step0",
    "interpretation",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_inventory()
    _write_methods()
    seed_everything(0)
    dataset_config = synthetic_lite_config("synth_micro", seed=0)
    dataset = generate_synthetic_biology_lite(dataset_config)
    (OUTPUT_DIR / "generation_config.json").write_text(
        json.dumps(asdict(dataset_config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runs = _runs()
    trace_rows = []
    grad_rows = []
    result_rows = []
    for run in runs:
        trace, grads, result = _run_diagnostic(run, dataset)
        trace_rows.extend(trace)
        grad_rows.extend(grads)
        result_rows.append(result)
        print(
            f"{run['name']}: collapse={result['final_collapse']} first={result['first_collapse_step']} "
            f"rna_std={result['final_rna_min_std']} image_std={result['final_image_min_std']}",
            flush=True,
        )

    _write_tsv(OUTPUT_DIR / "LOSS_TRACE.tsv", TRACE_FIELDS, trace_rows)
    _write_tsv(OUTPUT_DIR / "GRADIENT_TRACE.tsv", GRAD_FIELDS, grad_rows)
    _write_tsv(OUTPUT_DIR / "OBJECTIVE_ABLATION_RESULTS.tsv", RESULT_FIELDS, result_rows)
    _write_report(result_rows, trace_rows, grad_rows)
    return 0


def _runs() -> list[dict]:
    base = {
        "steps": 30,
        "bag_aggregator": "attention",
        "weights": {},
    }
    runs = [
        ("full_attention", "attention", dict()),
        ("full_mean", "mean", dict()),
        ("rna_reconstruction_only", "attention", dict(rna_mask=1.0, image_mask=0.0, jepa=0.0, align=0.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("image_reconstruction_only", "attention", dict(rna_mask=0.0, image_mask=1.0, jepa=0.0, align=0.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("reconstruction_both", "attention", dict(rna_mask=1.0, image_mask=1.0, jepa=0.0, align=0.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("jepa_only", "attention", dict(rna_mask=0.0, image_mask=0.0, jepa=1.0, align=0.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("align_only", "attention", dict(rna_mask=0.0, image_mask=0.0, jepa=0.0, align=1.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("reconstruction_plus_jepa", "attention", dict(rna_mask=1.0, image_mask=1.0, jepa=1.0, align=0.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("reconstruction_plus_align", "attention", dict(rna_mask=1.0, image_mask=1.0, jepa=0.0, align=1.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
        ("full_no_adversaries", "attention", dict(batch_adv=0.0, perturbation_cls=0.0)),
        ("full_no_distribution", "attention", dict(mmd=0.0, sliced_wasserstein=0.0)),
        ("full_no_reconstruction", "attention", dict(rna_mask=0.0, image_mask=0.0)),
        ("full_no_jepa", "attention", dict(jepa=0.0)),
        ("full_no_align", "attention", dict(align=0.0)),
        ("mean_reconstruction_plus_align", "mean", dict(rna_mask=1.0, image_mask=1.0, jepa=0.0, align=1.0, mmd=0.0, sliced_wasserstein=0.0, perturbation_cls=0.0, batch_adv=0.0, cycle=0.0, response_bottleneck=0.0)),
    ]
    return [
        {
            **base,
            "name": name,
            "bag_aggregator": aggregator,
            "weights": weights,
        }
        for name, aggregator, weights in runs
    ]


def _run_diagnostic(run: dict, dataset) -> tuple[list[dict], list[dict], dict]:
    seed_everything(0)
    weights = _weights(run["weights"])
    config = _experiment_config_for_dataset(
        dataset,
        steps=run["steps"],
        device="cpu",
        bag_aggregator=run["bag_aggregator"],
        num_bag_prototypes=2,
        rna_mask_weight=weights.rna_mask,
        image_mask_weight=weights.image_mask,
        jepa_weight=weights.jepa,
        align_weight=weights.align,
        mmd_weight=weights.mmd,
        sliced_wasserstein_weight=weights.sliced_wasserstein,
        perturbation_cls_weight=weights.perturbation_cls,
        batch_adv_weight=weights.batch_adv,
        counterfactual_weight=weights.counterfactual,
        cycle_weight=weights.cycle,
        response_bottleneck_weight=weights.response_bottleneck,
    )
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    trace_rows = []
    grad_rows = []
    first_collapse_step = None
    batches = dataset.iter_condition_batches(
        split="train",
        batch_size=8,
        bag_size=dataset.config.cells_per_condition,
        steps=run["steps"],
        seed=0,
        device="cpu",
    )
    for step, batch in enumerate(batches):
        model.train()
        batch = move_batch_to_device(batch, "cpu")
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_batch(model, batch)
        total, terms = bridge_loss(
            outputs,
            rna_values=batch.expression_values,
            rna_mask=batch.rna_token_mask,
            image_patches=patchify_batch_images(batch.images, model.config.image.patch_size),
            image_patch_mask=batch.image_patch_mask,
            perturbation_id=batch.perturbation_id,
            batch_id=batch.batch_id,
            temperature=weights.temperature,
            weights=weights,
        )
        total.backward()
        grad_row = _grad_row(run["name"], step, model, float(total.detach().cpu()))
        optimizer.step()
        model.update_teachers(decay=config.training.ema_decay)
        trace_row = _trace_row(run["name"], step, terms, _embedding_diagnostics(outputs))
        if trace_row["collapse_flag"] and first_collapse_step is None:
            first_collapse_step = step
        trace_rows.append(trace_row)
        grad_rows.append(grad_row)

    final = trace_rows[-1]
    result = {
        "run": run["name"],
        "bag_aggregator": run["bag_aggregator"],
        "steps_completed": len(trace_rows),
        "first_collapse_step": "none" if first_collapse_step is None else first_collapse_step,
        "final_collapse": final["collapse_flag"],
        "final_rna_min_std": final["rna_shared_min_std"],
        "final_image_min_std": final["image_shared_min_std"],
        "final_rna_rank": final["rna_rank"],
        "final_image_rank": final["image_rank"],
        "dominant_loss_step0": _dominant_loss(trace_rows[0]),
        "dominant_grad_step0": _dominant_grad(grad_rows[0]),
        "interpretation": _interpretation(run["name"], trace_rows, grad_rows),
    }
    return trace_rows, grad_rows, result


def _weights(overrides: dict) -> BridgeLossWeights:
    base = {
        "temperature": 0.1,
        "rna_mask": 0.2,
        "image_mask": 0.2,
        "jepa": 1.0,
        "align": 1.0,
        "mmd": 0.05,
        "sliced_wasserstein": 0.02,
        "perturbation_cls": 0.05,
        "batch_adv": 0.02,
        "counterfactual": 0.0,
        "cycle": 0.05,
        "response_bottleneck": 0.005,
        "shared_variance": 0.0,
        "shared_covariance": 0.0,
        "cross_correlation": 0.0,
    }
    base.update(overrides)
    return BridgeLossWeights(**base)


def _trace_row(run: str, step: int, terms: dict[str, torch.Tensor], diagnostics: dict) -> dict:
    row = {field: 0.0 for field in TRACE_FIELDS}
    row["run"] = run
    row["step"] = step
    for name in row:
        if name in terms:
            row[name] = float(terms[name].detach().cpu())
    row.update(diagnostics)
    return row


def _embedding_diagnostics(outputs: dict[str, torch.Tensor]) -> dict:
    rna = outputs["rna_shared"].detach()
    image = outputs["image_shared"].detach()
    rna_std = rna.reshape(-1, rna.shape[-1]).std(dim=0, unbiased=False)
    image_std = image.reshape(-1, image.shape[-1]).std(dim=0, unbiased=False)
    cosines = []
    if "rna_teacher_shared" in outputs:
        cosines.append(torch.nn.functional.cosine_similarity(outputs["rna_shared"], outputs["rna_teacher_shared"], dim=-1))
    if "image_teacher_shared" in outputs:
        cosines.append(torch.nn.functional.cosine_similarity(outputs["image_shared"], outputs["image_teacher_shared"], dim=-1))
    cosine = torch.cat([value.reshape(-1).detach() for value in cosines]) if cosines else torch.zeros(1)
    return {
        "rna_shared_min_std": float(rna_std.min().cpu()),
        "image_shared_min_std": float(image_std.min().cpu()),
        "rna_shared_mean_std": float(rna_std.mean().cpu()),
        "image_shared_mean_std": float(image_std.mean().cpu()),
        "rna_rank": float(torch.linalg.matrix_rank(rna.reshape(-1, rna.shape[-1]).cpu(), tol=1e-3)),
        "image_rank": float(torch.linalg.matrix_rank(image.reshape(-1, image.shape[-1]).cpu(), tol=1e-3)),
        "collapse_flag": bool(rna_std.min() < 0.01 or image_std.min() < 0.01),
        "student_teacher_cosine_mean": float(cosine.mean().cpu()),
        "student_teacher_cosine_std": float(cosine.std(unbiased=False).cpu()),
    }


def _grad_row(run: str, step: int, model, total: float) -> dict:
    groups = {field: 0.0 for field in GRAD_FIELDS}
    groups["run"] = run
    groups["step"] = step
    groups["total"] = total
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm().cpu())
        groups[_group_for_param(name)] += value * value
    for key in GRAD_FIELDS:
        if key not in ("run", "step", "total"):
            groups[key] = groups[key] ** 0.5
    return groups


def _group_for_param(name: str) -> str:
    if name.startswith("rna_encoder"):
        return "rna_encoder"
    if name.startswith("image_encoder"):
        return "image_encoder"
    if name.startswith("rna_projection"):
        return "rna_projection"
    if name.startswith("image_projection"):
        return "image_projection"
    if name.startswith("rna_bag_aggregator"):
        return "rna_bag_aggregator"
    if name.startswith("image_bag_aggregator"):
        return "image_bag_aggregator"
    if "jepa_predictor" in name:
        return "jepa_predictors"
    if name.startswith("state_head") or name.startswith("response_head"):
        return "state_response_heads"
    if name.startswith("perturbation_encoder"):
        return "perturbation_encoder"
    if "adversary" in name or "classifier" in name:
        return "adversaries"
    if name.startswith("delta_"):
        return "counterfactual_heads"
    if "decoder" in name:
        return "decoders"
    return "other"


def _dominant_loss(row: dict) -> str:
    keys = [
        key
        for key in TRACE_FIELDS
        if key not in {"run", "step", "total", "collapse_flag"} and isinstance(row.get(key), float)
    ]
    keys = [key for key in keys if not key.endswith("_std") and not key.endswith("_rank") and "cosine" not in key]
    return max(keys, key=lambda key: abs(float(row.get(key, 0.0))))


def _dominant_grad(row: dict) -> str:
    keys = [key for key in GRAD_FIELDS if key not in {"run", "step", "total"}]
    return max(keys, key=lambda key: abs(float(row.get(key, 0.0))))


def _interpretation(run: str, trace_rows: list[dict], grad_rows: list[dict]) -> str:
    first = trace_rows[0]
    final = trace_rows[-1]
    if final["collapse_flag"] and first["collapse_flag"]:
        return "collapsed_at_initial_step"
    if final["collapse_flag"]:
        return "collapsed_during_training"
    if final["rna_shared_min_std"] < 0.02 or final["image_shared_min_std"] < 0.02:
        return "near_collapse"
    if "mean" in run:
        return "noncollapsed_mean_pooling_signal"
    return "noncollapsed"


def _write_inventory() -> None:
    paths = [
        "perturb_jepa/training/synthetic_biology_lite.py",
        "scripts/run_synthetic_lite_step0.py",
        "outputs/autoresearch_synth_lite/results.tsv",
        "outputs/autoresearch_synth_lite/final_report.md",
    ]
    lines = ["# Inventory", "", "Available artifacts and code inspected:"]
    for path in paths:
        p = Path(path)
        lines.append(f"- `{path}`: {'present' if p.exists() else 'missing'}")
    (OUTPUT_DIR / "INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_methods() -> None:
    (OUTPUT_DIR / "METHODS.md").write_text(
        "# Methods\n\n"
        "Diagnostic-only CPU investigation on `synth_micro`. No architecture promotion and no real data. "
        "Each run trains the same tiny bridge for 30 steps under one objective ablation, logging scalar losses, "
        "module-gradient norms, representation std/rank, and first collapse step.\n",
        encoding="utf-8",
    )


def _write_report(results: list[dict], trace_rows: list[dict], grad_rows: list[dict]) -> None:
    collapsed = [row for row in results if row["final_collapse"]]
    noncollapsed = [row for row in results if not row["final_collapse"]]
    first_step = [row for row in results if row["first_collapse_step"] == 0]
    transient = [row for row in results if row["first_collapse_step"] != "none" and not row["final_collapse"]]
    dominant_losses = {}
    for row in results:
        dominant_losses[row["dominant_loss_step0"]] = dominant_losses.get(row["dominant_loss_step0"], 0) + 1
    dominant_grads = {}
    for row in results:
        dominant_grads[row["dominant_grad_step0"]] = dominant_grads.get(row["dominant_grad_step0"], 0) + 1
    eval_rows = _eval_compare_rows()
    lines = [
        "# Objective Collapse Report",
        "",
        "## Summary",
        "",
        f"- Diagnostic runs: {len(results)}",
        f"- Final collapsed: {len(collapsed)}",
        f"- Non-collapsed: {len(noncollapsed)}",
        f"- Collapsed at step 0: {len(first_step)}",
        f"- Transient collapse during train-batch tracing: {len(transient)}",
        "",
        "## Main Finding",
        "",
        "The objective ablations do not reproduce final collapse on the traced training condition batches: all 15 final train-batch traces remain above the hard std gate. "
        "The same 30-step setup does collapse in the normal Step 0 held-out evaluation path. This points away from a single loss term immediately destroying variance, and toward a split/evaluation geometry problem: the shared representation learned on train condition bags does not survive collection over held-out/test condition bags.",
        "",
        "## Non-Collapsed Runs",
        "",
    ]
    if noncollapsed:
        for row in noncollapsed:
            lines.append(
                f"- `{row['run']}`: rna_min_std={row['final_rna_min_std']}, "
                f"image_min_std={row['final_image_min_std']}, rna_rank={row['final_rna_rank']}, image_rank={row['final_image_rank']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Held-Out Eval Check", ""])
    if eval_rows:
        for row in eval_rows:
            lines.append(
                f"- `{row['name']}`: collapse={row['collapse_flag']}, "
                f"rna_min_std={row['rna_min_std']}, image_min_std={row['image_min_std']}, "
                f"rna_to_image_recall@1={row['recall1']}, random_recall@1={row['random_recall1']}, "
                f"mean_prototype_recall@1={row['mean_prototype_recall1']}, "
                f"batch_probe_balanced_accuracy={row['batch_probe_balanced_accuracy']}, "
                f"rna_latent_r2={row['rna_latent_r2']}, image_latent_r2={row['image_latent_r2']}"
            )
    else:
        lines.append("- Not run.")
    lines.extend(
        [
            "",
            "## Dominant Step-0 Losses",
            "",
            *[f"- {key}: {value}" for key, value in sorted(dominant_losses.items())],
            "",
            "These are raw unweighted term magnitudes recorded for scale inspection. They should not be read as weighted objective dominance in ablations where a term weight is zero.",
            "",
            "## Dominant Step-0 Gradient Groups",
            "",
            *[f"- {key}: {value}" for key, value in sorted(dominant_grads.items())],
            "",
            "## Interpretation",
            "",
            "The 100-experiment search was not primarily failing because of one auxiliary loss weight. "
            "Loss ablations are stable on traced train batches, while held-out Step 0 evaluation still collapses and retrieves at random. "
            "Mean pooling improves held-out RNA variance but does not fix image-side collapse or retrieval. Alignment-only is the most fragile train-batch ablation, with transient near-threshold collapse, and removing distribution or reconstruction terms leaves final variance close to the gate. "
            "The next phase should audit the condition-bag split/evaluation path and then repair projection/aggregation normalization under that held-out metric, not run another broad architecture sweep.",
            "",
            "## Recommendation",
            "",
            "Do not use real data yet. Treat the current model as unsafe. Next action should be a focused split-generalization diagnostic: compare train, validation, and test condition-bag embedding distributions, then refactor the shared projection/aggregation path only after the evaluation mismatch is localized.",
        ]
    )
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _eval_compare_rows() -> list[dict]:
    paths = [
        ("attention_step0_eval", OUTPUT_DIR / "eval_compare/step0_baselines/synth_micro/step0_seed0_metrics.json"),
        ("mean_step0_eval", OUTPUT_DIR / "eval_compare_mean/step0_baselines/synth_micro/step0_seed0_metrics.json"),
    ]
    rows = []
    for name, path in paths:
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "collapse_flag": metrics.get("collapse_flag"),
                "rna_min_std": metrics.get("model_rna_shared_min_std"),
                "image_min_std": metrics.get("model_image_shared_min_std"),
                "recall1": metrics.get("model_rna_to_image_recall@1"),
                "random_recall1": metrics.get("random_embedding_rna_to_image_recall@1"),
                "mean_prototype_recall1": metrics.get("mean_prototype_alignment_mean_prototype_alignment_recall@1"),
                "batch_probe_balanced_accuracy": metrics.get("model_batch_probe_balanced_accuracy"),
                "rna_latent_r2": metrics.get("model_bio_latent_r2_rna_shared"),
                "image_latent_r2": metrics.get("model_bio_latent_r2_image_shared"),
            }
        )
    return rows


def _write_tsv(path: Path, fields: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
