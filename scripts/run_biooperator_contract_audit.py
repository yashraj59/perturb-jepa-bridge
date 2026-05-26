from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.evaluation.retrieval import directional_retrieval_metrics


PHASE5_ROOT = Path("outputs/autoresearch_biooperator_jepa_phase5")
PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
REFERENCE_RIDGE_EVAL_IMPROVEMENT = 0.0057


@dataclass(frozen=True)
class LatentBundle:
    name: str
    arrays: dict[str, np.ndarray]
    metadata: pd.DataFrame

    @property
    def source(self) -> np.ndarray:
        return self.arrays["source_z_bio_teacher"].astype(np.float64)

    @property
    def target(self) -> np.ndarray:
        return self.arrays["target_z_bio_teacher"].astype(np.float64)

    @property
    def delta(self) -> np.ndarray:
        return self.target - self.source

    @property
    def action(self) -> np.ndarray:
        action = self.arrays.get("action_descriptor")
        if action is None or action.size == 0:
            return np.zeros((self.source.shape[0], 0), dtype=np.float64)
        return action.astype(np.float64)


@dataclass(frozen=True)
class RidgeFit:
    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray


@dataclass(frozen=True)
class LowRankRidgeFit:
    ridge: RidgeFit
    delta_mean: np.ndarray
    basis: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5 BioOperator contract audit.")
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--output-dir", type=Path, default=PHASE5_ROOT / "contract_audit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0e-2)
    parser.add_argument("--low-rank", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    PHASE5_ROOT.mkdir(parents=True, exist_ok=True)
    initialize_phase5_docs()
    train = load_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")
    rows, predictions = reproduce_floors(train, eval_bundle, alpha=args.alpha, low_rank=args.low_rank)
    write_predictions(args.output_dir / "predictions", predictions)
    decision = contract_decision(rows)
    write_contract_audit(args=args, rows=rows, decision=decision)
    update_results(args=args, rows=rows, decision=decision)
    if decision == "PHASE5_STOP_OPERATOR_FLOOR_NOT_REPRODUCED":
        write_final_report(
            decision="PHASE5_STOP_OPERATOR_FLOOR_NOT_REPRODUCED",
            rows=rows,
            recommendation="Close architecture search and debug metric/cache reproduction before adding neural operators.",
        )
    print(json.dumps({"decision": decision, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if decision == "PHASE5_OPERATOR_FLOOR_REPRODUCED" else 2


def initialize_phase5_docs() -> None:
    files = {
        "results.tsv": "\t".join(
            [
                "commit",
                "experiment_num",
                "stage",
                "family",
                "tier_reached",
                "decision_label",
                "status",
                "dataset",
                "eval_split",
                "seed_list",
                "primary_metric",
                "secondary_metric",
                "protected_metric_summary",
                "architectural_change",
                "description",
                "operator_train_transition_improvement",
                "operator_eval_transition_improvement",
                "operator_train_delta_cosine",
                "operator_eval_delta_cosine",
                "operator_eval_recall_at_1",
                "operator_eval_median_rank",
                "operator_predicted_delta_rank",
                "action_ridge_floor_gap",
                "sign_contract_pass",
                "ridge_equivalence_pass",
                "source_improvement_hinge_violation_fraction",
            ]
        )
        + "\n",
        "research_journal.md": "# BioOperator-JEPA Phase 5 Research Journal\n\n",
        "architectural_changes_log.md": "# Phase 5 Architectural Changes Log\n\nNo BioOperator model code before Stage A/B contracts pass.\n",
        "identity_violations_considered.md": "# Phase 5 Identity Violations Considered\n\n- PLS/raw-linear main path: forbidden; retained only as protected model-of-record reference.\n- `condition_key` or exact biological-key one-hot as input: forbidden.\n- Test target means or pooled train+test targets: forbidden.\n",
        "papers_consulted.md": papers_text(),
        "BASELINE_REGISTRY.md": baseline_registry_text(),
    }
    for name, text in files.items():
        path = PHASE5_ROOT / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def baseline_registry_text() -> str:
    return """# Phase 5 Baseline Registry

## Active Model Of Record

- Protected model: rank-3 train-split-only PLS raw-linear readout
- Role: protected baseline and audit reference only
- Status: no Phase 1-4 JEPA candidate has been promoted
- PLS restriction: never used as the BioOperator-JEPA main representation path

## Phase 4 Frozen-Latent Reference

- Dataset: `synth_genetic_anchor_lite`
- Eval split: `test_heldout_perturbation`
- Latent cache model: BTJ001 BioTech-JEPA checkpoint
- Forbidden shortcuts checked: no `condition_key`, no biological-key one-hot, no test target means, no pooled train+test targets

| Metric | Value |
| --- | ---: |
| train delta effective rank | 13.5627 |
| train delta mean norm | 0.4310 |
| train delta std mean | 0.0846 |
| train source-to-target cosine mean | 0.8977 |
| eval delta effective rank | 11.7819 |
| eval delta mean norm | 0.4252 |
| eval delta std mean | 0.0832 |
| eval source-to-target cosine mean | 0.9031 |
| eval action_ridge_delta transition improvement | +0.0057 |
| eval action_ridge_delta delta cosine | 0.3980 |
| eval action_ridge_delta delta rank | 10.2835 |
| eval action_low_rank_ridge transition improvement | +0.0046 |
| train action_ridge_delta transition improvement | +0.0769 |
| BFJ001 transition improvement | -0.0104 |
| BFJ001 delta cosine | -0.1054 |

Raw sources:

- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_OPERATOR_AUDIT.md`
- `outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit/DELTA_BASELINE_RESULTS.tsv`
- `outputs/autoresearch_bioflow_jepa_phase4/final_report.md`
"""


def papers_text() -> str:
    return """# Papers Consulted

| Source | Field | Mechanism Extracted | Phase 5 Use |
| --- | --- | --- | --- |
| V-JEPA 2 / V-JEPA 2-AC, https://arxiv.org/abs/2506.09985 | latent world models | stop-gradient latent prediction with action-conditioned transition | used as identity constraint for future BioOperator-JEPA wrapper |
| Flow Matching for Generative Modeling, https://arxiv.org/abs/2210.02747 | vector-field learning | learn vector fields against known trajectories | deferred; Phase 5 first requires ridge/operator contracts |
| Efficient Flow Matching using Latent Variables, https://arxiv.org/html/2505.04486v2 | latent flow matching | latent-variable transport efficiency | deferred until deterministic operator floor is passed |
| Koopman/control-inspired latent dynamics, https://openreview.net/forum?id=fkrYDQaHOJ | controlled dynamics | low-rank or structured action-conditioned linear operator | used in planned low-rank control-affine operator |
| CellOT, https://www.nature.com/articles/s41592-023-01969-x | single-cell perturbation transport | preserve population structure across perturbation maps | deferred until single-condition latent operator contracts pass |
| Distributional Transport for Single-Cell Perturbation Prediction, https://arxiv.org/html/2511.13124v1 | stochastic transport | stochastic dynamic maps for unpaired populations | rejected for Phase 5 Tier 1; stochastic transport would bypass deterministic floor |
"""


def load_bundle(prefix: Path, name: str) -> LatentBundle:
    arrays = dict(np.load(prefix.with_suffix(".npz")))
    metadata = pd.read_csv(prefix.with_suffix(".metadata.tsv"), sep="\t")
    return LatentBundle(name=name, arrays=arrays, metadata=metadata)


def reproduce_floors(
    train: LatentBundle,
    eval_bundle: LatentBundle,
    *,
    alpha: float,
    low_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    ridge = fit_ridge(features(train), train.delta, alpha=alpha)
    low_rank_ridge = fit_low_rank_ridge(features(train), train.delta, rank=low_rank, alpha=alpha)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for split, bundle in (("train", train), ("eval", eval_bundle)):
        ridge_delta = predict_ridge(ridge, features(bundle))
        low_rank_delta = predict_low_rank_ridge(low_rank_ridge, features(bundle))
        for name, delta in (
            ("action_ridge_delta", ridge_delta),
            ("action_low_rank_ridge", low_rank_delta),
        ):
            pred = bundle.source + delta
            rows.append({"split": split, "operator": name, **transition_metrics(bundle, pred)})
            predictions[f"{split}_{name}_delta"] = delta.astype(np.float32)
            predictions[f"{split}_{name}_target_pred"] = pred.astype(np.float32)
    return rows, predictions


def features(bundle: LatentBundle) -> np.ndarray:
    return np.concatenate((bundle.source, bundle.action), axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> RidgeFit:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    eye = np.eye(x.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xc.T @ xc + float(alpha) * eye, xc.T @ yc)
    return RidgeFit(x_mean=x_mean, y_mean=y_mean, coef=coef)


def predict_ridge(fit: RidgeFit, x: np.ndarray) -> np.ndarray:
    return (x - fit.x_mean) @ fit.coef + fit.y_mean


def fit_low_rank_ridge(x: np.ndarray, delta: np.ndarray, *, rank: int, alpha: float) -> LowRankRidgeFit:
    mean = delta.mean(axis=0, keepdims=True)
    centered = delta - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: max(1, min(int(rank), vt.shape[0]))]
    coeff = centered @ basis.T
    return LowRankRidgeFit(ridge=fit_ridge(x, coeff, alpha=alpha), delta_mean=mean, basis=basis)


def predict_low_rank_ridge(fit: LowRankRidgeFit, x: np.ndarray) -> np.ndarray:
    return fit.delta_mean + predict_ridge(fit.ridge, x) @ fit.basis


def transition_metrics(bundle: LatentBundle, pred: np.ndarray) -> dict[str, float]:
    source = bundle.source
    target = bundle.target
    true_delta = target - source
    pred_delta = pred - source
    pred_cos = row_cosine(pred, target)
    source_cos = row_cosine(source, target)
    delta_cos = row_cosine(pred_delta, true_delta)
    true_norm = np.linalg.norm(true_delta, axis=1)
    pred_norm = np.linalg.norm(pred_delta, axis=1)
    frame = metrics_frame(bundle)
    metrics = {
        "transition_source_cosine_improvement": float((pred_cos - source_cos).mean()),
        "absolute_target_cosine": float(pred_cos.mean()),
        "source_as_target_bio_cosine_to_teacher": float(source_cos.mean()),
        "delta_cosine": float(delta_cos.mean()),
        "delta_magnitude_ratio": float(np.mean(pred_norm / np.maximum(true_norm, 1.0e-8))),
        "delta_prediction_effective_rank": effective_rank(pred_delta),
        "delta_teacher_effective_rank": effective_rank(true_delta),
        "source_improvement_hinge_violation_fraction": float((pred_cos < source_cos + 0.02).mean()),
    }
    metrics.update(
        directional_retrieval_metrics(
            l2_normalize(pred),
            l2_normalize(target),
            frame,
            frame,
            label_col="condition_key",
            ks=(1, 5, 10),
            prefix="transition_to_target",
            stratify_by=(),
        )
    )
    return metrics


def contract_decision(rows: list[dict[str, Any]]) -> str:
    eval_ridge = _row(rows, split="eval", operator="action_ridge_delta")
    improvement = float(eval_ridge["transition_source_cosine_improvement"])
    delta_cosine = float(eval_ridge["delta_cosine"])
    rank = float(eval_ridge["delta_prediction_effective_rank"])
    pass_gate = (
        abs(improvement - REFERENCE_RIDGE_EVAL_IMPROVEMENT) <= 0.0020
        and delta_cosine >= 0.35
        and rank >= 8.0
    )
    return "PHASE5_OPERATOR_FLOOR_REPRODUCED" if pass_gate else "PHASE5_STOP_OPERATOR_FLOOR_NOT_REPRODUCED"


def write_predictions(path: Path, predictions: dict[str, np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "operator_floor_predictions.npz", **predictions)


def write_contract_audit(*, args: argparse.Namespace, rows: list[dict[str, Any]], decision: str) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "operator_floor_results.tsv", sep="\t", index=False)
    eval_ridge = _row(rows, split="eval", operator="action_ridge_delta")
    eval_low = _row(rows, split="eval", operator="action_low_rank_ridge")
    train_ridge = _row(rows, split="train", operator="action_ridge_delta")
    lines = [
        "# BioOperator-JEPA Phase 5 Operator Contract Audit",
        "",
        "## Scope",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Eval split: `{args.eval_split}`",
        f"- Latent cache: `{args.latent_cache}`",
        "- Stage: BOJ000 / Stage A, no BioOperator model training.",
        "- Forbidden shortcuts: no condition-key input, no biological-key one-hot, no test target means, no pooled train+test targets.",
        "",
        "## Reproduced Floors",
        "",
        markdown_table(frame),
        "",
        "## Stage A Gate",
        "",
        f"- eval action_ridge_delta transition improvement: `{float(eval_ridge['transition_source_cosine_improvement']):.4f}` vs required `0.0057 +/- 0.0020`",
        f"- eval action_ridge_delta delta cosine: `{float(eval_ridge['delta_cosine']):.4f}` vs required `>= 0.35`",
        f"- eval action_ridge_delta rank: `{float(eval_ridge['delta_prediction_effective_rank']):.4f}` vs required `>= 8.0`",
        f"- eval action_low_rank_ridge transition improvement: `{float(eval_low['transition_source_cosine_improvement']):.4f}`",
        f"- train action_ridge_delta transition improvement: `{float(train_ridge['transition_source_cosine_improvement']):.4f}`",
        "",
        f"Decision label: `{decision}`",
        "",
        "## Interpretation",
        "",
        "The analytical train-only frozen-latent action-ridge floor was reproduced from the Phase 4 cache using a common transition metric function. If this gate passes, Phase 5 may proceed to sign/gradient/loss contract tests before any BioOperator model training.",
    ]
    text = "\n".join(lines) + "\n"
    (args.output_dir / "operator_contract_audit.md").write_text(text, encoding="utf-8")
    (PHASE5_ROOT / "operator_contract_audit.md").write_text(text, encoding="utf-8")
    with (PHASE5_ROOT / "research_journal.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## BOJ000: Operator Floor Reproduction\n\n"
            "**Hypothesis**: before neural operators, the Phase 4 train-only action-ridge floor must be exactly reproducible from cached teacher latents.\n\n"
            f"**Result**: eval ridge improvement `{float(eval_ridge['transition_source_cosine_improvement']):.4f}`, delta cosine `{float(eval_ridge['delta_cosine']):.4f}`, rank `{float(eval_ridge['delta_prediction_effective_rank']):.4f}`.\n\n"
            f"**Decision label**: `{decision}`.\n\n"
        )


def update_results(*, args: argparse.Namespace, rows: list[dict[str, Any]], decision: str) -> None:
    eval_ridge = _row(rows, split="eval", operator="action_ridge_delta")
    train_ridge = _row(rows, split="train", operator="action_ridge_delta")
    row = {
        "commit": git_commit_label(),
        "experiment_num": "BOJ000",
        "stage": "StageA",
        "family": "diagnostics",
        "tier_reached": "audit",
        "decision_label": decision,
        "status": decision,
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "seed_list": "0",
        "primary_metric": f"eval_action_ridge_transition_improvement={float(eval_ridge['transition_source_cosine_improvement']):.4f}",
        "secondary_metric": f"eval_delta_cosine={float(eval_ridge['delta_cosine']):.4f}; eval_rank={float(eval_ridge['delta_prediction_effective_rank']):.4f}",
        "protected_metric_summary": "protected_rank3_train_split_pls_remains_model_of_record; no BioOperator model before contracts",
        "architectural_change": "none_contract_audit_only",
        "description": "Reproduced Phase 4 frozen-latent train-only action-ridge and low-rank ridge floors.",
        "operator_train_transition_improvement": f"{float(train_ridge['transition_source_cosine_improvement']):.4f}",
        "operator_eval_transition_improvement": f"{float(eval_ridge['transition_source_cosine_improvement']):.4f}",
        "operator_train_delta_cosine": f"{float(train_ridge['delta_cosine']):.4f}",
        "operator_eval_delta_cosine": f"{float(eval_ridge['delta_cosine']):.4f}",
        "operator_eval_recall_at_1": f"{float(eval_ridge['transition_to_target_recall@1']):.4f}",
        "operator_eval_median_rank": f"{float(eval_ridge['transition_to_target_median_rank']):.4f}",
        "operator_predicted_delta_rank": f"{float(eval_ridge['delta_prediction_effective_rank']):.4f}",
        "action_ridge_floor_gap": f"{float(eval_ridge['transition_source_cosine_improvement']) - REFERENCE_RIDGE_EVAL_IMPROVEMENT:.4f}",
        "sign_contract_pass": "",
        "ridge_equivalence_pass": "",
        "source_improvement_hinge_violation_fraction": f"{float(eval_ridge['source_improvement_hinge_violation_fraction']):.4f}",
    }
    append_or_replace_result(row)


def write_final_report(*, decision: str, rows: list[dict[str, Any]], recommendation: str) -> None:
    eval_ridge = _row(rows, split="eval", operator="action_ridge_delta")
    lines = [
        "# BioOperator-JEPA Phase 5 Final Report",
        "",
        "## Decision label",
        "",
        decision,
        "",
        "## Model of record",
        "",
        "Protected rank-3 train-split-only PLS raw-linear readout remains model of record unless a future Tier 3 pass explicitly supersedes it.",
        "",
        "## What was tested",
        "",
        "- Stage A operator floor reproduction",
        "",
        "## Key metrics",
        "",
        "| experiment | transition improvement | delta cosine | recall@1 | delta rank | magnitude ratio | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| BOJ000 action_ridge_delta | {float(eval_ridge['transition_source_cosine_improvement']):.4f} | {float(eval_ridge['delta_cosine']):.4f} | {float(eval_ridge['transition_to_target_recall@1']):.4f} | {float(eval_ridge['delta_prediction_effective_rank']):.4f} | {float(eval_ridge['delta_magnitude_ratio']):.4f} | {decision} |",
        "",
        "## Floor comparison",
        "",
        "- eval action_ridge_delta improvement reference: `+0.0057`",
        "- eval action_ridge_delta delta cosine reference: `0.3980`",
        "- eval action_ridge_delta rank reference: `10.2835`",
        "",
        "## What failed or passed",
        "",
        "Failure occurred in metric/cache reproduction before neural operator implementation.",
        "",
        "## Recommendation",
        "",
        recommendation,
    ]
    (PHASE5_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_or_replace_result(row: dict[str, Any]) -> None:
    path = PHASE5_ROOT / "results.tsv"
    if not path.exists():
        initialize_phase5_docs()
    frame = pd.read_csv(path, sep="\t")
    if "experiment_num" in frame.columns:
        frame = frame[frame["experiment_num"].astype(str) != str(row["experiment_num"])]
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)


def metrics_frame(bundle: LatentBundle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_key": bundle.metadata["condition_key"].astype(str),
            "perturbation": "pert_" + bundle.metadata["perturbation_id"].astype(str),
            "batch": "batch_" + bundle.metadata["batch_id"].astype(str),
            "cell_line": "cell_" + bundle.metadata["cell_line_id"].astype(str),
            "dose": "ignored",
            "time": "0",
        }
    )


def row_cosine(left: np.ndarray, right: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), eps)
    return np.sum(left * right, axis=1) / denom


def l2_normalize(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def effective_rank(values: np.ndarray, eps: float = 1.0e-12) -> float:
    if values.ndim != 2 or values.shape[0] < 2:
        return 0.0
    centered = values - values.mean(axis=0, keepdims=True)
    spectrum = np.linalg.svd(centered, compute_uv=False)
    total = spectrum.sum()
    if total <= eps:
        return 0.0
    probs = spectrum / total
    return float(math.exp(-float(np.sum(probs * np.log(np.maximum(probs, eps))))))


def markdown_table(frame: pd.DataFrame) -> str:
    keep = [
        "split",
        "operator",
        "transition_source_cosine_improvement",
        "transition_to_target_recall@1",
        "transition_to_target_median_rank",
        "delta_cosine",
        "delta_magnitude_ratio",
        "delta_prediction_effective_rank",
        "source_improvement_hinge_violation_fraction",
    ]
    frame = frame[keep].copy()
    columns = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            values.append(f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append(values)
    widths = [max(len(column), *(len(row[index]) for row in rows)) for index, column in enumerate(columns)]
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _row(rows: list[dict[str, Any]], *, split: str, operator: str) -> dict[str, Any]:
    for row in rows:
        if row["split"] == split and row["operator"] == operator:
            return row
    raise KeyError(f"missing {split}/{operator}")


def git_commit_label() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
        return f"{commit}+dirty" if dirty else commit
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
