from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.training.biospectral_operator import bundle_features, load_latent_bundle, paired_transition_metrics, write_leakage_report


PHASE4_CACHE = Path("outputs/autoresearch_bioflow_jepa_phase4/delta_operator_audit")
FULL_FLOOR = {
    "transition_source_cosine_improvement": 0.0057,
    "delta_cosine": 0.3980,
    "transition_to_target_recall@1": 0.4815,
    "delta_prediction_effective_rank": 10.2835,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved Phase 6 BioSpectral operator artifact.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="synth_genetic_anchor_lite")
    parser.add_argument("--eval-split", default="test_heldout_perturbation")
    parser.add_argument("--latent-cache", type=Path, default=PHASE4_CACHE)
    parser.add_argument("--output-name", default="reevaluation_metrics.json")
    args = parser.parse_args()

    payload = json.loads((args.artifact_dir / "operator_payload.json").read_text(encoding="utf-8"))
    train = load_latent_bundle(args.latent_cache / f"{args.dataset}_train_latents", "train")
    eval_bundle = load_latent_bundle(args.latent_cache / f"{args.dataset}_{args.eval_split}_latents", "eval")
    train_delta = predict_from_payload(payload, train)
    eval_delta = predict_from_payload(payload, eval_bundle)
    metrics = paired_transition_metrics(
        train,
        eval_bundle,
        train_delta,
        eval_delta,
        floor_eval_improvement=FULL_FLOOR["transition_source_cosine_improvement"],
        floor_eval_delta_cosine=FULL_FLOOR["delta_cosine"],
        floor_eval_recall_at_1=FULL_FLOOR["transition_to_target_recall@1"],
        floor_eval_rank=FULL_FLOOR["delta_prediction_effective_rank"],
    )
    metrics["artifact_dir"] = str(args.artifact_dir)
    (args.artifact_dir / args.output_name).write_text(json.dumps(jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_leakage_report(
        args.artifact_dir / "reevaluation_leakage_report.md",
        train_rows=train.source.shape[0],
        eval_rows=eval_bundle.source.shape[0],
        action_feature_names=[f"action_descriptor_{idx}" for idx in range(train.action.shape[1])],
        mode="reevaluate_saved_biospectral_artifact",
    )
    print(json.dumps(jsonable(metrics), sort_keys=True))
    return 0


def predict_from_payload(payload: dict[str, Any], bundle) -> np.ndarray:
    mode = payload.get("mode", "")
    x = bundle_features(bundle)
    if mode == "frozen_neural_low_rank_equivalence":
        weight = np.asarray(payload["linear_weight"], dtype=np.float64)
        bias = np.asarray(payload["linear_bias"], dtype=np.float64)
        basis = np.asarray(payload["basis"], dtype=np.float64)
        delta_mean = np.asarray(payload["delta_mean"], dtype=np.float64)
        coeff = x @ weight.T + bias
        return delta_mean + coeff @ basis
    if "ridge_floor" in payload:
        ridge = payload["ridge_floor"]
        return predict_ridge_payload(ridge, x)
    if "full_floor" in payload:
        return predict_ridge_payload(payload["full_floor"], x)
    raise ValueError(f"unsupported payload mode: {mode}")


def predict_ridge_payload(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x_mean = np.asarray(payload["x_mean"], dtype=np.float64)
    y_mean = np.asarray(payload["y_mean"], dtype=np.float64)
    coef = np.asarray(payload["coef"], dtype=np.float64)
    return (x - x_mean) @ coef + y_mean


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
