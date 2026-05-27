from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EXPERIMENT_ROOT = Path(
    "outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments"
)
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "F099_rosetta_source_geometry_audit"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "F099_ROSETTA_SOURCE_GEOMETRY_AUDIT.md"
DEFAULT_INPUTS = (
    DEFAULT_EXPERIMENT_ROOT / "F097_cpg0003_rosetta_external_confirmation" / "metrics_eval.json",
    DEFAULT_EXPERIMENT_ROOT / "F098_cpg0003_rosetta_replicate_holdout" / "metrics_eval.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Artifact-only source-geometry audit for cpg0003 Rosetta F097/F098 failures."
    )
    parser.add_argument("--metrics-json", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = [_load_payload(path) for path in args.metrics_json]
    seed_frame = _seed_geometry_frame(payloads)
    summary = _summary_frame(seed_frame)
    diagnosis = _diagnose(seed_frame, summary, payloads)

    seed_frame.to_csv(args.output_dir / "f099_seed_split_geometry.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "f099_geometry_summary.tsv", sep="\t", index=False)
    _write_json(args.output_dir / "f099_diagnosis.json", diagnosis)
    _write_report(args.report_path, diagnosis, summary, payloads)
    return 0


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    payload["_path"] = str(path)
    return payload


def _seed_geometry_frame(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        method = str(payload.get("candidate_method", "unknown"))
        experiment_id = method.split("_", 1)[0]
        preflight = payload.get("preflight", {})
        split_mode = str(preflight.get("split_mode", "compound_holdout"))
        for row in payload.get("per_seed_split_metrics", []):
            direct_abs = float(row["transition_image_cosine"])
            direct_improvement = float(row["image_teacher_transition_improvement"])
            source_to_target = direct_abs - direct_improvement
            calibrated_improvement = float(row["calibrated_transition_improvement"])
            calibrated_abs = source_to_target + calibrated_improvement
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "split_mode": split_mode,
                    "split": row["split"],
                    "seed": int(row["seed"]),
                    "source_to_target_cosine": source_to_target,
                    "direct_absolute_target_cosine": direct_abs,
                    "direct_transition_improvement": direct_improvement,
                    "calibrated_absolute_target_cosine": calibrated_abs,
                    "calibrated_transition_improvement": calibrated_improvement,
                    "calibrated_absolute_drop_vs_source": calibrated_abs - source_to_target,
                    "direct_delta_cosine": float(row["image_teacher_delta_cosine"]),
                    "calibrated_delta_cosine": float(row["calibrated_delta_cosine"]),
                    "direct_magnitude_ratio": float(row["image_teacher_magnitude_ratio"]),
                    "calibrated_magnitude_ratio": float(row["calibrated_magnitude_ratio"]),
                    "target_encoder_anchor_cosine": float(row["target_image_cosine"]),
                    "source_encoder_anchor_cosine": float(row["source_image_cosine"]),
                    "source_effective_rank": float(row["source_image_effective_rank"]),
                    "target_effective_rank": float(row["target_image_effective_rank"]),
                    "identity_violation": float(row["identity_violation"]),
                    "leakage_flag": float(row["leakage_flag"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No per_seed_split_metrics rows found in supplied payloads.")
    return frame


def _summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "source_to_target_cosine",
        "direct_absolute_target_cosine",
        "direct_transition_improvement",
        "calibrated_absolute_target_cosine",
        "calibrated_transition_improvement",
        "calibrated_absolute_drop_vs_source",
        "direct_delta_cosine",
        "calibrated_delta_cosine",
        "direct_magnitude_ratio",
        "calibrated_magnitude_ratio",
        "target_encoder_anchor_cosine",
        "source_encoder_anchor_cosine",
        "source_effective_rank",
        "target_effective_rank",
        "identity_violation",
        "leakage_flag",
    ]
    grouped = frame.groupby(["experiment_id", "split_mode", "split"], sort=True)[metrics]
    summary = grouped.agg(["mean", "min", "max"]).reset_index()
    summary.columns = [
        "_".join([part for part in column if part]).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _diagnose(
    seed_frame: pd.DataFrame,
    summary: pd.DataFrame,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    f098 = seed_frame[seed_frame["experiment_id"] == "F098"]
    focus = f098 if not f098.empty else seed_frame
    high_source_cosine = float(focus["source_to_target_cosine"].mean())
    direct_improvement = float(focus["direct_transition_improvement"].mean())
    calibrated_improvement = float(focus["calibrated_transition_improvement"].mean())
    calibrated_abs = float(focus["calibrated_absolute_target_cosine"].mean())
    source_rank = float(focus["source_effective_rank"].mean())
    target_rank = float(focus["target_effective_rank"].mean())
    direct_delta = float(focus["direct_delta_cosine"].mean())
    calibrated_delta = float(focus["calibrated_delta_cosine"].mean())
    identity_max = float(seed_frame["identity_violation"].max())
    leakage_max = float(seed_frame["leakage_flag"].max())

    source_state_contract_failure = (
        high_source_cosine >= 0.90
        and source_rank <= 1.05
        and direct_improvement >= 0.0
        and calibrated_improvement < -0.10
    )
    caveat_text = " ".join(str(payload.get("promotion_caveat", "")) for payload in payloads).lower()
    validator_mismatch = bool(
        source_state_contract_failure
        and "l1000" in caveat_text
        and "scrna" in caveat_text
    )
    decision = (
        "F099_SOURCE_STATE_CONTRACT_FAILURE_AND_VALIDATOR_MISMATCH_NO_PROMOTION"
        if source_state_contract_failure or validator_mismatch
        else "F099_INCONCLUSIVE_ROSETTA_GEOMETRY_AUDIT_NO_PROMOTION"
    )

    return {
        "experiment_id": "F099",
        "decision": decision,
        "model_promoted": False,
        "identity_violation_max": identity_max,
        "leakage_flag_max": leakage_max,
        "focus_experiment": "F098" if not f098.empty else "all",
        "mean_source_to_target_cosine": high_source_cosine,
        "mean_direct_transition_improvement": direct_improvement,
        "mean_calibrated_transition_improvement": calibrated_improvement,
        "mean_calibrated_absolute_target_cosine": calibrated_abs,
        "mean_source_effective_rank": source_rank,
        "mean_target_effective_rank": target_rank,
        "mean_direct_delta_cosine": direct_delta,
        "mean_calibrated_delta_cosine": calibrated_delta,
        "source_state_contract_failure": bool(source_state_contract_failure),
        "validator_mismatch": bool(validator_mismatch),
        "interpretation": (
            "F098 shows high source-as-target cosine with a rank-one repeated "
            "control source. The uncalibrated JEPA transition slightly improves "
            "absolute target cosine but points opposite the measured small delta; "
            "the train-only delta calibrator improves delta cosine while damaging "
            "absolute target cosine. This is best treated as a Rosetta source-state "
            "contract failure plus validator mismatch, not as promotion evidence "
            "and not as a reason to redesign the architecture before a strict "
            "fresh scRNA+imaging validation."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_report(
    path: Path,
    diagnosis: dict[str, Any],
    summary: pd.DataFrame,
    payloads: list[dict[str, Any]],
) -> None:
    lines = [
        "# F099 Rosetta Source Geometry Audit",
        "",
        "## Decision",
        f"`{diagnosis['decision']}`",
        "",
        "No model is promoted. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.",
        "",
        "## Scope",
        "- artifact-only audit of F097/F098 cpg0003 Rosetta outputs",
        "- no model training and no raw data loading",
        "- cpg0003 Rosetta is L1000 plus Cell Painting, not strict scRNA plus imaging",
        "",
        "## Focus Metrics",
        f"- focus experiment: `{diagnosis['focus_experiment']}`",
        f"- mean source-to-target cosine: `{diagnosis['mean_source_to_target_cosine']:.6f}`",
        f"- mean direct transition improvement: `{diagnosis['mean_direct_transition_improvement']:.6f}`",
        f"- mean calibrated transition improvement: `{diagnosis['mean_calibrated_transition_improvement']:.6f}`",
        f"- mean calibrated absolute target cosine: `{diagnosis['mean_calibrated_absolute_target_cosine']:.6f}`",
        f"- mean source effective rank: `{diagnosis['mean_source_effective_rank']:.6f}`",
        f"- mean target effective rank: `{diagnosis['mean_target_effective_rank']:.6f}`",
        f"- mean direct delta cosine: `{diagnosis['mean_direct_delta_cosine']:.6f}`",
        f"- mean calibrated delta cosine: `{diagnosis['mean_calibrated_delta_cosine']:.6f}`",
        f"- identity violation max: `{diagnosis['identity_violation_max']:.6f}`",
        f"- leakage flag max: `{diagnosis['leakage_flag_max']:.6f}`",
        "",
        "## Geometry Summary",
        "```tsv",
        summary.to_csv(sep="\t", index=False).strip(),
        "```",
        "",
        "## Interpretation",
        diagnosis["interpretation"],
        "",
        "## Inputs",
    ]
    for payload in payloads:
        lines.append(f"- `{payload.get('_path', 'unknown')}`")
    lines.extend(
        [
            "",
            "## Next Step",
            "Do not promote F097/F098. If cpg0003 remains useful as an auxiliary validator, run a source-state contract preflight before any new model work. A fresh strict scRNA+imaging Tier 3 confirmation is still required for promotion.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
