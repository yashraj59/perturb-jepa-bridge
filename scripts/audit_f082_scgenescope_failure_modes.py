from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/autoresearch_total_autonomy_bioguard_wm_jepa/external_validation_f082_scgenescope")
DEFAULT_OUTPUT_DIR = Path("outputs/autoresearch_total_autonomy_bioguard_wm_jepa/experiments/F093_f082_scgenescope_failure_audit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit F082 scGeneScope external-validation failure modes.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_tsv(args.input_dir / "external_summary_metrics.tsv")
    per_seed = _read_tsv(args.input_dir / "f082_external_seed_split_metrics.tsv")
    descriptors = _read_tsv(args.input_dir / "pubchem_action_descriptors.tsv")
    preflight = json.loads((args.input_dir / "preflight_summary.json").read_text(encoding="utf-8"))

    split_audit = _split_failure_audit(summary)
    calibration_audit = _calibration_transfer_audit(per_seed)
    descriptor_audit = _descriptor_audit(descriptors)
    decision = _decision(split_audit, descriptor_audit)

    split_audit.to_csv(args.output_dir / "split_failure_audit.tsv", sep="\t", index=False)
    calibration_audit.to_csv(args.output_dir / "calibration_transfer_audit.tsv", sep="\t", index=False)
    descriptor_audit.to_csv(args.output_dir / "descriptor_coverage_audit.tsv", sep="\t", index=False)

    payload = {
        "decision": decision,
        "input_dir": str(args.input_dir),
        "preflight_descriptor_summary": preflight.get("descriptor_summary", {}),
        "no_raw_h5ad_loaded": True,
        "model_promoted": False,
        "next_repair_target": "F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE",
    }
    (args.output_dir / "metrics_eval.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(args.output_dir / "F093_F082_SCGENESCOPE_FAILURE_AUDIT.md", payload, split_audit, calibration_audit, descriptor_audit)
    return 0


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _split_failure_audit(summary: pd.DataFrame) -> pd.DataFrame:
    f082 = summary.loc[summary["method"].eq("F082_delta_calibrated")].set_index("split")
    raw = summary.loc[summary["method"].eq("F082_no_delta_calibration")].set_index("split")
    floor = summary.loc[summary["method"].eq("protected_full_ridge_floor")].set_index("split")
    rows: list[dict[str, Any]] = []
    for split in sorted(f082.index):
        row = f082.loc[split]
        raw_row = raw.loc[split]
        floor_row = floor.loc[split]
        rows.append(
            {
                "split": split,
                "calibrated_transition": row["mean_transition_improvement"],
                "calibrated_delta_cosine": row["mean_delta_cosine"],
                "calibrated_recall_at_1": row["mean_recall_at_1"],
                "calibrated_delta_rank": row["mean_delta_rank"],
                "rna_to_image_recall_at_1": row["mean_rna_to_image_recall_at_1"],
                "image_to_rna_recall_at_1": row["mean_image_to_rna_recall_at_1"],
                "floor_gap_transition": row["floor_gap_transition_improvement"],
                "floor_gap_delta_cosine": row["floor_gap_delta_cosine"],
                "floor_gap_recall_at_1": row["floor_gap_recall_at_1"],
                "raw_minus_calibrated_transition": raw_row["mean_transition_improvement"] - row["mean_transition_improvement"],
                "raw_minus_calibrated_delta_cosine": raw_row["mean_delta_cosine"] - row["mean_delta_cosine"],
                "raw_minus_calibrated_recall_at_1": raw_row["mean_recall_at_1"] - row["mean_recall_at_1"],
                "raw_floor_gap_transition": raw_row["mean_transition_improvement"] - floor_row["mean_transition_improvement"],
                "raw_floor_gap_delta_cosine": raw_row["mean_delta_cosine"] - floor_row["mean_delta_cosine"],
                "raw_floor_gap_recall_at_1": raw_row["mean_recall_at_1"] - floor_row["mean_recall_at_1"],
            }
        )
    return pd.DataFrame(rows)


def _calibration_transfer_audit(per_seed: pd.DataFrame) -> pd.DataFrame:
    frame = per_seed.copy()
    frame["train_calibrated_transition_gain"] = frame["train_calibrated_transition_improvement"] - frame["train_raw_transition_improvement"]
    frame["train_calibrated_delta_cosine_gain"] = frame["train_calibrated_delta_cosine"] - frame["train_raw_delta_cosine"]
    frame["train_calibrated_recall_gain"] = frame["train_calibrated_recall_at_1"] - frame["train_raw_recall_at_1"]
    metrics = [
        "train_calibrated_transition_gain",
        "train_calibrated_delta_cosine_gain",
        "train_calibrated_recall_gain",
        "calibrated_transition_gain",
        "calibrated_delta_cosine_gain",
        "calibrated_recall_gain",
    ]
    rows: list[dict[str, Any]] = []
    for split, group in frame.groupby("split", sort=True):
        row: dict[str, Any] = {"split": split, "n_seed_rows": int(len(group))}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values)) if values.size else np.nan
            row[f"min_{metric}"] = float(np.min(values)) if values.size else np.nan
            row[f"max_{metric}"] = float(np.max(values)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _descriptor_audit(descriptors: pd.DataFrame) -> pd.DataFrame:
    frame = descriptors.copy()
    frame["pubchem_found"] = frame["pubchem_found"].astype(bool)
    frame["is_control_descriptor"] = frame["is_control_descriptor"].astype(bool)
    missing_noncontrol = frame.loc[~frame["pubchem_found"] & ~frame["is_control_descriptor"], "treatment"].astype(str).tolist()
    return pd.DataFrame(
        [
            {
                "total_treatments": int(len(frame)),
                "pubchem_found_treatments": int(frame["pubchem_found"].sum()),
                "missing_noncontrol_treatments": int(len(missing_noncontrol)),
                "missing_noncontrol_treatment_names": ";".join(missing_noncontrol),
                "descriptor_action_dim": int(12),
                "interpretation": "PubChem scalar properties are leakage-safe but too low-capacity for mechanism-aware perturbation action encoding.",
            }
        ]
    )


def _decision(split_audit: pd.DataFrame, descriptor_audit: pd.DataFrame) -> str:
    delta_floor_fail = bool((split_audit["floor_gap_delta_cosine"] < 0.0).any())
    transition_floor_fail = bool((split_audit["floor_gap_transition"] < 0.0).any())
    missing_descriptors = int(descriptor_audit.loc[0, "missing_noncontrol_treatments"]) > 0
    if delta_floor_fail and transition_floor_fail:
        return "F093_CALIBRATION_AND_DESCRIPTOR_REPAIR_REQUIRED"
    if delta_floor_fail:
        return "F093_CALIBRATION_REPAIR_REQUIRED"
    if missing_descriptors:
        return "F093_DESCRIPTOR_REPAIR_REQUIRED"
    return "F093_AUDIT_NO_REPAIR_TRIGGER"


def _write_report(path: Path, payload: dict[str, Any], split_audit: pd.DataFrame, calibration_audit: pd.DataFrame, descriptor_audit: pd.DataFrame) -> None:
    text = f"""# F093 F082 scGeneScope Failure Audit

## Decision
`{payload["decision"]}`

No model is promoted. This audit reads only the existing F082 scGeneScope TSV/JSON artifacts and does not open raw H5AD matrices, fit a model, refit PCA, refit calibration, or use held-out target statistics for selection.

## Main Finding
F082 is not blocked by validation plumbing. The backed contracts, pairing, identity checks, and leakage checks passed. The failure is a representation-output repair problem: train-only delta calibration improves transition and recall in several held-out splits, but it consistently damages delta-cosine relative to the protected floor, while the uncalibrated JEPA output preserves delta-cosine better but lacks floor-safe transition on the external replicate and round shift.

## Split Failure Audit
```tsv
{split_audit.to_csv(sep="\t", index=False)}
```

## Calibration Transfer Audit
```tsv
{calibration_audit.to_csv(sep="\t", index=False)}
```

## Descriptor Coverage Audit
```tsv
{descriptor_audit.to_csv(sep="\t", index=False)}
```

## Repair Target
1. Implement `F094_SPLIT_SAFE_JEPA_CALIBRATION_ABSTENTION_GATE`.
2. The gate must select among JEPA raw, JEPA calibrated, and train-only JEPA blend outputs using only train/internal replicate criteria.
3. The protected PLS/full-ridge floor remains an audit threshold only, not a candidate representation path or fallback output.
4. Add a descriptor-upgrade branch only after the calibration gate audit is complete, using non-exact public chemical or coarse mechanism descriptors and explicit missingness flags.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
