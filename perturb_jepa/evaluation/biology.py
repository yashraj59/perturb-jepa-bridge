from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import json
import math
import struct
import zlib

import numpy as np
import pandas as pd


NOT_AVAILABLE = "not available"


@dataclass(frozen=True)
class ValidationResult:
    metrics: pd.DataFrame
    de_table: pd.DataFrame
    pathway_scores: pd.DataFrame
    moa_enrichment: pd.DataFrame
    dose_time: pd.DataFrame
    messages: list[str]


def pseudobulk_aggregate(
    expression: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    groupby: str | Sequence[str],
    gene_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Average cell-level expression within metadata groups."""

    values = _as_2d(expression, name="expression")
    frame = _metadata_frame(metadata, n_rows=values.shape[0])
    group_cols = _columns(groupby)
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"metadata is missing pseudobulk columns: {missing}")
    genes = _gene_names(values.shape[1], gene_names)

    expression_frame = pd.DataFrame(values, columns=genes)
    grouped = pd.concat([frame[group_cols].reset_index(drop=True), expression_frame], axis=1)
    means = grouped.groupby(group_cols, sort=True, dropna=False)[genes].mean().reset_index()
    counts = grouped.groupby(group_cols, sort=True, dropna=False).size().reset_index(name="n_cells")
    return counts.merge(means, on=group_cols, how="left")


def log_fold_change(treated: np.ndarray, control: np.ndarray, *, pseudocount: float = 1e-6) -> np.ndarray:
    """Compute log2 fold-change with a small pseudocount."""

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    treated_values = np.asarray(treated, dtype=float)
    control_values = np.asarray(control, dtype=float)
    return np.log2((treated_values + pseudocount) / (control_values + pseudocount))


def topk_de_overlap(predicted_lfc: np.ndarray, observed_lfc: np.ndarray, *, k: int) -> float:
    """Fraction of observed top-k absolute DE genes recovered by predictions."""

    if k <= 0:
        raise ValueError("k must be positive")
    predicted = np.asarray(predicted_lfc, dtype=float).ravel()
    observed = np.asarray(observed_lfc, dtype=float).ravel()
    if predicted.shape != observed.shape:
        raise ValueError("predicted_lfc and observed_lfc must have the same shape")
    if observed.size == 0:
        return 0.0
    actual_k = min(k, observed.size)
    pred_top = set(np.argsort(-np.abs(predicted))[:actual_k])
    obs_top = set(np.argsort(-np.abs(observed))[:actual_k])
    return len(pred_top & obs_top) / actual_k


def de_direction_accuracy(
    predicted_lfc: np.ndarray,
    observed_lfc: np.ndarray,
    *,
    k: int | None = None,
    min_abs_lfc: float = 0.0,
) -> float:
    """Share of DE genes with matching up/down direction."""

    predicted = np.asarray(predicted_lfc, dtype=float).ravel()
    observed = np.asarray(observed_lfc, dtype=float).ravel()
    if predicted.shape != observed.shape:
        raise ValueError("predicted_lfc and observed_lfc must have the same shape")
    if observed.size == 0:
        return 0.0
    if k is None:
        mask = np.abs(observed) > min_abs_lfc
    else:
        if k <= 0:
            raise ValueError("k must be positive")
        mask = np.zeros(observed.size, dtype=bool)
        mask[np.argsort(-np.abs(observed))[: min(k, observed.size)]] = True
        if min_abs_lfc > 0:
            mask &= np.abs(observed) > min_abs_lfc
    if not mask.any():
        return 0.0
    return float(np.mean(np.sign(predicted[mask]) == np.sign(observed[mask])))


def differential_expression_table(
    observed_lfc: np.ndarray,
    *,
    predicted_lfc: np.ndarray | None = None,
    gene_names: Sequence[str] | None = None,
    k: int = 50,
) -> pd.DataFrame:
    observed = np.asarray(observed_lfc, dtype=float).ravel()
    predicted = None if predicted_lfc is None else np.asarray(predicted_lfc, dtype=float).ravel()
    if predicted is not None and predicted.shape != observed.shape:
        raise ValueError("predicted_lfc and observed_lfc must have the same shape")
    genes = _gene_names(observed.size, gene_names)
    top_observed = set(np.argsort(-np.abs(observed))[: min(k, observed.size)])
    records = []
    for idx, gene in enumerate(genes):
        row: dict[str, object] = {
            "gene": gene,
            "observed_log_fold_change": observed[idx],
            "observed_abs_rank": int(np.where(np.argsort(-np.abs(observed)) == idx)[0][0] + 1),
            f"in_observed_top{k}": idx in top_observed,
        }
        if predicted is not None:
            top_predicted = set(np.argsort(-np.abs(predicted))[: min(k, predicted.size)])
            row.update(
                {
                    "predicted_log_fold_change": predicted[idx],
                    "predicted_abs_rank": int(np.where(np.argsort(-np.abs(predicted)) == idx)[0][0] + 1),
                    f"in_predicted_top{k}": idx in top_predicted,
                    "direction_match": bool(np.sign(predicted[idx]) == np.sign(observed[idx])),
                }
            )
        records.append(row)
    return pd.DataFrame.from_records(records).sort_values("observed_abs_rank").reset_index(drop=True)


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    """Read a GMT-like pathway file: name, optional description, then genes."""

    gene_sets: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = [field.strip() for field in line.rstrip("\n").split("\t") if field.strip()]
            if len(fields) < 3:
                continue
            name = fields[0]
            genes = fields[2:]
            if not name or not genes:
                raise ValueError(f"invalid GMT row at line {line_number}")
            gene_sets[name] = genes
    return gene_sets


def pathway_score_table(
    lfc: np.ndarray,
    gene_names: Sequence[str],
    gene_sets: Mapping[str, Sequence[str]],
    *,
    predicted_lfc: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score pathways by mean signed log fold-change across member genes."""

    observed = np.asarray(lfc, dtype=float).ravel()
    predicted = None if predicted_lfc is None else np.asarray(predicted_lfc, dtype=float).ravel()
    genes = list(gene_names)
    if len(genes) != observed.size:
        raise ValueError("gene_names length must match lfc")
    if predicted is not None and predicted.shape != observed.shape:
        raise ValueError("predicted_lfc and lfc must have the same shape")
    index = {gene: idx for idx, gene in enumerate(genes)}
    records = []
    for pathway, members in sorted(gene_sets.items()):
        hits = [index[gene] for gene in members if gene in index]
        if not hits:
            records.append(
                {
                    "pathway": pathway,
                    "n_genes": 0,
                    "observed_score": np.nan,
                    "predicted_score": np.nan if predicted is not None else NOT_AVAILABLE,
                    "status": "not available: no pathway genes found in expression features",
                }
            )
            continue
        row: dict[str, object] = {
            "pathway": pathway,
            "n_genes": len(hits),
            "observed_score": float(np.mean(observed[hits])),
            "status": "available",
        }
        if predicted is not None:
            row["predicted_score"] = float(np.mean(predicted[hits]))
        records.append(row)
    return pd.DataFrame.from_records(records)


def moa_enrichment_table(
    de_table: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    moa_col: str = "moa",
    perturbation_col: str = "perturbation",
    k: int = 50,
) -> pd.DataFrame:
    """Summarize top observed DE genes for each MoA when metadata supports it."""

    if moa_col not in metadata.columns:
        return _not_available_frame("moa_enrichment", f"not available: metadata is missing {moa_col!r}")
    if perturbation_col not in metadata.columns:
        return _not_available_frame("moa_enrichment", f"not available: metadata is missing {perturbation_col!r}")
    observed_col = f"in_observed_top{k}"
    if observed_col not in de_table.columns:
        return _not_available_frame("moa_enrichment", "not available: differential expression table is missing top-k calls")
    top_genes = de_table.loc[de_table[observed_col], "gene"].astype(str).tolist()
    records = []
    for moa, group in metadata[[moa_col, perturbation_col]].drop_duplicates().groupby(moa_col, dropna=False):
        records.append(
            {
                moa_col: _clean_value(moa),
                "n_perturbations": int(group[perturbation_col].nunique()),
                "n_top_de_genes": len(top_genes),
                "top_de_genes": ";".join(top_genes),
                "status": "available",
            }
        )
    return pd.DataFrame.from_records(records)


def dose_time_summary(
    lfc_by_condition: pd.DataFrame,
    *,
    dose_col: str = "dose",
    time_col: str = "time",
    perturbation_col: str = "perturbation",
) -> pd.DataFrame:
    """Summarize response magnitude by perturbation, dose, and time if present."""

    missing = [column for column in (perturbation_col, dose_col, time_col) if column not in lfc_by_condition.columns]
    if missing:
        return _not_available_frame("dose_time", f"not available: metadata is missing {missing}")
    value_cols = [column for column in lfc_by_condition.columns if column.startswith("lfc__")]
    if not value_cols:
        return _not_available_frame("dose_time", "not available: no log fold-change columns were computed")
    frame = lfc_by_condition.copy()
    frame["response_l2"] = np.sqrt(np.square(frame[value_cols].to_numpy(dtype=float)).sum(axis=1))
    return (
        frame.groupby([perturbation_col, dose_col, time_col], sort=True, dropna=False)["response_l2"]
        .mean()
        .reset_index()
        .assign(status="available")
    )


def run_biology_validation(
    expression: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    predicted_expression: np.ndarray | None = None,
    gene_names: Sequence[str] | None = None,
    groupby: Sequence[str] | None = None,
    condition_col: str = "perturbation",
    control_values: Sequence[str] = ("control", "DMSO", "dmso", "vehicle", "untreated"),
    control_type_col: str = "perturbation_type",
    topk: int = 50,
    gmt_path: str | Path | None = None,
) -> ValidationResult:
    """Compute biology validation tables from cell-level expression and metadata."""

    observed = _as_2d(expression, name="expression")
    predicted = None if predicted_expression is None else _as_2d(predicted_expression, name="predicted_expression")
    if predicted is not None and predicted.shape != observed.shape:
        raise ValueError("predicted_expression must have the same shape as expression")
    frame = _metadata_frame(metadata, n_rows=observed.shape[0])
    genes = _gene_names(observed.shape[1], gene_names)
    group_cols = list(groupby or _default_groupby(frame, condition_col=condition_col))
    messages: list[str] = []

    bulk = pseudobulk_aggregate(observed, frame, groupby=group_cols, gene_names=genes)
    pred_bulk = None
    if predicted is not None:
        pred_bulk = pseudobulk_aggregate(predicted, frame, groupby=group_cols, gene_names=genes)

    control_mask = _control_mask(frame, condition_col=condition_col, control_type_col=control_type_col, control_values=control_values)
    if not control_mask.any():
        message = "not available: no control rows found; cannot compute log fold-change or DE metrics"
        return ValidationResult(
            metrics=_not_available_frame("metrics", message),
            de_table=_not_available_frame("differential_expression", message),
            pathway_scores=_not_available_frame("pathway_scores", "not available: log fold-change is missing"),
            moa_enrichment=_not_available_frame("moa_enrichment", message),
            dose_time=_not_available_frame("dose_time", message),
            messages=[message],
        )

    control_mean = observed[control_mask].mean(axis=0)
    pred_control_mean = predicted[control_mask].mean(axis=0) if predicted is not None else None
    condition_rows = []
    metrics = []
    de_tables = []
    pathway_tables = []
    gene_sets = read_gmt(gmt_path) if gmt_path is not None else None
    if gene_sets is None:
        messages.append("not available: pathway scoring skipped because no GMT file was provided")

    gene_cols = genes
    for _, row in bulk.iterrows():
        is_control = _row_is_control(row, condition_col=condition_col, control_values=control_values, control_type_col=control_type_col)
        if is_control:
            continue
        row_values = row[gene_cols].to_numpy(dtype=float)
        obs_lfc = log_fold_change(row_values, control_mean)
        pred_lfc = None
        if pred_bulk is not None and pred_control_mean is not None:
            match = pred_bulk
            for column in group_cols:
                match = match[match[column].astype(str) == str(row[column])]
            if not match.empty:
                pred_lfc = log_fold_change(match.iloc[0][gene_cols].to_numpy(dtype=float), pred_control_mean)

        condition_id = "|".join(str(row[column]) for column in group_cols)
        metric_row: dict[str, object] = {
            "condition": condition_id,
            "n_cells": int(row["n_cells"]),
            "topk": topk,
            "status": "available",
        }
        if pred_lfc is None:
            metric_row["topk_de_overlap"] = NOT_AVAILABLE
            metric_row["de_direction_accuracy"] = NOT_AVAILABLE
            metric_row["prediction_status"] = "not available: predicted expression was not provided"
        else:
            metric_row["topk_de_overlap"] = topk_de_overlap(pred_lfc, obs_lfc, k=topk)
            metric_row["de_direction_accuracy"] = de_direction_accuracy(pred_lfc, obs_lfc, k=topk)
            metric_row["prediction_status"] = "available"
        for column in group_cols:
            metric_row[column] = row[column]
        metrics.append(metric_row)

        de = differential_expression_table(obs_lfc, predicted_lfc=pred_lfc, gene_names=genes, k=topk)
        de.insert(0, "condition", condition_id)
        de_tables.append(de)

        lfc_record = {column: row[column] for column in group_cols}
        lfc_record["condition"] = condition_id
        for gene, value in zip(genes, obs_lfc, strict=True):
            lfc_record[f"lfc__{gene}"] = value
        condition_rows.append(lfc_record)

        if gene_sets is not None:
            pathway = pathway_score_table(obs_lfc, genes, gene_sets, predicted_lfc=pred_lfc)
            pathway.insert(0, "condition", condition_id)
            pathway_tables.append(pathway)

    if not metrics:
        message = "not available: no non-control pseudobulk conditions found"
        metrics_frame = _not_available_frame("metrics", message)
        de_frame = _not_available_frame("differential_expression", message)
    else:
        metrics_frame = pd.DataFrame.from_records(metrics)
        de_frame = pd.concat(de_tables, ignore_index=True)

    lfc_frame = pd.DataFrame.from_records(condition_rows)
    pathway_frame = (
        pd.concat(pathway_tables, ignore_index=True)
        if pathway_tables
        else _not_available_frame("pathway_scores", "not available: no GMT file was provided")
    )
    moa_frame = moa_enrichment_table(de_frame, frame, k=topk) if not de_frame.empty else _not_available_frame("moa_enrichment", "not available: DE table is empty")
    dose_frame = dose_time_summary(lfc_frame)
    return ValidationResult(metrics_frame, de_frame, pathway_frame, moa_frame, dose_frame, messages)


def write_validation_outputs(result: ValidationResult, output_dir: str | Path) -> dict[str, str]:
    """Write CSV, JSON, PNG, and Markdown validation outputs."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    paths = {
        "metrics_csv": out / "metrics.csv",
        "de_csv": out / "differential_expression.csv",
        "pathway_csv": out / "pathway_scores.csv",
        "moa_csv": out / "moa_enrichment.csv",
        "dose_time_csv": out / "dose_time_summary.csv",
        "json": out / "validation_summary.json",
        "interpretation": out / "interpretation.md",
        "figure": figures / "topk_de_overlap.png",
    }
    result.metrics.to_csv(paths["metrics_csv"], index=False)
    result.de_table.to_csv(paths["de_csv"], index=False)
    result.pathway_scores.to_csv(paths["pathway_csv"], index=False)
    result.moa_enrichment.to_csv(paths["moa_csv"], index=False)
    result.dose_time.to_csv(paths["dose_time_csv"], index=False)

    summary = _summary_payload(result)
    paths["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["interpretation"].write_text(_interpretation_markdown(result, summary), encoding="utf-8")
    _write_metric_png(result.metrics, paths["figure"], metric="topk_de_overlap")
    return {key: str(value) for key, value in paths.items()}


def _summary_payload(result: ValidationResult) -> dict[str, object]:
    numeric_metrics: dict[str, float] = {}
    for column in result.metrics.columns:
        values = pd.to_numeric(result.metrics[column], errors="coerce")
        if values.notna().any():
            numeric_metrics[f"mean_{column}"] = float(values.mean())
    messages = list(result.messages)
    for table_name, table in (
        ("metrics", result.metrics),
        ("pathway_scores", result.pathway_scores),
        ("moa_enrichment", result.moa_enrichment),
        ("dose_time", result.dose_time),
    ):
        if "status" in table.columns:
            for status in table["status"].dropna().astype(str).unique():
                if status.startswith("not available"):
                    messages.append(f"{table_name}: {status}")
    return {"metrics": numeric_metrics, "messages": sorted(set(messages))}


def _interpretation_markdown(result: ValidationResult, summary: Mapping[str, object]) -> str:
    lines = ["# Biological Validation", ""]
    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping) and metrics:
        lines.append("## Metric summary")
        for name, value in sorted(metrics.items()):
            lines.append(f"- {name}: {value:.4g}")
        lines.append("")
    messages = summary.get("messages", [])
    if messages:
        lines.append("## Availability")
        for message in messages:
            lines.append(f"- {message}")
        lines.append("")
    lines.append("## Outputs")
    lines.extend(
        [
            "- metrics.csv: per-condition DE recovery metrics.",
            "- differential_expression.csv: gene-level observed and optional predicted log fold-change.",
            "- pathway_scores.csv: optional GMT pathway scores.",
            "- moa_enrichment.csv: MoA helper summary when metadata contains MoA labels.",
            "- dose_time_summary.csv: dose/time response helper when metadata contains dose and time.",
            "- figures/topk_de_overlap.png: compact top-k DE overlap chart.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_metric_png(metrics: pd.DataFrame, path: Path, *, metric: str) -> None:
    width, height = 720, 360
    image = bytearray([255, 255, 255] * width * height)
    values = pd.to_numeric(metrics.get(metric, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    values = values[:20]
    if values.size:
        max_value = max(float(np.nanmax(values)), 1e-12)
        bar_width = max(8, (width - 120) // max(values.size, 1))
        for idx, value in enumerate(values):
            x0 = 60 + idx * bar_width
            x1 = min(x0 + bar_width - 4, width - 40)
            bar_height = int((height - 100) * max(value, 0.0) / max_value)
            y0 = height - 50 - bar_height
            _fill_rect(image, width, height, x0, y0, x1, height - 50, (47, 111, 157))
    else:
        _fill_rect(image, width, height, 60, height - 56, width - 40, height - 50, (180, 180, 180))
    _fill_rect(image, width, height, 50, 40, 54, height - 50, (30, 30, 30))
    _fill_rect(image, width, height, 50, height - 54, width - 40, height - 50, (30, 30, 30))
    path.write_bytes(_png_bytes(width, height, bytes(image)))


def _png_bytes(width: int, height: int, rgb: bytes) -> bytes:
    rows = [b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height)]
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _fill_rect(image: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        for x in range(x0, x1):
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)


def _as_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _metadata_frame(metadata: pd.DataFrame | Mapping[str, Sequence[object]], *, n_rows: int) -> pd.DataFrame:
    frame = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if len(frame) != n_rows:
        raise ValueError("metadata rows must match expression rows")
    return frame.reset_index(drop=True)


def _columns(columns: str | Sequence[str]) -> list[str]:
    if isinstance(columns, str):
        return [column.strip() for column in columns.split(",") if column.strip()]
    return [str(column) for column in columns]


def _gene_names(n_genes: int, gene_names: Sequence[str] | None) -> list[str]:
    if gene_names is None:
        return [f"gene_{idx}" for idx in range(n_genes)]
    genes = [str(gene) for gene in gene_names]
    if len(genes) != n_genes:
        raise ValueError("gene_names length must match expression feature count")
    return genes


def _default_groupby(frame: pd.DataFrame, *, condition_col: str) -> list[str]:
    preferred = [condition_col, "dose", "time", "cell_line"]
    columns = [column for column in preferred if column in frame.columns]
    if not columns:
        raise ValueError(f"metadata must contain {condition_col!r} or an explicit groupby must be provided")
    return columns


def _control_mask(
    frame: pd.DataFrame,
    *,
    condition_col: str,
    control_type_col: str,
    control_values: Sequence[str],
) -> np.ndarray:
    mask = np.zeros(len(frame), dtype=bool)
    controls = {str(value).lower() for value in control_values}
    if condition_col in frame.columns:
        mask |= frame[condition_col].astype(str).str.lower().isin(controls).to_numpy()
    if control_type_col in frame.columns:
        mask |= frame[control_type_col].astype(str).str.lower().eq("control").to_numpy()
    return mask


def _row_is_control(
    row: pd.Series,
    *,
    condition_col: str,
    control_values: Sequence[str],
    control_type_col: str,
) -> bool:
    controls = {str(value).lower() for value in control_values}
    if condition_col in row.index and str(row[condition_col]).lower() in controls:
        return True
    return control_type_col in row.index and str(row[control_type_col]).lower() == "control"


def _not_available_frame(kind: str, message: str) -> pd.DataFrame:
    return pd.DataFrame([{"table": kind, "status": message}])


def _clean_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)
