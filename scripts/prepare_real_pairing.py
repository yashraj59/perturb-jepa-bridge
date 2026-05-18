from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.images import filter_label_free_channels
from perturb_jepa.data.schema import add_hierarchical_condition_keys, normalize_image_manifest, normalize_scrna_obs


SYNONYM_TO_CANONICAL = {
    "saha": "vorinostat",
    "vorinostat": "vorinostat",
    "ms-275": "entinostat",
    "ms275": "entinostat",
    "entinostat": "entinostat",
    "tsa": "trichostatin a",
    "trichostatin a": "trichostatin a",
    "jnj-26481585": "quisinostat",
    "jnj26481585": "quisinostat",
    "quisinostat": "quisinostat",
    "lbh-589": "panobinostat",
    "lbh589": "panobinostat",
    "panobinostat": "panobinostat",
    "pci-24781": "abexinostat",
    "pci24781": "abexinostat",
    "abexinostat": "abexinostat",
    "fk228": "romidepsin",
    "romidepsin": "romidepsin",
    "ci-994": "tacedinaline",
    "ci994": "tacedinaline",
    "tacedinaline": "tacedinaline",
    "splitomicin": "splitomicin",
    "sirtinol": "sirtinol",
    "nicotinamide": "nicotinamide",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare aligned sci-Plex3/BF-MoA condition-level bridge inputs.")
    parser.add_argument("--rna-h5ad", type=Path, default=Path("data/raw/SrivatsanTrapnell2020_sciplex3.h5ad"))
    parser.add_argument("--bf-manifest-raw", type=Path, default=Path("data/processed/bf_moa_manifest_raw.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/bf_moa_images"))
    parser.add_argument("--cell-line", default="A549")
    parser.add_argument("--target-dose-um", type=float, default=10.0)
    parser.add_argument("--target-image-time-h", type=float, default=48.0)
    parser.add_argument("--preferred-rna-time-h", type=float, default=72.0)
    parser.add_argument("--min-shared-compounds", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rna-output", type=Path, default=Path("data/processed/sciplex3_a549_10uM_72h.h5ad"))
    parser.add_argument("--image-output", type=Path, default=Path("data/processed/bf_moa_manifest_aligned.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("metrics/compound_intersection.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("metrics/prepare_real_pairing_summary.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.rna_h5ad.exists():
        raise SystemExit(f"missing RNA h5ad: {args.rna_h5ad}")
    if not args.bf_manifest_raw.exists():
        raise SystemExit(f"missing BF-MoA manifest: {args.bf_manifest_raw}")
    if not args.image_root.exists():
        raise SystemExit(f"missing BF-MoA image root: {args.image_root}")

    import anndata as ad

    adata = ad.read_h5ad(args.rna_h5ad)
    print(f"sci-Plex3 AnnData: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print("sci-Plex3 obs columns:", ", ".join(map(str, adata.obs.columns)))
    rna_obs = _standardize_sciplex3_obs(pd.DataFrame(adata.obs).copy())
    rna_obs["_matrix_index"] = np.arange(adata.n_obs)

    image_long = normalize_image_manifest(pd.read_csv(args.bf_manifest_raw))
    image_bf = filter_label_free_channels(
        image_long,
        markers=("c1", "c2", "c3", "c4", "c5", "c6", "bf", "brightfield", "phase"),
    )
    if image_bf.empty:
        raise SystemExit("BF-MoA brightfield filter removed all image rows")
    if set(image_bf["dose"].astype(str)) != {"10uM"} or set(image_bf["time"].astype(str)) != {"48h"}:
        raise SystemExit("BF-MoA manifest does not have the expected fixed dose/time of 10uM and 48h")

    rna_obs, image_bf, synonym_summary = _add_compound_keys(rna_obs, image_bf)
    print("Synonyms applied:")
    if synonym_summary:
        for label, count in sorted(synonym_summary.items()):
            print(f"- {label}: {count}")
    else:
        print("- none")

    image_bf = _restrict_to_available_plate_images(image_bf, args.image_root)
    if image_bf.empty:
        raise SystemExit("No BF-MoA rows from the raw manifest resolve to downloaded image files")

    selected_rna, selected_image, selection = _select_overlap(
        rna_obs,
        image_bf,
        cell_line=args.cell_line,
        target_dose_um=args.target_dose_um,
        preferred_time_h=args.preferred_rna_time_h,
        image_target_time_h=args.target_image_time_h,
        min_shared=args.min_shared_compounds,
    )
    shared = sorted(set(selected_rna["compound_key"]) & set(selected_image["compound_key"]))
    if len(shared) < args.min_shared_compounds:
        rna_compounds = sorted(selected_rna["compound_key"].unique().tolist())
        image_compounds = sorted(selected_image["compound_key"].unique().tolist())
        raise SystemExit(
            f"Only {len(shared)} shared compounds remain after dose/time selection; "
            f"RNA compounds={rna_compounds}; image compounds={image_compounds}"
        )

    selected_rna = selected_rna.loc[selected_rna["compound_key"].isin(shared)].copy()
    selected_image = selected_image.loc[selected_image["compound_key"].isin(shared)].copy()
    aligned_image = _write_six_channel_images(selected_image, args.image_root, args.image_size)

    adata_subset = adata[selected_rna["_matrix_index"].to_numpy(), :].copy()
    obs_for_h5ad = selected_rna.drop(columns=["_matrix_index"]).copy()
    adata_subset.obs = obs_for_h5ad
    adata_subset.write_h5ad(args.rna_output)

    args.image_output.parent.mkdir(parents=True, exist_ok=True)
    aligned_image.to_csv(args.image_output, index=False)
    summary = _summary_table(selected_rna, aligned_image)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    payload = {
        "rna_input": str(args.rna_h5ad),
        "image_manifest_input": str(args.bf_manifest_raw),
        "image_root": str(args.image_root),
        "rna_output": str(args.rna_output),
        "image_output": str(args.image_output),
        "summary_output": str(args.summary_output),
        "cell_line": args.cell_line,
        "rna_dose": selection["dose"],
        "rna_time": selection["time"],
        "image_dose": "10uM",
        "image_time": "48h",
        "time_mismatch_limitation": f"sci-Plex3 {selection['time']} vs BF-MoA 48h",
        "n_shared_compounds": len(shared),
        "shared_compounds": shared,
        "synonyms_applied": dict(sorted(synonym_summary.items())),
        "n_rna_cells": int(len(selected_rna)),
        "n_image_sites": int(len(aligned_image)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote RNA subset: {args.rna_output}")
    print(f"Wrote aligned image manifest: {args.image_output}")
    print(f"Wrote summary: {args.summary_output}")
    return 0


def _standardize_sciplex3_obs(obs: pd.DataFrame) -> pd.DataFrame:
    frame = obs.copy()
    perturbation_col = _first_existing(
        frame,
        ("perturbation", "product_name", "compound", "drug", "drug_name", "perturbation_name"),
    )
    dose_col = _first_existing(frame, ("dose", "dose_value", "dose_nm", "dose_nM", "dose_um", "dose_uM"))
    time_col = _first_existing(frame, ("time", "timepoint", "time_point", "time_hr", "time_h"))
    cell_line_col = _first_existing(frame, ("cell_line", "cell_type", "cell", "cell_name", "celltype"))
    batch_col = _first_existing(frame, ("batch", "plate", "replicate", "sample", "sample_id", "gem_group"))
    moa_col = _first_existing(frame, ("moa", "pathway_level_1", "target", "pathway", "pathway_level_2"), required=False)

    standardized = frame.copy()
    standardized["perturbation"] = frame[perturbation_col]
    standardized["perturbation_type"] = "compound"
    standardized["dose"] = [_format_dose_um(value) for value in frame[dose_col]]
    standardized["time"] = [_format_time_h(value) for value in frame[time_col]]
    standardized["cell_line"] = frame[cell_line_col]
    standardized["batch"] = frame[batch_col] if batch_col is not None else "unknown"
    if moa_col is not None:
        standardized["moa"] = frame[moa_col].astype(str)
    else:
        standardized["moa"] = "unknown"
    standardized = normalize_scrna_obs(standardized)
    standardized["source_cell_line"] = standardized["cell_line"]
    return standardized


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if required:
        raise SystemExit(f"none of the expected columns were found: {tuple(candidates)}")
    return None


def _format_dose_um(value: object) -> str:
    text = str(value).strip()
    number = _number_from_value(text)
    if number is None:
        return text
    lower = text.lower()
    if "nm" in lower or ("um" not in lower and number > 100):
        number = number / 1000.0
    return f"{number:g}uM"


def _format_time_h(value: object) -> str:
    number = _number_from_value(value)
    if number is None:
        return str(value).strip()
    return f"{number:g}h"


def _number_from_value(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if match is None:
        return None
    return float(match.group(0))


def _normalize_compound(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    compact = text.replace(" ", "")
    candidates = [text, compact]
    before_parenthesis = text.split("(", 1)[0].strip()
    if before_parenthesis:
        candidates.extend((before_parenthesis, before_parenthesis.replace(" ", "")))
    candidates.extend(piece.strip() for piece in re.findall(r"\(([^)]*)\)", text))
    for parenthetical in re.findall(r"\(([^)]*)\)", text):
        candidates.extend(piece.strip() for piece in re.split(r"[,;/]", parenthetical))
    for candidate in candidates:
        compact_candidate = candidate.replace(" ", "")
        if candidate in SYNONYM_TO_CANONICAL:
            return SYNONYM_TO_CANONICAL[candidate]
        if compact_candidate in SYNONYM_TO_CANONICAL:
            return SYNONYM_TO_CANONICAL[compact_candidate]
    return before_parenthesis or text


def _add_compound_keys(rna: pd.DataFrame, image: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str]]:
    counts: Counter[str] = Counter()

    def apply(values: pd.Series, prefix: str) -> pd.Series:
        normalized = []
        for value in values:
            raw = re.sub(r"\s+", " ", str(value).strip().lower())
            key = _normalize_compound(value)
            if key != raw:
                counts[f"{prefix}: {raw} -> {key}"] += 1
            normalized.append(key)
        return pd.Series(normalized, index=values.index)

    rna = rna.copy()
    image = image.copy()
    rna["compound_key"] = apply(rna["perturbation"], "rna")
    image["compound_key"] = apply(image["compound"], "image")
    rna["perturbation"] = rna["compound_key"]
    image["perturbation"] = image["compound_key"]
    rna = add_hierarchical_condition_keys(rna)
    image = add_hierarchical_condition_keys(image)
    return rna, image, counts


def _restrict_to_available_plate_images(image: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    paths_by_name = {path.name: path.resolve() for path in image_root.rglob("*") if path.is_file()}
    if not paths_by_name:
        raise SystemExit(f"no image files found under {image_root}")
    image = image.copy()
    image["resolved_image_path"] = image["image_path"].map(lambda value: _resolve_image_path(value, paths_by_name))
    return image.loc[image["resolved_image_path"].notna()].copy()


def _resolve_image_path(value: object, paths_by_name: dict[str, Path]) -> str | None:
    path = Path(str(value))
    if path.exists():
        return str(path.resolve())
    return str(paths_by_name[path.name]) if path.name in paths_by_name else None


def _select_overlap(
    rna: pd.DataFrame,
    image: pd.DataFrame,
    *,
    cell_line: str,
    target_dose_um: float,
    preferred_time_h: float,
    image_target_time_h: float,
    min_shared: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    image_compounds = set(image["compound_key"])
    rna_cell = rna.loc[rna["cell_line"].astype(str).str.lower() == cell_line.lower()].copy()
    if rna_cell.empty:
        raise SystemExit(f"No sci-Plex3 rows found for cell line {cell_line!r}")
    dose_values = sorted(
        rna_cell["dose"].dropna().unique().tolist(),
        key=lambda value: abs((_number_from_value(value) or 0.0) - target_dose_um),
    )
    time_values = sorted(
        rna_cell["time"].dropna().unique().tolist(),
        key=lambda value: abs((_number_from_value(value) or 0.0) - image_target_time_h),
    )
    preferred_time = f"{preferred_time_h:g}h"
    if preferred_time in set(time_values):
        time_values = [preferred_time] + [value for value in time_values if value != preferred_time]
    target_dose = f"{target_dose_um:g}uM"
    if target_dose in set(dose_values):
        dose_values = [target_dose] + [value for value in dose_values if value != target_dose]

    best: tuple[pd.DataFrame, str, str, int] | None = None
    for dose_count in range(1, min(3, len(dose_values)) + 1):
        for time_count in range(1, min(3, len(time_values)) + 1):
            selected_doses = set(dose_values[:dose_count])
            selected_times = set(time_values[:time_count])
            subset = rna_cell.loc[rna_cell["dose"].isin(selected_doses) & rna_cell["time"].isin(selected_times)].copy()
            shared = len(set(subset["compound_key"]) & image_compounds)
            if best is None or shared > best[3]:
                best = (subset, ",".join(sorted(selected_doses)), ",".join(sorted(selected_times)), shared)
            if shared >= min_shared:
                return subset, image.copy(), {"dose": ",".join(sorted(selected_doses)), "time": ",".join(sorted(selected_times))}
    assert best is not None
    return best[0], image.copy(), {"dose": best[1], "time": best[2]}


def _write_six_channel_images(image: pd.DataFrame, image_root: Path, image_size: int) -> pd.DataFrame:
    from PIL import Image

    out_dir = Path("data/processed/bf_moa_6ch_arrays")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    grouped = image.groupby(["plate", "well", "site", "compound_key"], sort=True)
    for (plate, well, site, compound), group in grouped:
        channels = []
        for channel in [f"C{i}" for i in range(1, 7)]:
            channel_group = group.loc[group["channel_or_z"].astype(str).str.upper() == channel]
            if channel_group.empty:
                break
            path = Path(str(channel_group.iloc[0]["resolved_image_path"]))
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L").resize((image_size, image_size)), dtype=np.float32)
            if arr.max() > 1:
                arr = arr / 255.0
            channels.append(arr)
        if len(channels) != 6:
            continue
        array = np.stack(channels, axis=0).astype(np.float32, copy=False)
        file_name = f"{plate}_{well}_{site}_{compound}.npy".replace("/", "_").replace(" ", "_")
        path = out_dir / file_name
        np.save(path, array)
        first = group.iloc[0].copy()
        first["image_path"] = str(path.resolve())
        first["channel_or_z"] = "BF_C1-C6"
        first["perturbation"] = str(compound)
        first["compound"] = str(compound)
        first["condition_key_medium"] = f"{compound}|10uM|48h"
        rows.append(first)
    if not rows:
        raise SystemExit("No complete six-channel BF-MoA sites could be assembled")
    manifest = pd.DataFrame(rows).drop(columns=["resolved_image_path"], errors="ignore")
    manifest = add_hierarchical_condition_keys(manifest)
    manifest["condition_key"] = manifest["condition_key_medium"]
    return manifest.reset_index(drop=True)


def _summary_table(rna: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    rna_counts = rna.groupby("compound_key").size().rename("n_cells")
    image_counts = image.groupby("compound").size().rename("n_images")
    moa = image.groupby("compound")["moa"].agg(lambda values: ";".join(sorted(set(map(str, values))))).rename("moa")
    summary = pd.concat([rna_counts, image_counts, moa], axis=1).fillna({"n_cells": 0, "n_images": 0, "moa": ""})
    summary.index.name = "compound"
    return summary.reset_index().sort_values("compound")


if __name__ == "__main__":
    raise SystemExit(main())
