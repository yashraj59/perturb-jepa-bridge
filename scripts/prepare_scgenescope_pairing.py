from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.schema import normalize_image_manifest, normalize_scrna_obs, normalize_value


HF_REPO_ID = "altoslabs/scGeneScope"
OFFICIAL_TREATMENTS = (
    "Phenacetin",
    "PQ401",
    "Splitomicin",
    "(R)-MG132",
    "(R)-Roscovitine",
    "Wy 14643 / Pirinixic Acid",
    "Fluocinonide",
    "Caffeine",
    "LY303511 (hydrochloride)",
    "Simvastatin",
    "Colchicine",
    "Pantoprazole",
    "Cycloheximide",
    "Benzbromarone",
    "Thapsigargin",
    "BAY 11-7082",
    "CGK-733",
    "PD-98059",
    "GW-843682X",
    "12-O-tetradecanoylphorbol-13-acetate",
    "SKII",
    "AMG-900",
    "DBeQ",
    "Daporinad / FK-866",
    "Vorinostat / SAHA",
    "Quinidine",
    "Aloxistatin / E-64d",
    "HARMAN",
)
CONTROL_TOKENS = {"dmso", "control", "solvent control", "vehicle"}
OFFICIAL_SPLITS = {
    ("1", "3"): "train",
    ("1", "5"): "val",
    ("1", "4"): "we_test",
    ("2", "1"): "he_test",
    ("2", "2"): "he_test",
}
SPLIT_POLICIES = {
    "official": OFFICIAL_SPLITS,
    "round1_train_val": {
        ("1", "3"): "train",
        ("1", "5"): "train",
        ("1", "4"): "we_test",
        ("2", "1"): "he_test",
        ("2", "2"): "he_test",
    },
    "round2_anchor": {
        ("1", "3"): "train",
        ("1", "5"): "train",
        ("2", "1"): "train",
        ("1", "4"): "we_test",
        ("2", "2"): "he_test",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare scGeneScope precomputed embedding H5ADs for condition-bag bridge training. "
            "The output treats Cell Painting embeddings as small feature-grid .npy arrays so the "
            "existing image-manifest training path can be reused."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/scgenescope"))
    parser.add_argument(
        "--rna-round1",
        type=Path,
        default=Path("data/raw/scgenescope/features/rnaseq/scvi/n200/round_1.h5ad"),
    )
    parser.add_argument(
        "--rna-round2",
        type=Path,
        default=Path("data/raw/scgenescope/features/rnaseq/scvi/n200/round_2.h5ad"),
    )
    parser.add_argument(
        "--image-round1",
        type=Path,
        default=Path("data/raw/scgenescope/features/imaging/imagenet/vit-l/round_1.h5ad"),
    )
    parser.add_argument(
        "--image-round2",
        type=Path,
        default=Path("data/raw/scgenescope/features/imaging/imagenet/vit-l/round_2.h5ad"),
    )
    parser.add_argument("--download", action="store_true", help="Download missing embedding H5ADs from Hugging Face.")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token. Defaults to HF_TOKEN env var.")
    parser.add_argument("--rna-obsm-key", default="scvi_n200")
    parser.add_argument(
        "--image-obsm-key",
        default=None,
        help="Optional image embedding key in .obsm. If omitted, X is used unless the key exists in obsm.",
    )
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument(
        "--max-profiles-per-condition-split",
        type=int,
        default=512,
        help="Deterministic cap per treatment per split and modality. Use <=0 to keep every profile.",
    )
    parser.add_argument(
        "--split-policy",
        choices=sorted(SPLIT_POLICIES),
        default="official",
        help=(
            "official preserves scGeneScope's original split labels. round1_train_val trains on "
            "round 1 replicates 3/5 and keeps both round 2 replicates for HE. round2_anchor trains "
            "on round 1 replicates 3/5 plus round 2 replicate 1, then evaluates WE on round 1 "
            "replicate 4 and HE on held-out round 2 replicate 2."
        ),
    )
    parser.add_argument(
        "--control-center",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Subtract per-batch control mean and add global control mean before writing outputs.",
    )
    parser.add_argument("--include-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-treatments-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-shared-treatments", type=int, default=20)
    parser.add_argument("--rna-output", type=Path, default=Path("data/processed/scgenescope/rna_scvi_n200.h5ad"))
    parser.add_argument(
        "--image-output",
        type=Path,
        default=Path("data/processed/scgenescope/image_vitl_manifest.csv"),
    )
    parser.add_argument(
        "--image-array-dir",
        type=Path,
        default=Path("data/processed/scgenescope/image_vitl_arrays"),
    )
    parser.add_argument("--summary-output", type=Path, default=Path("metrics/scgenescope/condition_overlap.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("metrics/scgenescope/prepare_summary.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.download:
        _download_defaults(args)
    for path, label in (
        (args.rna_round1, "RNA round 1"),
        (args.rna_round2, "RNA round 2"),
        (args.image_round1, "image round 1"),
        (args.image_round2, "image round 2"),
    ):
        _require(path, label)

    rna_values, rna_obs = _load_rounds(
        [(args.rna_round1, 1), (args.rna_round2, 2)],
        modality="rna",
        data_key=args.rna_obsm_key,
    )
    image_values, image_obs = _load_rounds(
        [(args.image_round1, 1), (args.image_round2, 2)],
        modality="image",
        data_key=args.image_obsm_key,
    )
    rna_obs = _standardize_obs(
        rna_obs,
        include_controls=args.include_controls,
        official_treatments_only=args.official_treatments_only,
        split_policy=args.split_policy,
    )
    rna_values = rna_values[rna_obs["_value_index"].astype(int).to_numpy()]
    rna_obs = rna_obs.drop(columns=["_value_index"]).reset_index(drop=True)
    image_obs = _standardize_obs(
        image_obs,
        include_controls=args.include_controls,
        official_treatments_only=args.official_treatments_only,
        split_policy=args.split_policy,
    )
    image_values = image_values[image_obs["_value_index"].astype(int).to_numpy()]
    image_obs = image_obs.drop(columns=["_value_index"]).reset_index(drop=True)
    centering_summary: dict[str, object] = {"enabled": bool(args.control_center)}
    if args.control_center:
        rna_values, centering_summary["rna"] = _control_center_values(rna_values, rna_obs)
        image_values, centering_summary["image"] = _control_center_values(image_values, image_obs)
    rna_values, rna_obs = _filter_and_cap(
        rna_values,
        rna_obs,
        max_profiles_per_condition_split=args.max_profiles_per_condition_split,
        seed=13,
    )
    image_values, image_obs = _filter_and_cap(
        image_values,
        image_obs,
        max_profiles_per_condition_split=args.max_profiles_per_condition_split,
        seed=17,
    )

    shared_treatments = sorted(set(rna_obs["perturbation"]) & set(image_obs["perturbation"]))
    shared_non_control = [value for value in shared_treatments if not _is_control(value)]
    if len(shared_non_control) < args.min_shared_treatments:
        raise SystemExit(
            f"Only {len(shared_non_control)} non-control shared scGeneScope treatments remain; "
            f"minimum required is {args.min_shared_treatments}."
        )
    rna_keep = rna_obs["perturbation"].isin(shared_treatments).to_numpy()
    image_keep = image_obs["perturbation"].isin(shared_treatments).to_numpy()
    rna_values = rna_values[rna_keep]
    rna_obs = rna_obs.loc[rna_keep].reset_index(drop=True)
    image_values = image_values[image_keep]
    image_obs = image_obs.loc[image_keep].reset_index(drop=True)

    rna_output = _write_rna_h5ad(rna_values, rna_obs, args.rna_output)
    image_manifest = _write_image_arrays(
        image_values,
        image_obs,
        image_array_dir=args.image_array_dir,
        image_size=args.image_size,
    )
    args.image_output.parent.mkdir(parents=True, exist_ok=True)
    image_manifest.to_csv(args.image_output, index=False)

    summary = _summary_table(rna_obs, image_manifest)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    payload = {
        "status": "prepared",
        "source": "scGeneScope",
        "hf_repo_id": args.hf_repo_id,
        "condition_key": "condition_key_scgenescope",
        "condition_key_definition": "perturbation",
        "split_policy": args.split_policy,
        "control_centering": centering_summary,
        "rna_embedding": args.rna_obsm_key,
        "image_embedding": args.image_obsm_key or "X",
        "split_assignments": _format_split_policy(args.split_policy),
        "n_shared_treatments": int(len(shared_treatments)),
        "n_shared_non_control_treatments": int(len(shared_non_control)),
        "n_rna_profiles": int(len(rna_obs)),
        "n_image_profiles": int(len(image_manifest)),
        "n_rna_features": int(rna_values.shape[1]),
        "n_image_features": int(image_values.shape[1]),
        "feature_grid_shape": [1, int(args.image_size), int(args.image_size)],
        "profile_cap_per_condition_split": int(args.max_profiles_per_condition_split),
        "rna_output": str(rna_output),
        "image_output": str(args.image_output),
        "summary_output": str(args.summary_output),
        "source_files": {
            "rna_round1": str(args.rna_round1),
            "rna_round2": str(args.rna_round2),
            "image_round1": str(args.image_round1),
            "image_round2": str(args.image_round2),
        },
        "sha256": {
            "rna_round1": _sha256(args.rna_round1),
            "rna_round2": _sha256(args.rna_round2),
            "image_round1": _sha256(args.image_round1),
            "image_round2": _sha256(args.image_round2),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(summary.head(40).to_string(index=False))
    print(f"Shared scGeneScope treatments: {len(shared_treatments)}")
    print(f"Shared non-control treatments: {len(shared_non_control)}")
    print(f"Wrote RNA AnnData: {rna_output}")
    print(f"Wrote image manifest: {args.image_output}")
    print(f"Wrote summary: {args.summary_output}")
    return 0


def _download_defaults(args: argparse.Namespace) -> None:
    downloads = (
        ("features/rnaseq/scvi/n200/round_1.h5ad", args.rna_round1),
        ("features/rnaseq/scvi/n200/round_2.h5ad", args.rna_round2),
        ("features/imaging/imagenet/vit-l/round_1.h5ad", args.image_round1),
        ("features/imaging/imagenet/vit-l/round_2.h5ad", args.image_round2),
    )
    for hf_path, target in downloads:
        if target.exists():
            continue
        _download_hf_file(args.hf_repo_id, hf_path, target, token=args.hf_token or os.environ.get("HF_TOKEN"))


def _download_hf_file(repo_id: str, hf_path: str, target: Path, *, token: str | None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{hf_path}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    temp = target.with_suffix(target.suffix + ".tmp")
    print(f"Downloading {hf_path} -> {target}")
    with requests.get(url, stream=True, headers=headers, timeout=120) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp.replace(target)


def _load_rounds(
    paths: Iterable[tuple[Path, int]],
    *,
    modality: str,
    data_key: str | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install the data extra to read h5ad files: pip install -e '.[data]'") from exc

    values: list[np.ndarray] = []
    rows: list[pd.DataFrame] = []
    for path, round_id in paths:
        adata = ad.read_h5ad(path)
        matrix = _extract_matrix(adata, data_key=data_key)
        obs = pd.DataFrame(adata.obs).reset_index(drop=False).rename(columns={"index": "source_obs_id"})
        obs["source_round"] = str(round_id)
        obs["source_file"] = str(path)
        obs["source_modality"] = modality
        obs["_source_row"] = np.arange(len(obs), dtype=np.int64)
        values.append(matrix)
        rows.append(obs)
    stacked = np.concatenate(values, axis=0).astype(np.float32, copy=False)
    obs = pd.concat(rows, ignore_index=True, sort=False)
    obs["_value_index"] = np.arange(len(obs), dtype=np.int64)
    return stacked, obs


def _extract_matrix(adata, *, data_key: str | None) -> np.ndarray:
    if data_key and data_key in adata.obsm:
        matrix = adata.obsm[data_key]
    elif data_key and data_key not in adata.obsm:
        raise KeyError(f"AnnData object does not contain obsm key {data_key!r}")
    else:
        matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _standardize_obs(
    obs: pd.DataFrame,
    *,
    include_controls: bool,
    official_treatments_only: bool,
    split_policy: str,
) -> pd.DataFrame:
    frame = obs.copy()
    treatment_col = _find_column(frame, ("Treatment", "treatment", "perturbation", "compound"))
    replicate_col = _find_column(frame, ("Replicate", "replicate"))
    batch_col = _find_column(frame, ("batch", "Batch"))
    plate_col = _find_column(frame, ("plate", "Plate", "Metadata_Plate"))
    well_col = _find_column(frame, ("well", "Well", "Metadata_Well"))
    site_col = _find_column(frame, ("site", "Site", "Metadata_Site"))
    moa_col = _find_column(frame, ("moa", "MoA", "mechanism", "Mechanism", "pathway", "Pathway"), default=None)
    target_col = _find_column(frame, ("target", "Target", "target_gene", "TargetGene"), default=None)
    if treatment_col is None:
        raise SystemExit("scGeneScope AnnData.obs is missing a Treatment/treatment column")
    if replicate_col is None:
        raise SystemExit("scGeneScope AnnData.obs is missing a Replicate/replicate column")

    frame["perturbation"] = frame[treatment_col].map(normalize_value)
    control = frame["perturbation"].map(_is_control)
    treatment_set = set(OFFICIAL_TREATMENTS)
    keep = frame["perturbation"].isin(treatment_set)
    if include_controls:
        keep = keep | control
    if not official_treatments_only:
        keep = pd.Series(True, index=frame.index) if include_controls else ~control
    frame = frame.loc[keep].copy()
    control = frame["perturbation"].map(_is_control)
    frame["perturbation_type"] = np.where(control, "control", "compound")
    frame["dose"] = "NA"
    frame["time"] = "NA"
    frame["cell_line"] = "U2OS"
    replicate = frame[replicate_col].map(_normalize_integerish)
    round_id = frame["source_round"].map(_normalize_integerish)
    frame["replicate"] = replicate
    frame["round"] = round_id
    frame["split"] = [
        _split_for_round_replicate(round_value, rep_value, split_policy=split_policy)
        for round_value, rep_value in zip(round_id, replicate)
    ]
    frame = frame.loc[frame["split"].ne("ignore")].copy()
    batch_value = frame[batch_col].map(normalize_value) if batch_col else pd.Series("unknown", index=frame.index)
    frame["batch"] = [
        f"round_{round_value}|replicate_{rep_value}|batch_{batch}"
        for round_value, rep_value, batch in zip(frame["round"], frame["replicate"], batch_value)
    ]
    frame["plate"] = frame[plate_col].map(normalize_value) if plate_col else frame["batch"]
    frame["well"] = frame[well_col].map(normalize_value) if well_col else "unknown"
    frame["site"] = frame[site_col].map(normalize_value) if site_col else frame["_source_row"].astype(str)
    frame["channel_or_z"] = "embedding"
    frame["compound"] = frame["perturbation"]
    frame["moa"] = frame[moa_col].map(normalize_value) if moa_col else ""
    frame["target_gene"] = frame[target_col].map(normalize_value) if target_col else ""
    frame["condition_key_scgenescope"] = frame["perturbation"].map(normalize_value)
    return frame.reset_index(drop=True)


def _filter_and_cap(
    values: np.ndarray,
    obs: pd.DataFrame,
    *,
    max_profiles_per_condition_split: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(values) != len(obs):
        raise ValueError("values and obs row counts do not match")
    obs = obs.reset_index(drop=True).copy()
    keep = np.ones(len(obs), dtype=bool)
    if max_profiles_per_condition_split > 0:
        keep[:] = False
        rng = np.random.default_rng(seed)
        for _, group in obs.groupby(["split", "perturbation"], sort=True, observed=True):
            indices = group.index.to_numpy()
            if len(indices) > max_profiles_per_condition_split:
                indices = np.sort(rng.choice(indices, size=max_profiles_per_condition_split, replace=False))
            keep[indices] = True
    obs = obs.loc[keep].reset_index(drop=True)
    values = values[keep]
    return values, obs


def _control_center_values(values: np.ndarray, obs: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    if "batch" not in obs.columns:
        raise ValueError("control centering requires a batch column")
    if "perturbation_type" in obs.columns:
        control_mask = obs["perturbation_type"].astype(str).str.lower().eq("control").to_numpy()
    else:
        control_mask = obs["perturbation"].map(_is_control).to_numpy()
    summary: dict[str, object] = {
        "controls": int(control_mask.sum()),
        "groups_total": int(obs["batch"].nunique()),
        "groups_centered": 0,
        "groups_global_fallback": 0,
    }
    if not bool(control_mask.any()):
        summary["status"] = "skipped_no_controls"
        return values, summary

    centered = np.asarray(values, dtype=np.float32).copy()
    global_control_mean = centered[control_mask].mean(axis=0)
    centered_groups = 0
    fallback_groups = 0
    for batch, group in obs.groupby("batch", sort=True, observed=True):
        indices = group.index.to_numpy()
        group_controls = indices[control_mask[indices]]
        if len(group_controls) == 0:
            fallback_groups += 1
            continue
        batch_control_mean = centered[group_controls].mean(axis=0)
        centered[indices] = centered[indices] - batch_control_mean + global_control_mean
        centered_groups += 1
    summary["groups_centered"] = int(centered_groups)
    summary["groups_global_fallback"] = int(fallback_groups)
    summary["status"] = "centered"
    return centered.astype(np.float32, copy=False), summary


def _write_rna_h5ad(values: np.ndarray, obs: pd.DataFrame, output: Path) -> Path:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install the data extra to write h5ad files: pip install -e '.[data]'") from exc

    normalized = normalize_scrna_obs(obs)
    normalized["condition_key_scgenescope"] = obs["condition_key_scgenescope"].values
    normalized["condition_key"] = normalized["condition_key_scgenescope"]
    normalized["split"] = obs["split"].values
    for column in ("round", "replicate", "plate", "well", "site", "source_obs_id", "source_file", "source_modality"):
        if column in obs.columns:
            normalized[column] = obs[column].values
    var = pd.DataFrame(
        {
            "gene_id": [f"scvi_{index:04d}" for index in range(values.shape[1])],
            "gene_symbol": [f"scvi_{index:04d}" for index in range(values.shape[1])],
        }
    )
    adata = ad.AnnData(X=values.astype(np.float32, copy=False), obs=normalized, var=var)
    adata.obs_names = [f"rna_{index:08d}" for index in range(adata.n_obs)]
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    return output


def _write_image_arrays(
    values: np.ndarray,
    obs: pd.DataFrame,
    *,
    image_array_dir: Path,
    image_size: int,
) -> pd.DataFrame:
    image_array_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, (vector, row) in enumerate(zip(values, obs.to_dict(orient="records"), strict=True)):
        split = normalize_value(row["split"])
        perturbation = normalize_value(row["perturbation"])
        file_name = f"{split}_{_slug(perturbation)}_{index:08d}.npy"
        path = image_array_dir / split / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, _vector_to_grid(vector, image_size=image_size))
        rows.append(
            {
                "image_path": str(path),
                "plate": normalize_value(row.get("plate", "unknown")),
                "well": normalize_value(row.get("well", "unknown")),
                "site": normalize_value(row.get("site", index)),
                "channel_or_z": "embedding",
                "perturbation": perturbation,
                "compound": normalize_value(row.get("compound", perturbation)),
                "moa": normalize_value(row.get("moa", "")),
                "target_gene": normalize_value(row.get("target_gene", "")),
                "dose": "NA",
                "time": "NA",
                "cell_line": "U2OS",
                "batch": normalize_value(row.get("batch", "unknown")),
                "perturbation_type": normalize_value(row.get("perturbation_type", "compound")),
                "split": split,
                "round": normalize_value(row.get("round", "NA")),
                "replicate": normalize_value(row.get("replicate", "NA")),
                "condition_key_scgenescope": perturbation,
            }
        )
    manifest = normalize_image_manifest(pd.DataFrame(rows))
    manifest["condition_key_scgenescope"] = [row["condition_key_scgenescope"] for row in rows]
    manifest["condition_key"] = manifest["condition_key_scgenescope"]
    manifest["split"] = [row["split"] for row in rows]
    manifest["round"] = [row["round"] for row in rows]
    manifest["replicate"] = [row["replicate"] for row in rows]
    return manifest


def _vector_to_grid(vector: np.ndarray, *, image_size: int) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    width = image_size * image_size
    if flat.shape[0] > width:
        raise ValueError(
            f"embedding dimension {flat.shape[0]} does not fit in 1x{image_size}x{image_size}; "
            "increase --image-size"
        )
    padded = np.zeros(width, dtype=np.float32)
    padded[: flat.shape[0]] = flat
    return padded.reshape(1, image_size, image_size)


def _summary_table(rna_obs: pd.DataFrame, image_manifest: pd.DataFrame) -> pd.DataFrame:
    rna = (
        rna_obs.groupby(["split", "condition_key_scgenescope", "perturbation"], sort=True, observed=True)
        .size()
        .rename("n_rna")
        .reset_index()
    )
    image = (
        image_manifest.groupby(["split", "condition_key_scgenescope", "perturbation"], sort=True, observed=True)
        .size()
        .rename("n_image")
        .reset_index()
    )
    summary = rna.merge(image, on=["split", "condition_key_scgenescope", "perturbation"], how="outer").fillna(0)
    summary["n_rna"] = summary["n_rna"].astype(int)
    summary["n_image"] = summary["n_image"].astype(int)
    return summary.sort_values(["split", "perturbation"]).reset_index(drop=True)


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], *, default: str | None = None) -> str | None:
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return default


def _is_control(value: object) -> bool:
    token = normalize_value(value).strip().lower()
    return token in CONTROL_TOKENS or "dmso" in token


def _split_for_round_replicate(round_value: str, replicate_value: str, *, split_policy: str = "official") -> str:
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(f"unknown scGeneScope split_policy {split_policy!r}")
    return SPLIT_POLICIES[split_policy].get((round_value, replicate_value), "ignore")


def _format_split_policy(split_policy: str) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    for (round_value, replicate), split in sorted(SPLIT_POLICIES[split_policy].items()):
        assignments.setdefault(split, []).append(f"round_{round_value} Replicate={replicate}")
    return assignments


def _normalize_integerish(value: object) -> str:
    text = normalize_value(value)
    match = re.search(r"\d+", text)
    return str(int(match.group(0))) if match else text


def _slug(value: object) -> str:
    text = normalize_value(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:80] or "unknown"


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}. Pass --download to fetch default embedding files.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
