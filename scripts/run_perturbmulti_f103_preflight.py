from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import h5py
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
import numpy as np
import pandas as pd
import requests


HF_REPO_ID = "xingjiepan/PerturbMulti"
HF_REVISION = "8aac954eb631b68f6e11171a8313db61cc16c38c"
RNA_FILE = "RNA_scaled_crispr_screen_20240615.h5ad"
PROTEIN_FILE = "protein_intensities_crispr_screen_20240615.h5ad"
IMAGE_TAR_PREFIX = "crispr_screen_20240615_chunk_"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/autoresearch_total_autonomy_bioguard_wm_jepa/"
    "experiments/F103_perturbmulti_rna_obs_pairing_preflight"
)
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT.md"
REQUIRED_OBS_COLUMNS = ("cell_id", "cell_name", "condition", "perturbation", "fov", "x", "y", "z", "dataset")
SPLIT_USABLE_COLUMNS = ("dataset", "fov", "cell_name", "perturbation")
ID_ALIAS_COLUMNS = ("cell_id", "id", "_index", "cell_name")
PERTURBATION_ALIAS_COLUMNS = ("perturbation", "singlet_gene", "singlet_name", "bc1", "bc3")
COORDINATE_ALIAS_COLUMNS = ("fov", "x", "y", "z", "global_x", "global_y")
TAR_ID_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".npy", ".npz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run F103 PerturbMulti RNA obs and image/RNA/protein pairing preflight."
    )
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("/content/hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--download-rna", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-protein", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tar-prefix-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-tar-files", type=int, default=6)
    parser.add_argument("--max-image-members", type=int, default=200)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    env = _environment_summary()
    repo_files = _repo_manifest(args.repo_id, args.revision)
    manifest_frame = pd.DataFrame(repo_files)
    manifest_frame.to_csv(args.output_dir / "f103_perturbmulti_manifest.tsv", sep="\t", index=False)

    downloads = _ensure_h5ad_files(args)
    rna_path = Path(downloads[RNA_FILE]["local_path"])
    protein_path = Path(downloads[PROTEIN_FILE]["local_path"])

    rna_summary, rna_values = summarize_h5ad_obs(rna_path, label="rna")
    protein_summary, protein_values = summarize_h5ad_obs(protein_path, label="protein")
    obs_frame = pd.DataFrame([rna_summary, protein_summary])
    obs_frame.to_csv(args.output_dir / "f103_h5ad_obs_probe.tsv", sep="\t", index=False)

    rna_ids = _id_set_from_values(rna_values, primary_column=primary_id_column(rna_values))
    protein_ids = _id_set_from_values(protein_values, primary_column=primary_id_column(protein_values))
    rna_alias_ids = _id_set_from_values(rna_values)
    protein_alias_ids = _id_set_from_values(protein_values)
    image_rows, image_errors = _sample_image_tar_members(
        args.repo_id,
        args.revision,
        [row["path"] for row in repo_files if row["path"].startswith(IMAGE_TAR_PREFIX) and row["path"].endswith(".tar")],
        max_tar_files=args.max_tar_files,
        prefix_bytes=args.tar_prefix_bytes,
        max_members=args.max_image_members,
        timeout=args.request_timeout,
    )
    image_frame = pd.DataFrame(image_rows)
    if image_frame.empty:
        image_frame = pd.DataFrame(
            columns=["tar_file", "member_name", "normalized_image_id", "size_bytes", "typeflag", "offset"]
        )
    image_frame.to_csv(args.output_dir / "f103_image_tar_member_sample.tsv", sep="\t", index=False)

    image_ids = set(image_frame["normalized_image_id"].dropna().astype(str)) if "normalized_image_id" in image_frame else set()
    image_ids.discard("")
    overlap = _overlap_summary(rna_ids, protein_ids, rna_alias_ids, protein_alias_ids, image_ids)
    overlap_frame = pd.DataFrame([overlap])
    overlap_frame.to_csv(args.output_dir / "f103_pairing_overlap.tsv", sep="\t", index=False)

    preflight_pass = bool(
        rna_summary["contract_usable"]
        and protein_summary["contract_usable"]
        and overlap["rna_protein_cell_id_overlap"] > 0
        and overlap["image_rna_alias_overlap"] > 0
        and overlap["image_protein_alias_overlap"] > 0
    )
    decision = (
        "F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT_PASS_READY_FOR_SEALED_F082_VALIDATION_DESIGN"
        if preflight_pass
        else "F103_PERTURBMULTI_RNA_OBS_AND_PAIRING_PREFLIGHT_BLOCKED_NO_MODEL_RUN"
    )
    payload = {
        "decision": decision,
        "preflight_pass": preflight_pass,
        "environment": env,
        "downloads": downloads,
        "rna_summary": _json_safe(rna_summary),
        "protein_summary": _json_safe(protein_summary),
        "overlap": _json_safe(overlap),
        "image_tar_errors": image_errors,
        "model_trained": False,
        "model_promoted": False,
        "raw_data_outside_git": True,
        "notes": "HDF5 obs-only H5AD reads and HTTP range-read tar headers only; no .X matrix reads and no model fitting.",
    }
    _write_json(args.output_dir / "f103_preflight_summary.json", payload)
    _write_report(args.report_path, payload, manifest_frame, obs_frame, overlap_frame, image_frame)
    return 0 if preflight_pass else 1


def _environment_summary() -> dict[str, Any]:
    return {
        "cwd": str(Path.cwd()),
        "hf_home": os.environ.get("HF_HOME", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pid": os.getpid(),
    }


def _repo_manifest(repo_id: str, revision: str) -> list[dict[str, Any]]:
    info = HfApi().dataset_info(repo_id, revision=revision)
    rows = []
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        rows.append(
            {
                "path": sibling.rfilename,
                "size_bytes": getattr(sibling, "size", None),
                "is_crispr_rna": sibling.rfilename == RNA_FILE,
                "is_crispr_protein": sibling.rfilename == PROTEIN_FILE,
                "is_crispr_image_tar": sibling.rfilename.startswith(IMAGE_TAR_PREFIX)
                and sibling.rfilename.endswith(".tar"),
            }
        )
    return rows


def _ensure_h5ad_files(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for filename, should_download in ((RNA_FILE, args.download_rna), (PROTEIN_FILE, args.download_protein)):
        local_path = _cached_snapshot_path(args.cache_dir, args.repo_id, args.revision, filename)
        existed_before = local_path.exists()
        if not existed_before:
            if not should_download:
                raise FileNotFoundError(f"{filename} not found in {args.cache_dir} and download is disabled")
            local_path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=args.revision,
                    cache_dir=str(args.cache_dir),
                )
            )
        status[filename] = {
            "local_path": str(local_path),
            "exists": local_path.exists(),
            "existed_before": existed_before,
            "size_bytes": int(local_path.stat().st_size) if local_path.exists() else 0,
        }
    return status


def _cached_snapshot_path(cache_dir: Path, repo_id: str, revision: str, filename: str) -> Path:
    namespace, name = repo_id.split("/", 1)
    candidates = [
        cache_dir / "hub" / f"datasets--{namespace}--{name}" / "snapshots" / revision / filename,
        cache_dir / f"datasets--{namespace}--{name}" / "snapshots" / revision / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def summarize_h5ad_obs(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        if "obs" not in handle:
            raise KeyError(f"{path} does not contain an obs group")
        obs = handle["obs"]
        columns = obs_columns(obs)
        index_key = decode_value(obs.attrs.get("_index", "_index"))
        requested = sorted(
            set(REQUIRED_OBS_COLUMNS)
            | set(ID_ALIAS_COLUMNS)
            | set(PERTURBATION_ALIAS_COLUMNS)
            | set(COORDINATE_ALIAS_COLUMNS)
            | {index_key}
        )
        values = {column: read_h5ad_elem(obs[column]) for column in requested if column in obs}
        if index_key in values:
            values.setdefault("_index", values[index_key])
        id_aliases = [column for column in ID_ALIAS_COLUMNS if column in values]
        perturbation_aliases = [column for column in PERTURBATION_ALIAS_COLUMNS if column in values]
        coordinate_aliases = [column for column in COORDINATE_ALIAS_COLUMNS if column in values]
        n_obs = axis_length(obs)
        n_vars = var_length(handle)
        rows: dict[str, Any] = {
            "label": label,
            "path": str(path),
            "index_key": index_key,
            "n_obs": int(n_obs),
            "n_vars": int(n_vars),
            "obs_columns": ",".join(columns),
            "required_obs_columns_present": bool(set(REQUIRED_OBS_COLUMNS).issubset(columns)),
            "missing_required_obs_columns": ",".join(sorted(set(REQUIRED_OBS_COLUMNS) - set(columns))),
            "split_usable_columns_present": ",".join([column for column in SPLIT_USABLE_COLUMNS if column in columns]),
            "id_alias_columns_present": ",".join(id_aliases),
            "perturbation_alias_columns_present": ",".join(perturbation_aliases),
            "coordinate_alias_columns_present": ",".join(coordinate_aliases),
            "contract_usable": bool(
                id_aliases
                and perturbation_aliases
                and "fov" in values
                and "x" in values
                and "y" in values
            ),
        }
        for column in REQUIRED_OBS_COLUMNS:
            if column in values:
                rows[f"{column}_unique"] = int(pd.Series(values[column]).nunique(dropna=True))
                rows[f"{column}_head"] = ",".join(map(str, values[column][:5]))
            else:
                rows[f"{column}_unique"] = 0
                rows[f"{column}_head"] = ""
        return rows, values


def obs_columns(obs: h5py.Group) -> list[str]:
    if "column-order" in obs.attrs:
        return [decode_value(value) for value in obs.attrs["column-order"]]
    return sorted(key for key in obs.keys() if key != "_index")


def axis_length(obs: h5py.Group) -> int:
    index_key = decode_value(obs.attrs.get("_index", "_index"))
    if index_key in obs and isinstance(obs[index_key], h5py.Dataset):
        return int(obs[index_key].shape[0])
    for node in obs.values():
        if isinstance(node, h5py.Dataset) and node.shape:
            return int(node.shape[0])
        if isinstance(node, h5py.Group) and "codes" in node:
            return int(node["codes"].shape[0])
    raise ValueError("Could not infer obs row count")


def var_length(handle: h5py.File) -> int:
    if "var" in handle:
        var = handle["var"]
        index_key = decode_value(var.attrs.get("_index", "_index"))
        if index_key in var and isinstance(var[index_key], h5py.Dataset):
            return int(var[index_key].shape[0])
    if "X" in handle:
        x = handle["X"]
        if isinstance(x, h5py.Dataset):
            return int(x.shape[1])
        if "shape" in x.attrs:
            return int(x.attrs["shape"][1])
    return 0


def read_h5ad_elem(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return decode_array(node[()])
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = decode_array(node["categories"][()])
        codes = np.asarray(node["codes"][()])
        out = np.empty(codes.shape[0], dtype=object)
        for idx, code in enumerate(codes):
            out[idx] = "" if int(code) < 0 else categories[int(code)]
        return out
    raise TypeError(f"Unsupported H5AD obs encoding for {node.name}")


def decode_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {"S", "O", "U"}:
        return np.asarray([decode_value(value) for value in arr], dtype=object)
    return arr


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _id_set_from_values(values: dict[str, np.ndarray], primary_column: str | None = None) -> set[str]:
    columns = [primary_column] if primary_column else ["cell_id", "_index", "cell_name"]
    ids: set[str] = set()
    for column in columns:
        if not column or column not in values:
            continue
        for value in values[column]:
            normalized = normalize_cell_id(value)
            if normalized:
                ids.add(normalized)
    return ids


def primary_id_column(values: dict[str, np.ndarray]) -> str | None:
    for column in ("id", "cell_name", "_index", "cell_id"):
        if column in values:
            return column
    return None


def normalize_cell_id(value: Any) -> str:
    text = decode_value(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.split("?", 1)[0].split("#", 1)[0]
    text = text.replace("\\", "/").rstrip("/")
    text = text.rsplit("/", 1)[-1]
    lowered = text.lower()
    changed = True
    while changed:
        changed = False
        for suffix in TAR_ID_SUFFIXES:
            if lowered.endswith(suffix):
                text = text[: -len(suffix)]
                lowered = text.lower()
                changed = True
                break
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _sample_image_tar_members(
    repo_id: str,
    revision: str,
    tar_files: list[str],
    *,
    max_tar_files: int,
    prefix_bytes: int,
    max_members: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for tar_file in sorted(tar_files)[:max_tar_files]:
        try:
            url = hf_hub_url(repo_id=repo_id, filename=tar_file, repo_type="dataset", revision=revision)
            data, status_code, content_range = _range_read_prefix(url, prefix_bytes, timeout=timeout)
            parsed = parse_tar_members_from_prefix(data, tar_file=tar_file)
            for row in parsed:
                rows.append(row)
                if len(rows) >= max_members:
                    return rows, errors
            if not parsed:
                errors.append(
                    {
                        "tar_file": tar_file,
                        "error": "no_tar_members_parsed_from_prefix",
                        "status_code": status_code,
                        "content_range": content_range,
                        "bytes_read": len(data),
                    }
                )
        except Exception as exc:  # pragma: no cover - exercised by live preflight failures.
            errors.append({"tar_file": tar_file, "error": repr(exc)})
    return rows, errors


def _range_read_prefix(url: str, n_bytes: int, *, timeout: float) -> tuple[bytes, int, str]:
    chunks: list[bytes] = []
    total = 0
    headers = {"Range": f"bytes=0-{max(int(n_bytes) - 1, 0)}"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= n_bytes:
                break
        return b"".join(chunks)[:n_bytes], int(response.status_code), response.headers.get("Content-Range", "")


def parse_tar_members_from_prefix(data: bytes, *, tar_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset + 512 <= len(data):
        block = data[offset : offset + 512]
        if block == b"\0" * 512:
            break
        name = _tar_text(block[0:100])
        prefix = _tar_text(block[345:500])
        if prefix:
            name = f"{prefix}/{name}" if name else prefix
        if not name:
            break
        size_text = _tar_text(block[124:136]).strip() or "0"
        try:
            size = int(size_text, 8)
        except ValueError:
            break
        typeflag = _tar_text(block[156:157]) or "0"
        rows.append(
            {
                "tar_file": tar_file,
                "member_name": name,
                "normalized_image_id": normalize_cell_id(name),
                "size_bytes": int(size),
                "typeflag": typeflag,
                "offset": int(offset),
            }
        )
        data_blocks = int(math.ceil(size / 512.0))
        next_offset = offset + 512 + data_blocks * 512
        if next_offset <= offset:
            break
        offset = next_offset
    return rows


def _tar_text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _overlap_summary(
    rna_ids: set[str],
    protein_ids: set[str],
    rna_alias_ids: set[str],
    protein_alias_ids: set[str],
    image_ids: set[str],
) -> dict[str, Any]:
    rna_protein = rna_ids & protein_ids
    image_rna = image_ids & rna_alias_ids
    image_protein = image_ids & protein_alias_ids
    image_all = image_ids & rna_alias_ids & protein_alias_ids
    return {
        "rna_cell_id_count": len(rna_ids),
        "protein_cell_id_count": len(protein_ids),
        "rna_alias_id_count": len(rna_alias_ids),
        "protein_alias_id_count": len(protein_alias_ids),
        "image_sample_id_count": len(image_ids),
        "rna_protein_cell_id_overlap": len(rna_protein),
        "rna_protein_cell_id_overlap_fraction_of_rna": _safe_fraction(len(rna_protein), len(rna_ids)),
        "rna_protein_cell_id_overlap_fraction_of_protein": _safe_fraction(len(rna_protein), len(protein_ids)),
        "image_rna_alias_overlap": len(image_rna),
        "image_protein_alias_overlap": len(image_protein),
        "image_rna_protein_alias_overlap": len(image_all),
        "rna_protein_examples": ",".join(sorted(rna_protein)[:5]),
        "image_rna_examples": ",".join(sorted(image_rna)[:5]),
        "image_protein_examples": ",".join(sorted(image_protein)[:5]),
    }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _write_report(
    path: Path,
    payload: dict[str, Any],
    manifest_frame: pd.DataFrame,
    obs_frame: pd.DataFrame,
    overlap_frame: pd.DataFrame,
    image_frame: pd.DataFrame,
) -> None:
    manifest_view = manifest_frame.loc[
        manifest_frame["is_crispr_rna"] | manifest_frame["is_crispr_protein"] | manifest_frame["is_crispr_image_tar"]
    ].head(16)
    image_view = image_frame.head(20)
    lines = [
        "# F103 PerturbMulti RNA Obs and Pairing Preflight",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        "No model is promoted. No new architecture is trained. The protected rank-3 train-split-only PLS raw-linear readout remains the model of record.",
        "",
        "## Scope",
        "- candidate: PerturbMulti CRISPR screen",
        "- model path under protection: frozen F082/F096 ProgramBootstrapJEPA path",
        "- checks run: Hugging Face manifest, RNA/protein H5AD obs-only HDF5 reads, image tar header range samples",
        "- checks not run: `.X` matrix loading, image payload extraction, model fitting, calibration, or promotion",
        "",
        "## H5AD Obs Probe",
        obs_frame.to_csv(sep="\t", index=False),
        "## Pairing Overlap",
        overlap_frame.to_csv(sep="\t", index=False),
        "## Image Tar Header Sample",
        image_view.to_csv(sep="\t", index=False),
        "## Relevant Manifest Rows",
        manifest_view.to_csv(sep="\t", index=False),
        "## Raw Data Handling",
        f"- RNA H5AD local path: `{payload['downloads'][RNA_FILE]['local_path']}`",
        f"- Protein H5AD local path: `{payload['downloads'][PROTEIN_FILE]['local_path']}`",
        "- raw payloads are outside git under the Hugging Face cache",
        "",
        "## Next Step",
        "If this preflight passes, design a small sealed PerturbMulti F082 validation run on GPU. Keep PLS/full-ridge as the protected audit floor only and do not promote without a fresh external Tier 3 pass.",
        "",
        "## Machine-Readable Summary",
        "```json",
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
