from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class DatasetDownload:
    key: str
    name: str
    modality: str
    pairing_tier: str
    access_url: str
    command: str
    output_hint: str
    license_note: str
    commercial_status: str
    size_note: str
    metadata_only: bool
    direct_download_ok: bool
    checksum: str | None = None


DATASETS: tuple[DatasetDownload, ...] = (
    DatasetDownload(
        key="sciplex3",
        name="Sci-Plex3 / Srivatsan-Trapnell",
        modality="scRNA-seq chemical perturbation",
        pairing_tier="Tier 4 RNA-only",
        access_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944",
        command=(
            "curl -L 'https://zenodo.org/records/13350497/files/"
            "SrivatsanTrapnell2020_sciplex3.h5ad?download=1' "
            "-o data/raw/SrivatsanTrapnell2020_sciplex3.h5ad"
        ),
        output_hint="data/raw/SrivatsanTrapnell2020_sciplex3.h5ad",
        license_note="GEO public; scPerturb Zenodo harmonized files are CC BY 4.0.",
        commercial_status="commercial_ok",
        size_note="large h5ad / GEO raw tar about 9 GB",
        metadata_only=False,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="replogle",
        name="Replogle/Weissman Perturb-seq",
        modality="scRNA-seq CRISPRi perturbation",
        pairing_tier="Tier 4 RNA-only",
        access_url="https://plus.figshare.com/articles/dataset/Replogle_et_al_2022_K562_RPE1_GWPS/20029387",
        command=(
            "curl -L 'https://zenodo.org/records/13350497/files/"
            "ReplogleWeissman2022_K562_essential.h5ad?download=1' "
            "-o data/raw/ReplogleWeissman2022_K562_essential.h5ad"
        ),
        output_hint="data/raw/ReplogleWeissman2022_K562_essential.h5ad",
        license_note="Figshare+ and scPerturb harmonized files reported as CC BY 4.0.",
        commercial_status="commercial_ok",
        size_note="large h5ad files; choose K562/RPE1 subset explicitly",
        metadata_only=False,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="norman",
        name="Norman/Weissman Perturb-seq",
        modality="scRNA-seq CRISPRa perturbation",
        pairing_tier="Tier 4 RNA-only",
        access_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344",
        command=(
            "curl -L 'https://zenodo.org/records/13350497/files/"
            "NormanWeissman2019_filtered.h5ad?download=1' "
            "-o data/raw/NormanWeissman2019_filtered.h5ad"
        ),
        output_hint="data/raw/NormanWeissman2019_filtered.h5ad",
        license_note="GEO public without explicit reuse license; scPerturb harmonized file is CC BY 4.0.",
        commercial_status="commercial_ok",
        size_note="large h5ad / GEO matrix files",
        metadata_only=False,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="adamson",
        name="Adamson/Weissman Perturb-seq",
        modality="scRNA-seq CRISPRi perturbation",
        pairing_tier="Tier 4 RNA-only",
        access_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546",
        command="curl -L 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE90546&format=file' -o data/raw/GSE90546_RAW.tar",
        output_hint="data/raw/GSE90546_RAW.tar",
        license_note="GEO public; no explicit commercial reuse license found on GEO record.",
        commercial_status="unclear",
        size_note="GEO raw/processed archive",
        metadata_only=False,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="dixit",
        name="Dixit/Regev Perturb-seq",
        modality="scRNA-seq CRISPR perturbation",
        pairing_tier="Tier 4 RNA-only",
        access_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063",
        command="curl -L 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE90063&format=file' -o data/raw/GSE90063_RAW.tar",
        output_hint="data/raw/GSE90063_RAW.tar",
        license_note="GEO public; no explicit commercial reuse license found on GEO record.",
        commercial_status="unclear",
        size_note="GEO raw archive",
        metadata_only=False,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="bf-moa",
        name="BF-MoA brightfield mechanism-of-action images",
        modality="brightfield and Cell Painting microscopy",
        pairing_tier="Tier 4 image-only / compound-level weak correspondence",
        access_url="https://doi.org/10.17044/SCILIFELAB.21378906",
        command="curl -L 'https://ndownloader.figshare.com/files/37984380' -o data/raw/bf_moa_data_tables.tar.gz",
        output_hint="data/raw/bf_moa_data_tables.tar.gz",
        license_note="SciLifeLab Figshare dataset is CC BY 4.0.",
        commercial_status="commercial_ok",
        size_note="metadata archive is small; full image download is about 590 GB",
        metadata_only=True,
        direct_download_ok=True,
    ),
    DatasetDownload(
        key="rxrx1",
        name="RxRx1",
        modality="fluorescence microscopy siRNA perturbation",
        pairing_tier="Tier 4 image-only",
        access_url="https://www.rxrx.ai/rxrx1",
        command="curl -L 'https://www.rxrx.ai/rxrx1' -o data/raw/rxrx1_access_instructions.html",
        output_hint="data/raw/rxrx1_access_instructions.html",
        license_note="Recursion license / CC BY-NC-SA 4.0; commercial reuse is not safe without separate permission.",
        commercial_status="research_only",
        size_note="125k+ images; download through RxRx portal tooling",
        metadata_only=True,
        direct_download_ok=True,
    ),
    DatasetDownload(
        key="jump-cell-painting",
        name="JUMP Cell Painting cpg0016",
        modality="Cell Painting microscopy compound, ORF, and CRISPR perturbations",
        pairing_tier="Tier 4 image-only / gene-perturbation weak correspondence",
        access_url="https://registry.opendata.aws/cellpainting-gallery/",
        command="aws s3 ls --no-sign-request s3://cellpainting-gallery/cpg0016-jump/",
        output_hint="s3://cellpainting-gallery/cpg0016-jump/",
        license_note="Cell Painting Gallery/JUMP data are released as CC0; metadata repository is BSD-3-Clause.",
        commercial_status="commercial_ok",
        size_note="multi-TB raw images and profiles; use metadata/profile subsets first",
        metadata_only=True,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="visium",
        name="10x Genomics Visium public datasets",
        modality="spatial transcriptomics + histology images",
        pairing_tier="Tier 1 spot-level image-expression",
        access_url="https://www.10xgenomics.com/resources/datasets",
        command="python -m webbrowser 'https://www.10xgenomics.com/resources/datasets'",
        output_hint="user-selected Space Ranger output folder",
        license_note="10x public example data terms vary by dataset; verify selected dataset before commercial reuse.",
        commercial_status="unclear",
        size_note="varies by tissue; includes filtered matrices, spatial coordinates, and images",
        metadata_only=True,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="spatiallibd",
        name="SpatialLIBD human DLPFC Visium",
        modality="spatial transcriptomics + histology images",
        pairing_tier="Tier 1 spot-level image-expression",
        access_url="https://www.bioconductor.org/packages/release/data/experiment/html/spatialLIBD.html",
        command=(
            "Rscript -e 'BiocManager::install(\"spatialLIBD\"); "
            "path <- spatialLIBD::fetch_data(type=\"spe\"); print(path)'"
        ),
        output_hint="Bioconductor spatialLIBD cache",
        license_note="Bioconductor package license Artistic-2.0; raw data commercial terms should be verified.",
        commercial_status="unclear",
        size_note="about 47k Visium spots across human DLPFC samples",
        metadata_only=True,
        direct_download_ok=False,
    ),
    DatasetDownload(
        key="optical-pooled",
        name="PERISCOPE / pooled Cell Painting optical screen",
        modality="Cell Painting + in situ sequencing barcode readout",
        pairing_tier="Tier 3 image-to-genotype paired",
        access_url="https://registry.opendata.aws/cellpainting-gallery/",
        command="aws s3 ls --no-sign-request s3://cellpainting-gallery/cpg0021-periscope/",
        output_hint="s3://cellpainting-gallery/cpg0021-periscope/",
        license_note="Cell Painting Gallery is CC0; associated code repositories are permissively licensed.",
        commercial_status="commercial_ok",
        size_note="large raw/corrected/cropped image and profile tree",
        metadata_only=True,
        direct_download_ok=False,
    ),
)

ALIASES: dict[str, tuple[str, ...]] = {
    "all-metadata": tuple(item.key for item in DATASETS if item.metadata_only),
    "all": tuple(item.key for item in DATASETS),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or print commands for public Perturb-JEPA benchmark assets.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset key. May be repeated. Supported: sciplex3, replogle, norman, adamson, dixit, "
            "bf-moa, rxrx1, jump-cell-painting, visium, spatiallibd, optical-pooled, all-metadata, all."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Alias for --dataset all.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifest without downloading.")
    parser.add_argument("--download-large", action="store_true", help="Allow large downloads. Otherwise large assets are only printed.")
    parser.add_argument("--output-dir", default="data/raw", type=Path, help="Destination directory for direct downloads.")
    parser.add_argument(
        "--manifest-out",
        default=Path("data/download_manifest.json"),
        type=Path,
        help="JSON manifest written for every run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_keys = list(args.dataset)
    if args.all:
        selected_keys.append("all")
    if not selected_keys:
        raise SystemExit("error: pass --dataset at least once, --dataset all-metadata, or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_datasets(selected_keys)
    records = []
    for item in selected:
        print(f"\n# {item.key}: {item.name}")
        print(f"# {item.modality}; {item.pairing_tier}")
        print(f"# access: {item.access_url}")
        print(f"# license: {item.license_note} commercial_status={item.commercial_status}")
        print(item.command)
        status = "printed"
        output_path = _resolve_output(output_dir, item.output_hint)
        if item.direct_download_ok and not args.dry_run:
            if item.metadata_only or args.download_large:
                status = _download(item.access_url if item.key == "rxrx1" else _command_url(item.command), output_path)
            else:
                status = "skipped_large_without_download_large"
        elif not args.dry_run and not item.metadata_only and not args.download_large:
            print("# skipped: large asset; rerun with --download-large to execute this download")
            status = "skipped_large_without_download_large"
        elif not args.dry_run and args.download_large:
            subprocess.run(item.command, shell=True, check=True)
            status = "executed_command"
        records.append(
            {
                **asdict(item),
                "status": status,
                "date_accessed_utc": datetime.now(timezone.utc).isoformat(),
                "resolved_output": str(output_path),
                "sha256": _sha256(output_path) if output_path.exists() and output_path.is_file() else item.checksum,
            }
        )
    _write_manifest(args.manifest_out, records)
    print(f"\nWrote download manifest: {args.manifest_out}")
    return 0


def _select_datasets(keys: Iterable[str]) -> list[DatasetDownload]:
    expanded: list[str] = []
    valid = {item.key for item in DATASETS}
    for key in keys:
        if key in ALIASES:
            expanded.extend(ALIASES[key])
        elif key in valid:
            expanded.append(key)
        else:
            raise SystemExit(f"error: unknown dataset {key!r}; valid keys are {sorted(valid | set(ALIASES))}")
    by_key = {item.key: item for item in DATASETS}
    result = []
    seen = set()
    for key in expanded:
        if key not in seen:
            result.append(by_key[key])
            seen.add(key)
    return result


def _command_url(command: str) -> str:
    marker = "curl -L '"
    if marker not in command:
        raise RuntimeError(f"cannot extract URL from command: {command}")
    return command.split(marker, 1)[1].split("'", 1)[0]


def _resolve_output(output_dir: Path, output_hint: str) -> Path:
    if output_hint.startswith(("s3://", "http://", "https://", "user-selected", "Bioconductor")):
        return output_dir / "external_or_manual_download"
    path = Path(output_hint)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return path
    return output_dir / path.name


def _download(url: str, output_path: Path) -> str:
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    print(f"# downloaded: {output_path}")
    return "downloaded"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"datasets": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
