from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class DownloadItem:
    dataset: str
    url: str
    output: str
    notes: str


DOWNLOADS = (
    DownloadItem(
        "norman-weissman",
        "https://zenodo.org/records/13350497/files/NormanWeissman2019_filtered.h5ad?download=1",
        "NormanWeissman2019_filtered.h5ad",
        "scPerturb harmonized combinatorial CRISPRa benchmark",
    ),
    DownloadItem(
        "replogle-k562-essential",
        "https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "scPerturb harmonized CRISPRi essential-gene screen",
    ),
    DownloadItem(
        "srivatsan-sciplex3",
        "https://zenodo.org/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1",
        "SrivatsanTrapnell2020_sciplex3.h5ad",
        "scPerturb harmonized chemical perturbation dataset",
    ),
    DownloadItem(
        "bf-moa-metadata",
        "https://ndownloader.figshare.com/files/37984380",
        "bf_moa_data_tables.tar.gz",
        "BF-MoA metadata archive, small enough to fetch before raw images",
    ),
    DownloadItem(
        "bf-moa-manifest",
        "https://api.figshare.com/v2/articles/21378906",
        "bf_moa_figshare_manifest.json",
        "BF-MoA Figshare API manifest for plate archive URLs",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public Perturb-JEPA data assets.")
    parser.add_argument("--dataset", action="append", help="Dataset key to download. May be repeated.")
    parser.add_argument("--all", action="store_true", help="Download every configured item.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--output-dir", default="data/raw", help="Destination directory.")
    args = parser.parse_args()

    selected = {item.dataset for item in DOWNLOADS} if args.all else set(args.dataset or [])
    if not selected:
        parser.error("pass --dataset at least once or use --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in DOWNLOADS:
        if item.dataset not in selected:
            continue
        output = output_dir / item.output
        command = ["curl", "-L", item.url, "-o", str(output)]
        print(f"# {item.dataset}: {item.notes}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
