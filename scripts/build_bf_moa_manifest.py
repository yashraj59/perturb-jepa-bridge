from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb_jepa.data.images import bf_moa_table_to_manifest, extract_bf_moa_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized BF-MoA image manifest.")
    parser.add_argument("--data-tables", required=True, help="Path to BF-MoA data_tables.tar.gz.")
    parser.add_argument("--output", required=True, help="Output CSV manifest.")
    parser.add_argument("--image-root", default="", help="Root directory containing extracted plate images.")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        bf_csv = extract_bf_moa_metadata(args.data_tables, temp_dir)
        table = pd.read_csv(bf_csv)
        manifest = bf_moa_table_to_manifest(table, image_root=args.image_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Wrote {len(manifest):,} manifest rows to {output}")


if __name__ == "__main__":
    main()
