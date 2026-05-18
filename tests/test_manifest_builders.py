import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.build_image_manifest import build_image_manifest
from scripts.build_paired_manifest import build_paired_manifest
from scripts.build_rna_manifest import build_rna_manifest


def _args(**kwargs):
    return type("Args", (), kwargs)()


def _image_manifest(paths: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": paths,
            "plate": ["plate1", "plate2"],
            "well": ["A01", "B01"],
            "site": ["1", "1"],
            "channel_or_z": ["BF", "BF"],
            "perturbation": ["drugA", "drugB"],
            "perturbation_type": ["compound", "compound"],
            "compound": ["drugA", "drugB"],
            "moa": ["m1", "m2"],
            "target_gene": ["", ""],
            "dose": ["10uM", "5uM"],
            "time": ["48h", "24h"],
            "cell_line": ["U2OS", "A549"],
            "batch": ["batch1", "batch2"],
        }
    )


@pytest.mark.parametrize(
    "script",
    [
        "scripts/build_rna_manifest.py",
        "scripts/build_image_manifest.py",
        "scripts/build_paired_manifest.py",
    ],
)
def test_manifest_builder_clis_have_help(script):
    result = subprocess.run([sys.executable, script, "--help"], check=False, capture_output=True, text=True)

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_image_manifest_normalizes_paths_and_allows_missing(tmp_path):
    np.save(tmp_path / "present.npy", np.zeros((2, 2), dtype=np.float32))
    input_csv = tmp_path / "raw_images.csv"
    output_csv = tmp_path / "images.csv"
    _image_manifest(["present.npy", "missing.npy"]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="do not exist"):
        build_image_manifest(
            _args(
                input_manifest=input_csv,
                output_manifest=output_csv,
                image_root=tmp_path,
                allow_missing_images=False,
                qc_json=None,
                qc_csv=None,
            )
        )

    qc = build_image_manifest(
        _args(
            input_manifest=input_csv,
            output_manifest=output_csv,
            image_root=tmp_path,
            allow_missing_images=True,
            qc_json=None,
            qc_csv=None,
        )
    )
    manifest = pd.read_csv(output_csv)

    assert qc["n_missing_images"] == 1
    assert manifest["condition_key"].tolist() == ["drugA|10uM|48h|U2OS", "drugB|5uM|24h|A549"]
    assert "batch1" not in manifest.loc[0, "condition_key"]
    assert "plate1" not in manifest.loc[0, "condition_key"]
    assert manifest["image_exists"].tolist() == [True, False]
    assert output_csv.with_name("images_qc_summary.json").exists()
    assert output_csv.with_name("images_condition_qc.csv").exists()


def test_paired_manifest_pairs_only_four_field_biological_key(tmp_path):
    rna_csv = tmp_path / "rna_metadata.csv"
    image_csv = tmp_path / "image_manifest.csv"
    paired_csv = tmp_path / "paired.csv"
    pd.DataFrame(
        {
            "obs_index": ["cell0", "cell1", "cell2"],
            "perturbation": ["drugA", "drugA", "rnaOnly"],
            "dose": ["10uM", "10uM", "1uM"],
            "time": ["48h", "48h", "24h"],
            "cell_line": ["U2OS", "U2OS", "U2OS"],
            "batch": ["rnaBatch1", "rnaBatch2", "rnaBatch3"],
        }
    ).to_csv(rna_csv, index=False)
    _image_manifest(["a.npy", "b.npy"]).to_csv(image_csv, index=False)

    qc = build_paired_manifest(
        _args(rna_metadata=rna_csv, image_manifest=image_csv, output_manifest=paired_csv, qc_json=None, qc_csv=None)
    )
    paired = pd.read_csv(paired_csv)

    assert qc["n_paired_conditions"] == 1
    assert paired.loc[0, "pairing_tier"] == "tier2_weakly_paired_sample_well_condition"
    assert paired.loc[0, "pairing_key"] == "drugA|10uM|48h|U2OS"
    assert paired.loc[0, "n_rna_cells"] == 2
    assert paired.loc[0, "n_images"] == 1
    assert "rnaBatch" not in paired.loc[0, "condition_key"]
    assert "plate" not in paired.loc[0, "condition_key"]
    assert paired_csv.with_name("paired_pairing_qc.csv").exists()


def test_paired_manifest_uses_explicit_spot_pairing_key(tmp_path):
    rna_csv = tmp_path / "rna_metadata.csv"
    image_csv = tmp_path / "image_manifest.csv"
    paired_csv = tmp_path / "paired.csv"
    pd.DataFrame(
        {
            "obs_index": ["spot0", "spot1"],
            "sample_id": ["sampleA", "sampleA"],
            "spot_id": ["s0", "s1"],
            "perturbation": ["drugA", "drugB"],
            "dose": ["10uM", "5uM"],
            "time": ["48h", "24h"],
            "cell_line": ["tissue", "tissue"],
            "batch": ["section1", "section1"],
        }
    ).to_csv(rna_csv, index=False)
    image = _image_manifest(["a.npy", "b.npy"])
    image["sample_id"] = ["sampleA", "sampleA"]
    image["spot_id"] = ["s0", "s1"]
    image["cell_line"] = ["tissue", "tissue"]
    image.to_csv(image_csv, index=False)

    qc = build_paired_manifest(
        _args(
            rna_metadata=rna_csv,
            image_manifest=image_csv,
            output_manifest=paired_csv,
            output_dir=None,
            output_rna_metadata=None,
            output_image_manifest=None,
            pairing_table=None,
            pairing_key_type="auto",
            pairing_tier=None,
            qc_json=None,
            qc_csv=None,
        )
    )
    pairing_table = pd.read_csv(paired_csv.with_name("pairing_table.csv"))

    assert qc["pairing_key_type"] == "spot"
    assert qc["pairing_key_columns"] == ["sample_id", "spot_id"]
    assert set(pairing_table["pairing_id"]) == {"sampleA|s0", "sampleA|s1"}
    assert set(pairing_table["pairing_tier"]) == {"tier1_true_paired_image_expression"}


def test_paired_manifest_rejects_technical_condition_key_leakage(tmp_path):
    rna_csv = tmp_path / "rna_metadata.csv"
    image_csv = tmp_path / "image_manifest.csv"
    pd.DataFrame(
        {
            "perturbation": ["drugA"],
            "dose": ["10uM"],
            "time": ["48h"],
            "cell_line": ["U2OS"],
            "batch": ["batch1"],
            "condition_key": ["drugA|10uM|48h|U2OS|batch1"],
        }
    ).to_csv(rna_csv, index=False)
    _image_manifest(["a.npy", "b.npy"]).to_csv(image_csv, index=False)

    with pytest.raises(ValueError, match="condition_key values"):
        build_paired_manifest(
            _args(
                rna_metadata=rna_csv,
                image_manifest=image_csv,
                output_manifest=tmp_path / "paired.csv",
                qc_json=None,
                qc_csv=None,
            )
        )


def test_rna_manifest_normalizes_h5ad_and_writes_condition_bags(tmp_path):
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("pyarrow")

    adata = anndata.AnnData(
        X=np.array([[1, 0], [3, 1], [0, 5]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "perturbation": ["drugA", "drugA", "drugB"],
                "perturbation_type": ["compound", "compound", "compound"],
                "dose": ["10uM", "10uM", "5uM"],
                "time": ["48h", "48h", "24h"],
                "cell_line": ["U2OS", "U2OS", "A549"],
                "batch": ["batch1", "batch2", "batch3"],
                "plate": ["plate1", "plate1", "plate2"],
            },
            index=["cell0", "cell1", "cell2"],
        ),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    input_h5ad = tmp_path / "input.h5ad"
    adata.write_h5ad(input_h5ad)

    qc = build_rna_manifest(
        _args(
            input_h5ad=input_h5ad,
            output_dir=tmp_path / "rna",
            output_h5ad=None,
            metadata_csv=None,
            condition_bags_parquet=None,
            qc_json=None,
            qc_csv=None,
        )
    )
    metadata = pd.read_csv(tmp_path / "rna" / "rna_metadata.csv")
    bags = pd.read_parquet(tmp_path / "rna" / "rna_condition_bags.parquet")
    qc_json = json.loads((tmp_path / "rna" / "rna_qc_summary.json").read_text())

    assert qc["n_cells"] == 3
    assert qc_json["condition_key_columns"] == ["perturbation", "dose", "time", "cell_line"]
    assert metadata.loc[0, "condition_key"] == "drugA|10uM|48h|U2OS"
    assert "batch1" not in metadata.loc[0, "condition_key"]
    assert "plate1" not in metadata.loc[0, "condition_key"]
    assert bags.set_index("condition_key").loc["drugA|10uM|48h|U2OS", "n_cells"] == 2
