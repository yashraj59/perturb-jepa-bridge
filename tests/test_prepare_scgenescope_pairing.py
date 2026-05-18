import json

import numpy as np
import pandas as pd

from scripts.prepare_scgenescope_pairing import main as prepare_scgenescope_main


def test_prepare_scgenescope_pairing_writes_condition_level_outputs(tmp_path):
    ad = pytest_import_anndata()
    raw = tmp_path / "raw"
    rna_round1 = raw / "rna_round1.h5ad"
    rna_round2 = raw / "rna_round2.h5ad"
    image_round1 = raw / "image_round1.h5ad"
    image_round2 = raw / "image_round2.h5ad"
    _write_embedding_h5ad(rna_round1, round_id=1, use_obsm=True)
    _write_embedding_h5ad(rna_round2, round_id=2, use_obsm=True)
    _write_embedding_h5ad(image_round1, round_id=1, use_obsm=False)
    _write_embedding_h5ad(image_round2, round_id=2, use_obsm=False)

    rna_output = tmp_path / "processed/rna.h5ad"
    image_output = tmp_path / "processed/image_manifest.csv"
    array_dir = tmp_path / "processed/image_arrays"
    summary_output = tmp_path / "metrics/overlap.csv"
    json_output = tmp_path / "metrics/summary.json"

    exit_code = prepare_scgenescope_main(
        [
            "--rna-round1",
            str(rna_round1),
            "--rna-round2",
            str(rna_round2),
            "--image-round1",
            str(image_round1),
            "--image-round2",
            str(image_round2),
            "--rna-output",
            str(rna_output),
            "--image-output",
            str(image_output),
            "--image-array-dir",
            str(array_dir),
            "--summary-output",
            str(summary_output),
            "--json-output",
            str(json_output),
            "--image-size",
            "3",
            "--max-profiles-per-condition-split",
            "2",
            "--min-shared-treatments",
            "2",
        ]
    )

    assert exit_code == 0
    rna = ad.read_h5ad(rna_output)
    manifest = pd.read_csv(image_output)
    summary = pd.read_csv(summary_output)

    assert rna.X.shape[1] == 5
    assert set(rna.obs["split"]) == {"train", "val", "we_test", "he_test"}
    assert set(manifest["split"]) == {"train", "val", "we_test", "he_test"}
    assert set(rna.obs["condition_key_scgenescope"]) == {"Phenacetin", "PQ401", "DMSO"}
    assert set(manifest["condition_key_scgenescope"]) == {"Phenacetin", "PQ401", "DMSO"}
    assert (rna.obs["condition_key"] == rna.obs["perturbation"]).all()
    assert (manifest["condition_key"] == manifest["perturbation"]).all()
    assert not manifest["condition_key"].astype(str).str.contains("replicate|batch|well", case=False).any()
    assert (manifest["perturbation"].eq("DMSO") == manifest["perturbation_type"].eq("control")).all()
    assert summary.groupby(["split", "perturbation"])[["n_rna", "n_image"]].sum().min().min() > 0

    first_array = np.load(manifest.iloc[0]["image_path"])
    assert first_array.shape == (1, 3, 3)


def test_prepare_scgenescope_pairing_can_anchor_round2_and_control_center(tmp_path):
    ad = pytest_import_anndata()
    raw = tmp_path / "raw"
    rna_round1 = raw / "rna_round1.h5ad"
    rna_round2 = raw / "rna_round2.h5ad"
    image_round1 = raw / "image_round1.h5ad"
    image_round2 = raw / "image_round2.h5ad"
    _write_embedding_h5ad(rna_round1, round_id=1, use_obsm=True)
    _write_embedding_h5ad(rna_round2, round_id=2, use_obsm=True)
    _write_embedding_h5ad(image_round1, round_id=1, use_obsm=False)
    _write_embedding_h5ad(image_round2, round_id=2, use_obsm=False)

    rna_output = tmp_path / "processed/rna_centered.h5ad"
    image_output = tmp_path / "processed/image_manifest_centered.csv"
    json_output = tmp_path / "metrics/summary_centered.json"

    exit_code = prepare_scgenescope_main(
        [
            "--rna-round1",
            str(rna_round1),
            "--rna-round2",
            str(rna_round2),
            "--image-round1",
            str(image_round1),
            "--image-round2",
            str(image_round2),
            "--rna-output",
            str(rna_output),
            "--image-output",
            str(image_output),
            "--image-array-dir",
            str(tmp_path / "processed/image_arrays_centered"),
            "--summary-output",
            str(tmp_path / "metrics/overlap_centered.csv"),
            "--json-output",
            str(json_output),
            "--image-size",
            "3",
            "--max-profiles-per-condition-split",
            "2",
            "--min-shared-treatments",
            "2",
            "--split-policy",
            "round2_anchor",
            "--control-center",
        ]
    )

    assert exit_code == 0
    rna = ad.read_h5ad(rna_output)
    manifest = pd.read_csv(image_output)
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert set(rna.obs["split"]) == {"train", "we_test", "he_test"}
    assert set(manifest["split"]) == {"train", "we_test", "he_test"}
    assert payload["split_policy"] == "round2_anchor"
    assert payload["control_centering"]["enabled"] is True
    assert payload["control_centering"]["rna"]["groups_centered"] > 0


def pytest_import_anndata():
    import anndata as ad

    return ad


def _write_embedding_h5ad(path, *, round_id: int, use_obsm: bool) -> None:
    ad = pytest_import_anndata()
    replicates = ["3", "5", "4"] if round_id == 1 else ["1", "2"]
    treatments = ["Phenacetin", "PQ401", "DMSO"]
    rows = []
    values = []
    for replicate in replicates:
        for treatment in treatments:
            for repeat in range(3):
                rows.append(
                    {
                        "Treatment": treatment,
                        "Replicate": replicate,
                        "batch": f"b{replicate}",
                        "Plate": f"p{replicate}",
                        "Well": f"A{repeat + 1:02d}",
                        "Seen": True,
                    }
                )
                base = float(len(values))
                width = 5 if use_obsm else 6
                values.append(np.arange(base, base + width, dtype=np.float32))
    obs = pd.DataFrame(rows)
    obs.index = [f"cell_{round_id}_{index}" for index in range(len(obs))]
    matrix = np.vstack(values).astype(np.float32)
    if use_obsm:
        var = pd.DataFrame(index=["placeholder"])
        adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs, var=var)
        adata.obsm["scvi_n200"] = matrix
    else:
        var = pd.DataFrame(index=[f"feat_{index}" for index in range(matrix.shape[1])])
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
