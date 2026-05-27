from __future__ import annotations

from io import BytesIO
import tarfile

import h5py
import numpy as np

from scripts.run_perturbmulti_f103_preflight import (
    normalize_cell_id,
    parse_tar_members_from_prefix,
    summarize_h5ad_obs,
)


def test_normalize_cell_id_strips_paths_and_image_suffixes():
    assert normalize_cell_id("plate/a/b/12345.ome.tif") == "12345"
    assert normalize_cell_id("678.0") == "678"
    assert normalize_cell_id("cell.png?download=true") == "cell"


def test_parse_tar_members_from_prefix_extracts_member_ids():
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        data = b"abc"
        info = tarfile.TarInfo("images/101.ome.tif")
        info.size = len(data)
        archive.addfile(info, BytesIO(data))
        info = tarfile.TarInfo("images/102.png")
        info.size = len(data)
        archive.addfile(info, BytesIO(data))

    rows = parse_tar_members_from_prefix(payload.getvalue(), tar_file="chunk.tar")

    assert [row["normalized_image_id"] for row in rows] == ["101", "102"]
    assert rows[0]["size_bytes"] == 3


def test_summarize_h5ad_obs_reads_required_obs_without_x_payload(tmp_path):
    path = tmp_path / "mini.h5ad"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", shape=(2, 3), dtype="float32")
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.array(["g1", "g2", "g3"], dtype=object), dtype=string_dtype)
        obs = handle.create_group("obs")
        columns = np.array(["cell_id", "cell_name", "condition", "perturbation", "fov", "x", "y", "z", "dataset"], dtype=object)
        obs.attrs["_index"] = "_index"
        obs.attrs["column-order"] = columns
        obs.create_dataset("_index", data=np.array(["row0", "row1"], dtype=object), dtype=string_dtype)
        obs.create_dataset("cell_id", data=np.array([101, 102]))
        obs.create_dataset("cell_name", data=np.array(["cellA", "cellB"], dtype=object), dtype=string_dtype)
        obs.create_dataset("condition", data=np.array(["adlib", "adlib"], dtype=object), dtype=string_dtype)
        obs.create_dataset("perturbation", data=np.array(["gene1", "gene2"], dtype=object), dtype=string_dtype)
        obs.create_dataset("fov", data=np.array([1, 1]))
        obs.create_dataset("x", data=np.array([10.0, 11.0]))
        obs.create_dataset("y", data=np.array([20.0, 21.0]))
        obs.create_dataset("z", data=np.array([0.0, 0.0]))
        obs.create_dataset("dataset", data=np.array(["d1", "d1"], dtype=object), dtype=string_dtype)

    summary, values = summarize_h5ad_obs(path, label="rna")

    assert summary["required_obs_columns_present"] is True
    assert summary["n_obs"] == 2
    assert summary["n_vars"] == 3
    assert summary["perturbation_unique"] == 2
    assert values["cell_id"].tolist() == [101, 102]
