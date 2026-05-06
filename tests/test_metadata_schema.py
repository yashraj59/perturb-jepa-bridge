import pandas as pd
import pytest

from perturb_jepa.data.schema import (
    MetadataSchema,
    make_bio_key,
    make_condition_id,
    make_tech_key,
    validate_metadata_columns,
)


def test_metadata_schema_declares_biological_and_technical_columns():
    schema = MetadataSchema()

    assert schema.biological_keys == ("perturbation", "dose", "time", "cell_line")
    assert "moa" in schema.optional_biological_keys
    assert "batch" in schema.technical_keys
    assert "plate" in schema.technical_keys


def test_bio_key_and_condition_id_exclude_technical_metadata():
    row = {
        "perturbation": "drugA",
        "dose": "10uM",
        "time": "48h",
        "cell_line": "U2OS",
        "perturbation_type": "compound",
        "moa": "kinase",
        "batch": "batch1",
        "plate": "plate7",
        "run": "run3",
        "well": "A01",
        "site": "2",
        "z_plane": "0",
        "sequencing_lane": "lane1",
        "library_id": "lib9",
    }

    assert make_bio_key(row) == ("drugA", "10uM", "48h", "U2OS")
    assert make_condition_id(row) == "drugA|10uM|48h|U2OS"
    for technical_value in ("batch1", "plate7", "run3", "A01", "2", "0", "lane1", "lib9"):
        assert technical_value not in make_bio_key(row)
    for technical_value in ("batch1", "plate7", "run3", "A01", "lane1", "lib9"):
        assert technical_value not in make_condition_id(row)


def test_tech_key_uses_fixed_technical_order():
    row = {"batch": "b1", "plate": "p1", "well": "A01", "library_id": "lib1"}

    assert make_tech_key(row) == ("b1", "p1", "NA", "A01", "NA", "NA", "NA", "NA", "lib1")


def test_validate_metadata_columns_requires_core_biology():
    frame = pd.DataFrame({"perturbation": ["drugA"], "dose": ["1uM"], "time": ["24h"]})

    with pytest.raises(ValueError, match="cell_line"):
        validate_metadata_columns(frame)
