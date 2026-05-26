import pandas as pd
import pytest

from perturb_jepa.data.scgenescope import (
    audit_scgenescope_local_root,
    build_scgenescope_condition_pairs,
    classify_scgenescope_remote_path,
    scgenescope_croissant_dry_run_manifest,
    scgenescope_feature_pair_candidates,
    scgenescope_manifest_template,
    scgenescope_split_from_replicate,
    summarize_scgenescope_remote_entries,
    validate_scgenescope_croissant_contract,
    validate_scgenescope_manifest,
)


def test_scgenescope_manifest_template_builds_condition_pair():
    manifest = scgenescope_manifest_template()

    pairs = build_scgenescope_condition_pairs(manifest)

    assert pairs.shape[0] == 1
    assert bool(pairs.loc[0, "condition_pair_ready"])
    assert pairs.loc[0, "rna_record_count"] == 1
    assert pairs.loc[0, "image_record_count"] == 1
    assert "dose" in manifest.columns


def test_scgenescope_manifest_rejects_shortcut_columns():
    manifest = scgenescope_manifest_template()
    manifest["condition_key"] = "control::round_1"

    with pytest.raises(ValueError, match="forbidden shortcut"):
        validate_scgenescope_manifest(manifest)


def test_scgenescope_manifest_derives_split_and_fixed_dose_from_croissant_aliases():
    frame = pd.DataFrame(
        [
            {
                "modality": "rna",
                "Treatment": "drug_a",
                "round": "round_1",
                "Batch": "batch_1",
                "Replicate": "3",
                "uri": "rna_round_1.h5ad",
            }
        ]
    )

    manifest = validate_scgenescope_manifest(frame)

    assert manifest.loc[0, "treatment_id"] == "drug_a"
    assert manifest.loc[0, "dose"] == "fixed"
    assert manifest.loc[0, "split"] == "train"
    assert manifest.loc[0, "batch"] == "batch_1"


def test_scgenescope_split_mapping_matches_supplement_contract():
    assert scgenescope_split_from_replicate("3") == "train"
    assert scgenescope_split_from_replicate("5") == "validation"
    assert scgenescope_split_from_replicate("4") == "test"
    assert scgenescope_split_from_replicate("1") == "alternate_test"
    assert scgenescope_split_from_replicate("2") == "alternate_test"


def test_scgenescope_audit_reports_missing_local_root(tmp_path):
    audit = audit_scgenescope_local_root(tmp_path / "missing_scgenescope")

    assert not audit.root_exists
    assert not audit.manifest_valid
    assert audit.paired_condition_count == 0


def test_scgenescope_audit_validates_local_manifest(tmp_path):
    root = tmp_path / "scgenescope"
    root.mkdir()
    manifest = pd.concat(
        [
            scgenescope_manifest_template(),
            pd.DataFrame(
                [
                    {
                        "modality": "rna",
                        "treatment_id": "drug_a",
                        "dose": "1",
                        "round": "round_2",
                        "batch": "batch_2",
                        "replicate": "replicate_1",
                        "split": "test",
                        "uri": "rna_drug_a.h5ad",
                    },
                    {
                        "modality": "image",
                        "treatment_id": "drug_a",
                        "dose": "1",
                        "round": "round_2",
                        "batch": "batch_2",
                        "replicate": "replicate_1",
                        "split": "test",
                        "uri": "image_drug_a.parquet",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    manifest.to_csv(root / "scgenescope_manifest.tsv", sep="\t", index=False)

    audit = audit_scgenescope_local_root(root)

    assert audit.root_exists
    assert audit.manifest_valid
    assert audit.paired_condition_count == 2
    assert audit.rna_record_count == 2
    assert audit.image_record_count == 2
    assert audit.split_count == 2


def test_scgenescope_remote_path_classifier_identifies_large_feature_payloads():
    row = classify_scgenescope_remote_path(
        "features/rnaseq/scvi/n200/round_1.h5ad",
        size=4_366_952_116,
        entry_type="file",
    )

    assert row["modality"] == "rna"
    assert row["representation"] == "feature"
    assert row["round"] == "round_1"
    assert row["is_feature_h5ad"]
    assert row["is_large_payload"]
    assert not row["is_light_metadata_candidate"]


def test_scgenescope_remote_summary_flags_light_metadata_candidates():
    frame = summarize_scgenescope_remote_entries(
        [
            {"type": "file", "path": "metadata/treatments.csv", "size": 1024},
            {"type": "file", "path": "measured/rnaseq/round_1.h5ad", "size": 48_101_886_092},
        ]
    )

    metadata_row = frame[frame["path"] == "metadata/treatments.csv"].iloc[0]
    measured_row = frame[frame["path"] == "measured/rnaseq/round_1.h5ad"].iloc[0]

    assert bool(metadata_row["is_light_metadata_candidate"])
    assert bool(measured_row["is_large_payload"])


def test_scgenescope_feature_pair_candidates_pair_feature_h5ads_by_round():
    frame = summarize_scgenescope_remote_entries(
        [
            {"type": "file", "path": "features/rnaseq/scvi/n200/round_1.h5ad", "size": 4_000},
            {"type": "file", "path": "features/rnaseq/scvi/n200/round_2.h5ad", "size": 2_000},
            {"type": "file", "path": "features/imaging/imagenet/vit-l/round_1.h5ad", "size": 8_000},
            {"type": "file", "path": "features/imaging/imagenet/vit-l/round_2.h5ad", "size": 3_000},
            {"type": "file", "path": "measured/imaging/round_2.h5", "size": 1_000},
        ]
    )

    pairs = scgenescope_feature_pair_candidates(frame)

    assert pairs.shape[0] == 2
    assert pairs.loc[0, "round"] == "round_2"
    assert pairs.loc[0, "paired_size_bytes"] == 5_000
    assert "measured/imaging/round_2.h5" not in set(pairs["image_path"])


def test_scgenescope_croissant_contract_and_dry_run_manifest_build_pairs():
    fields = pd.DataFrame(
        [
            {"field": "cell_id"},
            {"field": "Treatment"},
            {"field": "Replicate"},
            {"field": "batch"},
            {"field": "Group"},
        ]
    )
    split_contract = {
        "replicate_to_split": {
            "1": "alternate_test",
            "2": "alternate_test",
            "3": "train",
            "4": "test",
            "5": "validation",
        },
        "target_column": "Treatment",
        "control_column": "Group",
        "batch_column": "batch",
    }

    validation = validate_scgenescope_croissant_contract(fields, split_contract)
    manifest = scgenescope_croissant_dry_run_manifest(split_contract)
    pairs = build_scgenescope_condition_pairs(manifest)

    assert validation["adapter_contract_valid"]
    assert manifest["split"].nunique() == 4
    assert manifest.shape[0] == 40
    assert pairs["condition_pair_ready"].all()
    assert pairs.shape[0] == 16
