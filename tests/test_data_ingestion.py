import numpy as np
import pandas as pd
import pytest
import torch

from perturb_jepa.data.conditions import (
    MetadataVocab,
    build_condition_bags,
    compute_condition_prototypes,
    prototype_lookup_indices,
)
from perturb_jepa.data.images import (
    ImageManifestCollator,
    ImageManifestDataset,
    ImageManifestExample,
    image_array_to_chw_float,
)
from perturb_jepa.data.scrna import SCRNATokenCollator, SCRNATokenDataset, library_size_log1p


def _obs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "perturbation": ["drugA", "drugA", "drugB"],
            "perturbation_type": ["compound", "compound", "compound"],
            "dose": ["10uM", "10uM", "5uM"],
            "time": ["48h", "48h", "24h"],
            "cell_line": ["U2OS", "U2OS", "A549"],
            "batch": ["batch1", "batch1", "batch2"],
        },
        index=["cell0", "cell1", "cell2"],
    )


def _var_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id": ["g0", "g1", "g2", "g3"],
            "gene_symbol": ["G0", "G1", "G2", "G3"],
        }
    )


def _manifest(paths: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": paths,
            "plate": ["p1", "p1"],
            "well": ["A01", "A01"],
            "site": ["1", "2"],
            "channel_or_z": ["BF1", "BF1"],
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


def test_scrna_token_dataset_normalizes_and_collates_metadata():
    matrix = np.array(
        [
            [0.0, 2.0, 4.0, 8.0],
            [1.0, 1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    dataset = SCRNATokenDataset(matrix, _obs_frame(), _var_frame(), gene_indices=[3, 1, 0])

    assert len(dataset) == 3
    assert dataset.obs.loc["cell0", "condition_key"] == "drugA|compound|10uM|48h|U2OS"
    assert dataset.gene_vocab.token_to_idx["G3"] == 3
    np.testing.assert_allclose(dataset[0].expression_values, library_size_log1p(matrix)[0, [3, 1, 0]])

    batch = SCRNATokenCollator(mask_prob=1.0)([dataset[0], dataset[2]])
    assert batch.gene_ids.shape == (2, 3)
    assert batch.expression_values.dtype == torch.float32
    assert batch.rna_token_mask.all()
    assert batch.condition_key == ["drugA|compound|10uM|48h|U2OS", "drugB|compound|5uM|24h|A549"]
    assert batch.obs_index == ["cell0", "cell2"]
    assert batch.dose.tolist() == [10.0, 5.0]
    assert batch.time.tolist() == [48.0, 24.0]


def test_scrna_dataset_validates_shapes_and_gene_indices():
    matrix = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="obs rows"):
        SCRNATokenDataset(matrix, _obs_frame(), pd.DataFrame(index=["g0", "g1", "g2"]))

    obs = _obs_frame().iloc[:2].copy()
    with pytest.raises(ValueError, match="out-of-bounds"):
        SCRNATokenDataset(matrix, obs, pd.DataFrame(index=["g0", "g1", "g2"]), gene_indices=[0, 3])


def test_image_manifest_dataset_loads_numpy_images_and_collates_patch_masks(tmp_path):
    first = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    second = np.ones((4, 4, 3), dtype=np.uint8) * 255
    np.save(tmp_path / "first.npy", first)
    np.save(tmp_path / "second.npy", second)

    dataset = ImageManifestDataset(_manifest(["first.npy", "second.npy"]), image_root=tmp_path, channels=3)
    assert len(dataset) == 2
    assert dataset[0].image.shape == (3, 4, 4)
    np.testing.assert_allclose(dataset[0].image[:, 0, 0], first[0, 0, :] / 255.0)

    batch = ImageManifestCollator(patch_size=2, patch_mask_prob=1.0)([dataset[0], dataset[1]])
    assert batch.images.shape == (2, 3, 4, 4)
    assert batch.image_patch_mask.shape == (2, 4)
    assert batch.image_patch_mask.all()
    assert batch.condition_key == ["drugA|compound|10uM|48h|U2OS", "drugB|compound|5uM|24h|A549"]
    assert batch.dose.tolist() == [10.0, 5.0]


def test_image_helpers_validate_shapes_and_channel_layouts():
    chw = np.ones((3, 4, 4), dtype=np.float32)
    assert image_array_to_chw_float(chw).shape == (3, 4, 4)

    with pytest.raises(ValueError, match="same shape"):
        ImageManifestCollator()(
            [
                ImageManifestExample(
                    image=np.ones((1, 4, 4), dtype=np.float32),
                    image_path="a.npy",
                    condition_key="a",
                    perturbation_id=0,
                    perturbation_type_id=0,
                    cell_line_id=0,
                    batch_id=0,
                    dose=0.0,
                    time=0.0,
                ),
                ImageManifestExample(
                    image=np.ones((1, 8, 8), dtype=np.float32),
                    image_path="b.npy",
                    condition_key="b",
                    perturbation_id=0,
                    perturbation_type_id=0,
                    cell_line_id=0,
                    batch_id=0,
                    dose=0.0,
                    time=0.0,
                ),
            ]
        )


def test_shared_metadata_vocab_condition_bags_and_prototypes(tmp_path):
    matrix = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    obs = _obs_frame()
    var = pd.DataFrame({"gene_id": ["g0", "g1"], "gene_symbol": ["G0", "G1"]})
    np.save(tmp_path / "image.npy", np.ones((4, 4, 1), dtype=np.float32))
    manifest = _manifest(["image.npy", "image.npy"])
    vocab = MetadataVocab.from_frames([obs, manifest])

    rna = SCRNATokenDataset(matrix, obs, var, metadata_vocab=vocab)
    images = ImageManifestDataset(manifest, image_root=tmp_path, channels=1, metadata_vocab=vocab)
    assert rna[0].perturbation_id == images[0].perturbation_id

    bags = build_condition_bags(rna.obs)
    assert bags.keys == ["drugA|compound|10uM|48h|U2OS", "drugB|compound|5uM|24h|A549"]
    np.testing.assert_array_equal(bags["drugA|compound|10uM|48h|U2OS"], np.array([0, 1]))

    prototypes = compute_condition_prototypes(
        matrix,
        rna.obs["condition_key"].tolist(),
    )
    assert prototypes.counts.tolist() == [2, 1]
    np.testing.assert_allclose(prototypes.lookup(["drugA|compound|10uM|48h|U2OS"]), [[2.0, 0.0]])
    np.testing.assert_array_equal(
        prototype_lookup_indices(rna.obs["condition_key"], prototypes.keys),
        np.array([0, 0, 1]),
    )
