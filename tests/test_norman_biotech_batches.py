import numpy as np

from perturb_jepa.data.norman2019 import NormanDataset, add_gears_simulation_split
from perturb_jepa.training.norman_biotech_batches import build_norman_biotech_spec, iter_norman_biotech_condition_batches


def _fake_norman_dataset() -> NormanDataset:
    conditions = ("ctrl", "A", "B", "C", "D", "A+B", "A+C", "A+D", "B+C", "C+D")
    genes = tuple(f"g{i}" for i in range(16))
    rng = np.random.default_rng(0)
    means = rng.normal(size=(len(conditions), len(genes))).astype(np.float32)
    counts = np.exp(means).astype(np.float32)
    return NormanDataset(
        h5ad_path="fake.h5ad",  # type: ignore[arg-type]
        gene_ids=genes,
        gene_names=genes,
        conditions=conditions,
        condition_full_names={condition: condition for condition in conditions},
        condition_means=means,
        count_means=counts,
        ctrl_condition="ctrl",
        top_non_zero_de_20={},
        top_non_dropout_de_20={},
    )


def test_norman_biotech_batches_use_gene_descriptor_not_batch_or_dose():
    dataset = add_gears_simulation_split(_fake_norman_dataset(), seed=0)
    spec = build_norman_biotech_spec(dataset, gene_count=8)
    batch = next(iter_norman_biotech_condition_batches(dataset, spec, split="train", batch_size=2, steps=1, seed=0))

    assert batch.control_images is None
    assert batch.target_images is None
    assert batch.descriptor is not None
    assert batch.descriptor.shape[1] == spec.descriptor_dim
    assert set(batch.batch_id.detach().cpu().tolist()) == {0}
    assert set(batch.dose.detach().cpu().tolist()) == {1.0}
