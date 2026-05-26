import itertools

from perturb_jepa.data.norman2019 import (
    assert_no_combo_order_leakage,
    canonical_condition_key,
    gears_simulation_split,
    is_combo,
    is_single,
    perturbation_genes,
)


def _toy_norman_conditions(gene_count: int = 10) -> list[str]:
    singles = [f"G{i}+ctrl" for i in range(gene_count)]
    combos = [f"G{i}+G{j}" for i, j in itertools.combinations(range(gene_count), 2)]
    return ["ctrl", *singles, *combos]


def test_condition_key_canonicalizes_order_and_control_suffixes():
    assert canonical_condition_key("A+B") == canonical_condition_key("B+A")
    assert canonical_condition_key("A+ctrl") == canonical_condition_key("A")
    assert canonical_condition_key("ctrl+A") == canonical_condition_key("A")
    assert canonical_condition_key("ctrl") == ("ctrl",)


def test_gears_simulation_split_has_no_order_alias_leakage():
    split = gears_simulation_split(_toy_norman_conditions(), seed=1)

    assert_no_combo_order_leakage(split)
    train_aliases = {canonical_condition_key(condition) for condition in split.train_conditions}
    for condition in split.test_conditions:
        assert canonical_condition_key(condition) not in train_aliases


def test_exact_train_combo_subset_requires_both_component_singles_in_train():
    conditions = _toy_norman_conditions()
    split = gears_simulation_split(conditions, seed=1)
    train_conditions = set(split.train_conditions)

    assert split.exact_train_combo_conditions
    for condition in split.exact_train_combo_conditions:
        assert is_combo(condition)
        for gene in perturbation_genes(condition):
            assert f"{gene}+ctrl" in train_conditions


def test_unseen_single_subset_contains_only_single_gene_conditions():
    split = gears_simulation_split(_toy_norman_conditions(), seed=1)

    assert split.unseen_single_conditions
    assert all(is_single(condition) for condition in split.unseen_single_conditions)
