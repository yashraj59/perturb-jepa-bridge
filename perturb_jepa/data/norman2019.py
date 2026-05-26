from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_NORMAN_H5AD = Path("data/raw/gears_norman/norman/perturb_processed.h5ad")
DEFAULT_GEARS_NORMAN_URL = "https://dataverse.harvard.edu/api/access/datafile/6154020"
BIOLOGICAL_SPLIT_SEED = 1


@dataclass(frozen=True)
class NormanSplit:
    seed: int
    train_gene_set_size: float
    combo_seen2_train_frac: float
    train_conditions: tuple[str, ...]
    val_conditions: tuple[str, ...]
    test_conditions: tuple[str, ...]
    test_subgroups: dict[str, tuple[str, ...]]
    val_subgroups: dict[str, tuple[str, ...]]
    exact_train_combo_conditions: tuple[str, ...]
    unseen_single_conditions: tuple[str, ...]


@dataclass(frozen=True)
class NormanDataset:
    h5ad_path: Path
    gene_ids: tuple[str, ...]
    gene_names: tuple[str, ...]
    conditions: tuple[str, ...]
    condition_full_names: dict[str, str]
    condition_means: np.ndarray
    count_means: np.ndarray
    ctrl_condition: str
    top_non_zero_de_20: dict[str, tuple[str, ...]]
    top_non_dropout_de_20: dict[str, tuple[str, ...]]
    split: NormanSplit | None = None

    @property
    def condition_to_index(self) -> dict[str, int]:
        return {condition: index for index, condition in enumerate(self.conditions)}

    @property
    def ctrl_mean(self) -> np.ndarray:
        return self.condition_means[self.condition_to_index[self.ctrl_condition]]

    @property
    def ctrl_count_mean(self) -> np.ndarray:
        return self.count_means[self.condition_to_index[self.ctrl_condition]]

    def mean_for(self, condition: str) -> np.ndarray:
        return self.condition_means[self.condition_to_index[condition]]

    def delta_for(self, condition: str) -> np.ndarray:
        return self.mean_for(condition) - self.ctrl_mean

    def count_logfc_for(self, condition: str, *, pseudocount: float = 1.0) -> np.ndarray:
        counts = self.count_means[self.condition_to_index[condition]]
        return np.log2((counts + float(pseudocount)) / (self.ctrl_count_mean + float(pseudocount)))

    def perturbation_ids_for(self, condition: str) -> tuple[str, ...]:
        return perturbation_genes(condition)

    def perturbation_label_for(self, condition: str) -> str:
        count = len(self.perturbation_ids_for(condition))
        if count == 0:
            return "control"
        if count == 1:
            return "single"
        if count == 2:
            return "double"
        return f"{count}_perturbation"

    def de20_gene_ids(self, condition: str) -> tuple[str, ...]:
        full = self.condition_full_names.get(condition, condition)
        return self.top_non_zero_de_20.get(full, self.top_non_dropout_de_20.get(full, ()))

    def de20_indices(self, condition: str) -> np.ndarray:
        gene_to_index = {gene_id: index for index, gene_id in enumerate(self.gene_ids)}
        indices = [gene_to_index[gene_id] for gene_id in self.de20_gene_ids(condition) if gene_id in gene_to_index]
        if len(indices) >= 20:
            return np.asarray(indices[:20], dtype=np.int64)
        delta = np.abs(self.delta_for(condition))
        fallback = np.argsort(delta)[::-1]
        merged = list(dict.fromkeys(indices + [int(value) for value in fallback]).keys())[:20]
        return np.asarray(merged, dtype=np.int64)

    def with_split(self, split: NormanSplit) -> "NormanDataset":
        return replace(self, split=split)


def load_norman2019_condition_data(path: str | Path = DEFAULT_NORMAN_H5AD) -> NormanDataset:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Norman 2019 loading requires scanpy/anndata") from exc

    h5ad_path = Path(path)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Norman h5ad not found: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path, backed="r")
    required_obs = {"condition", "condition_name"}
    missing_obs = required_obs - set(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Norman h5ad is missing required obs columns: {sorted(missing_obs)}")
    if "gene_name" not in adata.var.columns:
        raise ValueError("Norman h5ad is missing var['gene_name']")

    conditions = tuple(str(value) for value in adata.obs["condition"].astype(str).unique())
    if "ctrl" not in conditions:
        raise ValueError("Norman h5ad must contain a 'ctrl' condition")
    gene_ids = tuple(str(value) for value in adata.var_names)
    gene_names = tuple(str(value) for value in adata.var["gene_name"].astype(str))
    condition_full_names = {
        str(condition): str(full)
        for condition, full in adata.obs[["condition", "condition_name"]].drop_duplicates().itertuples(index=False)
    }
    expression_means = []
    count_means = []
    counts_layer = adata.layers["counts"] if "counts" in adata.layers else None
    for condition in conditions:
        mask = np.asarray(adata.obs["condition"].astype(str) == condition)
        expression = adata[mask].X
        expression_array = expression.toarray() if hasattr(expression, "toarray") else np.asarray(expression)
        expression_means.append(np.asarray(expression_array, dtype=np.float32).mean(axis=0))
        if counts_layer is not None:
            count_values = counts_layer[mask]
            count_array = count_values.toarray() if hasattr(count_values, "toarray") else np.asarray(count_values)
            count_means.append(np.asarray(count_array, dtype=np.float32).mean(axis=0))
        else:
            count_means.append(np.full(len(gene_ids), np.nan, dtype=np.float32))

    return NormanDataset(
        h5ad_path=h5ad_path,
        gene_ids=gene_ids,
        gene_names=gene_names,
        conditions=conditions,
        condition_full_names=condition_full_names,
        condition_means=np.stack(expression_means).astype(np.float32),
        count_means=np.stack(count_means).astype(np.float32),
        ctrl_condition="ctrl",
        top_non_zero_de_20=_uns_gene_dict(adata.uns.get("top_non_zero_de_20", {})),
        top_non_dropout_de_20=_uns_gene_dict(adata.uns.get("top_non_dropout_de_20", {})),
    )


def add_gears_simulation_split(
    dataset: NormanDataset,
    *,
    seed: int = BIOLOGICAL_SPLIT_SEED,
    train_gene_set_size: float = 0.75,
    combo_seen2_train_frac: float = 0.75,
) -> NormanDataset:
    split = gears_simulation_split(
        dataset.conditions,
        seed=seed,
        train_gene_set_size=train_gene_set_size,
        combo_seen2_train_frac=combo_seen2_train_frac,
    )
    return dataset.with_split(split)


def gears_simulation_split(
    conditions: Iterable[str],
    *,
    seed: int = BIOLOGICAL_SPLIT_SEED,
    train_gene_set_size: float = 0.75,
    combo_seen2_train_frac: float = 0.75,
) -> NormanSplit:
    pert_list = [condition for condition in conditions if condition != "ctrl"]
    first_train, first_test, test_subgroup = _simulation_split_lists(
        pert_list,
        train_gene_set_size=train_gene_set_size,
        combo_seen2_train_frac=combo_seen2_train_frac,
        seed=seed,
    )
    train, val, val_subgroup = _simulation_split_lists(
        first_train,
        train_gene_set_size=0.9,
        combo_seen2_train_frac=0.9,
        seed=seed,
    )
    train = list(train) + ["ctrl"]
    train_set = set(train)
    exact_train_combo = tuple(
        condition
        for condition in test_subgroup["combo_seen2"]
        if all(single_condition_for_gene(gene, conditions) in train_set for gene in perturbation_genes(condition))
    )
    return NormanSplit(
        seed=int(seed),
        train_gene_set_size=float(train_gene_set_size),
        combo_seen2_train_frac=float(combo_seen2_train_frac),
        train_conditions=tuple(train),
        val_conditions=tuple(val),
        test_conditions=tuple(first_test),
        test_subgroups={key: tuple(value) for key, value in test_subgroup.items()},
        val_subgroups={key: tuple(value) for key, value in val_subgroup.items()},
        exact_train_combo_conditions=exact_train_combo,
        unseen_single_conditions=tuple(test_subgroup["unseen_single"]),
    )


def assert_no_combo_order_leakage(split: NormanSplit) -> None:
    train_aliases = {canonical_condition_key(condition) for condition in split.train_conditions}
    for condition in split.test_conditions:
        key = canonical_condition_key(condition)
        if key in train_aliases and condition != "ctrl":
            raise AssertionError(f"test condition leaks into train under perturbation-order alias: {condition}")


def perturbation_genes(condition: str) -> tuple[str, ...]:
    if condition == "ctrl":
        return ()
    parts = tuple(part for part in str(condition).split("+") if part and part != "ctrl")
    return parts


def is_single(condition: str) -> bool:
    return len(perturbation_genes(condition)) == 1


def is_combo(condition: str) -> bool:
    return len(perturbation_genes(condition)) == 2


def canonical_condition_key(condition: str) -> tuple[str, ...]:
    genes = perturbation_genes(condition)
    return tuple(sorted(genes)) if genes else ("ctrl",)


def single_condition_for_gene(gene: str, conditions: Iterable[str]) -> str | None:
    wanted = str(gene)
    for condition in conditions:
        if is_single(condition) and perturbation_genes(condition)[0] == wanted:
            return condition
    return None


def _simulation_split_lists(
    pert_list: list[str],
    *,
    train_gene_set_size: float,
    combo_seen2_train_frac: float,
    seed: int,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    unique_pert_genes = _genes_from_perts(pert_list)
    rng = np.random.RandomState(seed)
    train_gene_candidates = rng.choice(
        unique_pert_genes,
        int(len(unique_pert_genes) * train_gene_set_size),
        replace=False,
    )
    ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)
    pert_single_train = _perts_from_genes(train_gene_candidates, pert_list, "single")
    pert_combo = _perts_from_genes(train_gene_candidates, pert_list, "combo")
    pert_train: list[str] = list(pert_single_train)
    pert_test: list[str] = []
    combo_seen1 = [
        condition
        for condition in pert_combo
        if len([gene for gene in perturbation_genes(condition) if gene in train_gene_candidates]) == 1
    ]
    pert_test.extend(combo_seen1)
    pert_combo_remaining = np.setdiff1d(pert_combo, combo_seen1)
    rng = np.random.RandomState(seed)
    train_count = int(len(pert_combo_remaining) * combo_seen2_train_frac)
    pert_combo_train = rng.choice(pert_combo_remaining, train_count, replace=False).tolist() if train_count else []
    combo_seen2 = np.setdiff1d(pert_combo_remaining, pert_combo_train).tolist()
    pert_test.extend(combo_seen2)
    pert_train.extend(pert_combo_train)
    unseen_single = _perts_from_genes(ood_genes, pert_list, "single")
    combo_ood = _perts_from_genes(ood_genes, pert_list, "combo")
    pert_test.extend(unseen_single)
    combo_seen0 = [
        condition
        for condition in combo_ood
        if len([gene for gene in perturbation_genes(condition) if gene in train_gene_candidates]) == 0
    ]
    pert_test.extend(combo_seen0)
    expected = len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2)
    if expected != len(pert_list):
        raise AssertionError(f"GEARS simulation split accounting mismatch: expected {expected}, got {len(pert_list)}")
    return (
        pert_train,
        pert_test,
        {"combo_seen0": combo_seen0, "combo_seen1": combo_seen1, "combo_seen2": combo_seen2, "unseen_single": unseen_single},
    )


def _perts_from_genes(genes: Iterable[str], pert_list: Iterable[str], type_: str) -> list[str]:
    gene_set = set(str(gene) for gene in genes)
    if type_ == "single":
        candidates = [condition for condition in pert_list if is_single(condition)]
    elif type_ == "combo":
        candidates = [condition for condition in pert_list if is_combo(condition)]
    elif type_ == "both":
        candidates = list(pert_list)
    else:
        raise ValueError(f"unsupported perturbation type: {type_}")
    return [condition for condition in candidates if any(gene in gene_set for gene in perturbation_genes(condition))]


def _genes_from_perts(perts: Iterable[str]) -> np.ndarray:
    genes = []
    for condition in np.unique(list(perts)):
        genes.extend(perturbation_genes(str(condition)))
    return np.unique([gene for gene in genes if gene != "ctrl"])


def _uns_gene_dict(payload: Any) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    if not isinstance(payload, dict):
        return result
    for key, value in payload.items():
        result[str(key)] = tuple(str(item) for item in list(value))
    return result
