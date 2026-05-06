import numpy as np
import pandas as pd

from perturb_jepa.baselines import (
    batch_only_retrieval_metrics,
    metadata_only_retrieval_metrics,
)
from perturb_jepa.evaluation.retrieval import cross_modal_retrieval_metrics


def test_cross_modal_retrieval_reports_both_directions_and_strata():
    embeddings = np.eye(4)
    metadata = pd.DataFrame(
        {
            "condition_key": ["a", "b", "c", "d"],
            "perturbation": ["p1", "p2", "p3", "p4"],
            "dose": [1, 1, 10, 10],
            "time": [24, 24, 48, 48],
            "cell_line": ["A", "A", "B", "B"],
            "batch": ["x", "x", "y", "y"],
            "moa": ["m1", "m2", "m1", "m2"],
        }
    )

    metrics = cross_modal_retrieval_metrics(embeddings, embeddings, metadata, metadata, ks=(1, 5, 10))

    assert metrics["rna_to_image_recall@1"] == 1.0
    assert metrics["rna_to_image_recall@5"] == 1.0
    assert metrics["rna_to_image_recall@10"] == 1.0
    assert metrics["rna_to_image_map"] == 1.0
    assert metrics["rna_to_image_median_rank"] == 1.0
    assert metrics["image_to_rna_recall@1"] == 1.0
    assert metrics["rna_to_image_same_perturbation_enrichment@10"] >= 1.0
    assert metrics["rna_to_image_same_moa_enrichment@10"] >= 1.0
    assert metrics["rna_to_image_by_batch=x_recall@1"] == 1.0
    assert metrics["rna_to_image_by_cell_line=B_median_rank"] == 1.0


def test_metadata_and_batch_only_retrieval_use_requested_columns():
    query = pd.DataFrame(
        {
            "condition_key": ["drugA|1|24|A549", "drugB|1|24|A549"],
            "perturbation": ["drugA", "drugB"],
            "dose": [1, 1],
            "time": [24, 24],
            "cell_line": ["A549", "A549"],
            "batch": ["batch1", "batch1"],
        }
    )
    gallery = query.copy()

    metadata_metrics = metadata_only_retrieval_metrics(query, gallery, ks=(1,))
    batch_metrics = batch_only_retrieval_metrics(query, gallery, ks=(1,))

    assert metadata_metrics["metadata_only_recall@1"] == 1.0
    assert batch_metrics["batch_only_recall@1"] == 0.5
