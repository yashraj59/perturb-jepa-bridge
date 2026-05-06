"""Evaluation utilities for perturbation modeling."""

from perturb_jepa.evaluation.baselines import (
    CentroidRetrievalBaseline,
    ControlMeanBaseline,
    PerturbationMeanBaseline,
    centroid_retrieval_metrics,
    infer_control_mask,
    label_shuffle_centroid_retrieval_metrics,
    label_shuffle_retrieval_metrics,
    shuffle_labels,
)
from perturb_jepa.evaluation.metrics import (
    centroid_by_condition,
    distance_matrix_spearman,
    mean_average_precision,
    pearson_delta,
    recall_at_k,
    spearman_delta,
    topk_jaccard,
)
from perturb_jepa.evaluation.reporting import (
    default_delta_metrics,
    delta_mean_absolute_error,
    delta_mean_squared_error,
    grouped_metric_report,
)

__all__ = [
    "CentroidRetrievalBaseline",
    "ControlMeanBaseline",
    "PerturbationMeanBaseline",
    "centroid_by_condition",
    "centroid_retrieval_metrics",
    "default_delta_metrics",
    "delta_mean_absolute_error",
    "delta_mean_squared_error",
    "distance_matrix_spearman",
    "grouped_metric_report",
    "infer_control_mask",
    "label_shuffle_centroid_retrieval_metrics",
    "label_shuffle_retrieval_metrics",
    "mean_average_precision",
    "pearson_delta",
    "recall_at_k",
    "shuffle_labels",
    "spearman_delta",
    "topk_jaccard",
]
