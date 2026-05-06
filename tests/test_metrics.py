import numpy as np

from perturb_jepa.evaluation.metrics import (
    distance_matrix_spearman,
    mean_average_precision,
    pearson_delta,
    recall_at_k,
    topk_jaccard,
)


def test_expression_metrics():
    control = np.array([1.0, 1.0, 1.0])
    observed = np.array([2.0, 1.0, 0.5])
    predicted = np.array([1.9, 1.1, 0.4])
    assert pearson_delta(predicted, observed, control) > 0.9
    assert topk_jaccard(predicted - control, observed - control, k=1) == 1.0


def test_retrieval_metrics():
    query = np.eye(3)
    gallery = np.eye(3)
    labels = ["a", "b", "c"]
    assert mean_average_precision(query, gallery, labels, labels) == 1.0
    assert recall_at_k(query, gallery, labels, labels, k=1) == 1.0


def test_distance_matrix_spearman_is_high_for_same_geometry():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert distance_matrix_spearman(points, points) > 0.99
