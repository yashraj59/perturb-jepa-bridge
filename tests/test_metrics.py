import numpy as np

from perturb_jepa.evaluation.metrics import (
    distance_matrix_spearman,
    dose_response_monotonicity,
    mean_average_precision,
    pearson_delta,
    recall_at_k,
    topk_de_recovery,
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


def test_topk_de_recovery_averages_per_sample_recall():
    control = np.zeros((2, 4))
    observed = np.array([[0.0, 1.0, 4.0, 0.0], [3.0, 0.0, 1.0, 0.0]])
    predicted = np.array([[0.0, 2.0, 5.0, 1.0], [2.0, 4.0, 1.0, 0.0]])

    assert topk_de_recovery(predicted, observed, control, k=1) == 0.5
    assert topk_de_recovery(predicted, observed, control, k=2) == 0.75


def test_dose_response_monotonicity_scores_pairwise_ordering():
    responses = np.array([0.1, 0.5, 0.4])

    assert dose_response_monotonicity(responses, ["1uM", "2uM", "3uM"]) == 2 / 3


def test_dose_response_monotonicity_averages_groups():
    responses = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    doses = np.array([1.0, 10.0, 1.0, 10.0])
    groups = ["drugA", "drugA", "drugB", "drugB"]

    assert dose_response_monotonicity(responses, doses, groups=groups, control=np.zeros(2)) == 0.5
