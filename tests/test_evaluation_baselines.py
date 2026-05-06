import numpy as np
import pandas as pd

from perturb_jepa.evaluation.baselines import (
    CentroidRetrievalBaseline,
    ControlMeanBaseline,
    PerturbationMeanBaseline,
    centroid_retrieval_metrics,
    label_shuffle_centroid_retrieval_metrics,
    shuffle_labels,
)


def test_control_mean_baseline_predicts_control_average():
    expression = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
        ]
    )
    metadata = pd.DataFrame(
        {
            "perturbation": ["DMSO", "DMSO", "drugA"],
            "perturbation_type": ["control", "control", "compound"],
        }
    )

    prediction = ControlMeanBaseline().fit(expression, metadata).predict(2)

    np.testing.assert_allclose(prediction, np.array([[2.0, 3.0], [2.0, 3.0]]))


def test_perturbation_mean_baseline_uses_group_mean_and_control_fallback():
    train_expression = np.array(
        [
            [1.0, 1.0],
            [10.0, 0.0],
            [12.0, 2.0],
        ]
    )
    train_metadata = pd.DataFrame(
        {
            "perturbation": ["DMSO", "drugA", "drugA"],
            "perturbation_type": ["control", "compound", "compound"],
        }
    )
    eval_metadata = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugB"],
            "perturbation_type": ["compound", "compound"],
        }
    )

    prediction = (
        PerturbationMeanBaseline(groupby="perturbation", fallback="control")
        .fit(train_expression, train_metadata)
        .predict(eval_metadata)
    )

    np.testing.assert_allclose(prediction, np.array([[11.0, 1.0], [1.0, 1.0]]))


def test_centroid_retrieval_and_label_shuffle_negative_control():
    gallery = np.eye(3)
    query = np.eye(3)
    labels = ["a", "b", "c"]

    metrics = centroid_retrieval_metrics(query, gallery, labels, labels, ks=(1,))
    shuffled_metrics = label_shuffle_centroid_retrieval_metrics(
        query,
        gallery,
        labels,
        labels,
        ks=(1,),
        random_state=7,
    )

    assert metrics["centroid_map"] == 1.0
    assert metrics["centroid_recall@1"] == 1.0
    assert shuffled_metrics["centroid_recall@1"] == 0.0
    assert shuffle_labels(labels, random_state=7) != labels


def test_centroid_retrieval_baseline_predicts_labels_from_metadata():
    gallery = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    gallery_metadata = pd.DataFrame({"condition_key": ["drugA", "drugA", "drugB"]})
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_metadata = pd.DataFrame({"condition_key": ["drugA", "drugB"]})

    baseline = CentroidRetrievalBaseline(label_col="condition_key").fit(gallery, gallery_metadata)

    assert baseline.predict(query) == ["drugA", "drugB"]
    assert baseline.evaluate(query, query_metadata, ks=(1,))["centroid_recall@1"] == 1.0
