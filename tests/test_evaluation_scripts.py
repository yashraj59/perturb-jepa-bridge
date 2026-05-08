import numpy as np
import pandas as pd

from scripts.evaluate_expression_baselines import main as expression_main
from scripts.evaluate_retrieval import main as learned_retrieval_main
from scripts.evaluate_retrieval_baselines import main as retrieval_main


def test_evaluate_expression_baselines_cli_writes_grouped_report(tmp_path):
    train_expression = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [4.0, 1.0],
        ]
    )
    eval_expression = np.array(
        [
            [3.0, 1.0],
            [1.0, 2.0],
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
    train_expression_path = tmp_path / "train_expression.npy"
    eval_expression_path = tmp_path / "eval_expression.npy"
    train_metadata_path = tmp_path / "train_metadata.csv"
    eval_metadata_path = tmp_path / "eval_metadata.csv"
    output_path = tmp_path / "metrics.csv"
    np.save(train_expression_path, train_expression)
    np.save(eval_expression_path, eval_expression)
    train_metadata.to_csv(train_metadata_path, index=False)
    eval_metadata.to_csv(eval_metadata_path, index=False)

    exit_code = expression_main(
        [
            "--train-expression",
            str(train_expression_path),
            "--train-metadata",
            str(train_metadata_path),
            "--eval-expression",
            str(eval_expression_path),
            "--eval-metadata",
            str(eval_metadata_path),
            "--topk",
            "1",
            "--output",
            str(output_path),
        ]
    )

    report = pd.read_csv(output_path)
    assert exit_code == 0
    assert set(report["baseline"]) == {"control_mean", "perturbation_mean"}
    assert set(report["group"]) == {"overall", "drugA", "drugB"}


def test_evaluate_expression_baselines_cli_supports_cell_line_transfer_grouping(tmp_path):
    train_expression = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [5.0, 1.0],
        ]
    )
    eval_expression = np.array(
        [
            [4.0, 1.0],
            [4.0, 2.0],
        ]
    )
    train_metadata = pd.DataFrame(
        {
            "perturbation": ["DMSO", "drugA", "drugA"],
            "perturbation_type": ["control", "compound", "compound"],
            "cell_line": ["U2OS", "U2OS", "U2OS"],
        }
    )
    eval_metadata = pd.DataFrame(
        {
            "perturbation": ["drugA", "drugA"],
            "perturbation_type": ["compound", "compound"],
            "cell_line": ["U2OS", "A549"],
        }
    )
    train_expression_path = tmp_path / "train_expression.npy"
    eval_expression_path = tmp_path / "eval_expression.npy"
    train_metadata_path = tmp_path / "train_metadata.csv"
    eval_metadata_path = tmp_path / "eval_metadata.csv"
    output_path = tmp_path / "transfer_metrics.csv"
    np.save(train_expression_path, train_expression)
    np.save(eval_expression_path, eval_expression)
    train_metadata.to_csv(train_metadata_path, index=False)
    eval_metadata.to_csv(eval_metadata_path, index=False)

    exit_code = expression_main(
        [
            "--train-expression",
            str(train_expression_path),
            "--train-metadata",
            str(train_metadata_path),
            "--eval-expression",
            str(eval_expression_path),
            "--eval-metadata",
            str(eval_metadata_path),
            "--report-groupby",
            "cell_line_transfer,cell_line",
            "--topk",
            "1",
            "--output",
            str(output_path),
        ]
    )

    report = pd.read_csv(output_path)
    assert exit_code == 0
    assert set(report["cell_line_transfer"].dropna()) == {"seen", "held_out"}
    assert set(report["group"]) == {"overall", "held_out|A549", "seen|U2OS"}


def test_evaluate_retrieval_baselines_cli_writes_centroid_and_shuffle_rows(tmp_path):
    embeddings = np.eye(3)
    metadata = pd.DataFrame({"condition_key": ["a", "b", "c"]})
    query_path = tmp_path / "query.npy"
    gallery_path = tmp_path / "gallery.npy"
    query_metadata_path = tmp_path / "query.csv"
    gallery_metadata_path = tmp_path / "gallery.csv"
    output_path = tmp_path / "retrieval.csv"
    np.save(query_path, embeddings)
    np.save(gallery_path, embeddings)
    metadata.to_csv(query_metadata_path, index=False)
    metadata.to_csv(gallery_metadata_path, index=False)

    exit_code = retrieval_main(
        [
            "--query-embeddings",
            str(query_path),
            "--query-metadata",
            str(query_metadata_path),
            "--gallery-embeddings",
            str(gallery_path),
            "--gallery-metadata",
            str(gallery_metadata_path),
            "--ks",
            "1",
            "--output",
            str(output_path),
        ]
    )

    report = pd.read_csv(output_path)
    assert exit_code == 0
    assert report["baseline"].tolist() == ["centroid_retrieval", "label_shuffle_centroid"]
    assert report.loc[0, "retrieval_recall@1"] == 1.0
    assert report.loc[1, "retrieval_recall@1"] == 0.0


def test_learned_retrieval_cli_labels_oracle_and_trainfit_mean_prototypes(tmp_path):
    embeddings = np.eye(3, dtype=np.float32)
    metadata = pd.DataFrame(
        {
            "condition_key": ["a", "b", "c"],
            "perturbation": ["a", "b", "c"],
            "dose": ["1", "1", "1"],
            "time": ["24", "24", "24"],
            "cell_line": ["A", "A", "A"],
            "batch": ["rna", "rna", "rna"],
        }
    )
    train_target = np.eye(3, dtype=np.float32)
    train_target_metadata = metadata.assign(batch=["img_train", "img_train", "img_train"])
    paths = {
        "rna_embeddings": tmp_path / "rna.npy",
        "image_embeddings": tmp_path / "image.npy",
        "rna_metadata": tmp_path / "rna.csv",
        "image_metadata": tmp_path / "image.csv",
        "train_target_embeddings": tmp_path / "train_target.npy",
        "train_target_metadata": tmp_path / "train_target.csv",
        "output": tmp_path / "learned.csv",
    }
    np.save(paths["rna_embeddings"], embeddings)
    np.save(paths["image_embeddings"], embeddings)
    np.save(paths["train_target_embeddings"], train_target)
    metadata.to_csv(paths["rna_metadata"], index=False)
    metadata.assign(batch=["img", "img", "img"]).to_csv(paths["image_metadata"], index=False)
    train_target_metadata.to_csv(paths["train_target_metadata"], index=False)

    exit_code = learned_retrieval_main(
        [
            "--rna-embeddings",
            str(paths["rna_embeddings"]),
            "--image-embeddings",
            str(paths["image_embeddings"]),
            "--rna-metadata",
            str(paths["rna_metadata"]),
            "--image-metadata",
            str(paths["image_metadata"]),
            "--train-target-embeddings",
            str(paths["train_target_embeddings"]),
            "--train-target-metadata",
            str(paths["train_target_metadata"]),
            "--output",
            str(paths["output"]),
        ]
    )

    report = pd.read_csv(paths["output"])
    assert exit_code == 0
    assert set(report["method"]) == {
        "learned",
        "metadata_only",
        "batch_only",
        "mean_prototype_trainfit",
        "mean_prototype_oracle",
    }
