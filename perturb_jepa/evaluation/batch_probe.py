from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def batch_probe_metrics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame | Mapping[str, Sequence[object]],
    *,
    label_col: str = "batch",
    prefix: str = "batch_probe",
    n_splits: int = 5,
    random_state: int = 0,
) -> dict[str, float]:
    """Estimate how easily technical labels can be decoded from embeddings.

    This is a leakage diagnostic, not a biological score. High probe accuracy
    means the retrieval embedding still carries technical acquisition signal.
    Cross-validation is used whenever each class has at least two examples.
    """

    values = _as_2d_float_array(embeddings, name="embeddings")
    frame = _as_metadata_frame(metadata, n_samples=values.shape[0])
    if label_col not in frame.columns:
        return {
            f"{prefix}_n_samples": float(values.shape[0]),
            f"{prefix}_n_classes": 0.0,
            f"{prefix}_majority_accuracy": 0.0,
            f"{prefix}_resub_accuracy": float("nan"),
            f"{prefix}_resub_balanced_accuracy": float("nan"),
            f"{prefix}_accuracy": float("nan"),
            f"{prefix}_balanced_accuracy": float("nan"),
            f"{prefix}_cv_folds": 0.0,
        }

    labels = frame[label_col].map(_string_value).to_numpy(dtype=object)
    keep = np.asarray([label not in {"", "NA", "nan"} for label in labels], dtype=bool)
    values = values[keep]
    labels = labels[keep]
    if labels.size == 0:
        return {
            f"{prefix}_n_samples": 0.0,
            f"{prefix}_n_classes": 0.0,
            f"{prefix}_majority_accuracy": 0.0,
            f"{prefix}_resub_accuracy": float("nan"),
            f"{prefix}_resub_balanced_accuracy": float("nan"),
            f"{prefix}_accuracy": float("nan"),
            f"{prefix}_balanced_accuracy": float("nan"),
            f"{prefix}_cv_folds": 0.0,
        }

    classes, counts = np.unique(labels, return_counts=True)
    majority_accuracy = float(counts.max() / counts.sum())
    if classes.size < 2:
        return {
            f"{prefix}_n_samples": float(labels.size),
            f"{prefix}_n_classes": float(classes.size),
            f"{prefix}_majority_accuracy": majority_accuracy,
            f"{prefix}_resub_accuracy": float("nan"),
            f"{prefix}_resub_balanced_accuracy": float("nan"),
            f"{prefix}_accuracy": float("nan"),
            f"{prefix}_balanced_accuracy": float("nan"),
            f"{prefix}_cv_folds": 0.0,
        }

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as exc:
        raise ImportError("batch probe metrics require scikit-learn; install the evaluation extra") from exc

    resub_estimator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
    resub_estimator.fit(values, labels)
    resub_predictions = resub_estimator.predict(values)
    result = {
        f"{prefix}_n_samples": float(labels.size),
        f"{prefix}_n_classes": float(classes.size),
        f"{prefix}_majority_accuracy": majority_accuracy,
        f"{prefix}_resub_accuracy": float(accuracy_score(labels, resub_predictions)),
        f"{prefix}_resub_balanced_accuracy": float(balanced_accuracy_score(labels, resub_predictions)),
    }

    if counts.min() < 2:
        result.update(
            {
                f"{prefix}_accuracy": float("nan"),
                f"{prefix}_balanced_accuracy": float("nan"),
                f"{prefix}_cv_folds": 0.0,
            }
        )
        return result

    folds = min(int(n_splits), int(counts.min()))
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    predictions = cross_val_predict(estimator, values, labels, cv=cv)
    result.update(
        {
            f"{prefix}_accuracy": float(accuracy_score(labels, predictions)),
            f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            f"{prefix}_cv_folds": float(folds),
        }
    )
    return result


def _as_2d_float_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features]")
    return array


def _as_metadata_frame(metadata: pd.DataFrame | Mapping[str, Sequence[object]], *, n_samples: int) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if len(frame) != n_samples:
        raise ValueError("metadata row count must match embedding rows")
    return frame


def _string_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(value)
