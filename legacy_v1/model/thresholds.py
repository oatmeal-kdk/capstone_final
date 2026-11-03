# legacy_v1: _select_thresholds of src/pipeline/run_esn.py (copied unchanged).
from __future__ import annotations

import numpy as np

from legacy_v1.model.metrics import classification_like_metrics, score_to_signal


def _select_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_values: list[float],
) -> dict[str, float]:
    best: dict[str, float] | None = None
    y_true_signal = np.asarray(score_to_signal(y_true, 0.0), dtype=int)
    for threshold in threshold_values:
        y_signal = np.asarray(score_to_signal(y_pred, threshold), dtype=int)
        metrics = classification_like_metrics(y_true_signal, y_signal)
        score = float(metrics["accuracy"])
        if best is None or score > best["validation_accuracy"]:
            best = {
                "threshold": float(threshold),
                "validation_accuracy": score,
            }
    if best is None:
        raise ValueError("No threshold combinations were provided.")
    return best
