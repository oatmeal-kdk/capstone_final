# legacy_v1 copy of src/esn/metrics.py (frozen v1 recipe). Only import paths were changed.
"""Metrics and signal thresholding helpers for ESN outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas availability is environment-specific
    pd = None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return basic continuous-output metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred lengths differ: {len(y_true)} != {len(y_pred)}.")

    error = y_true - y_pred
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    return {
        "mse": float(np.mean(error**2)) if len(error) else 0.0,
        "mae": float(np.mean(np.abs(error))) if len(error) else 0.0,
        "directional_accuracy": float(np.mean(true_direction == pred_direction))
        if len(error)
        else 0.0,
    }


def signalize_predictions(
    y_pred: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
) -> np.ndarray:
    """Convert continuous predictions into {-1, 0, 1} signals.

    This compatibility wrapper keeps the older buy/sell threshold API. For the
    CPM triangle score convention, prefer ``score_to_signal``.
    """

    if sell_threshold >= buy_threshold:
        raise ValueError(
            f"sell_threshold must be smaller than buy_threshold: "
            f"{sell_threshold} >= {buy_threshold}"
        )
    y_pred = np.asarray(y_pred, dtype=float)
    signals = np.zeros(len(y_pred), dtype=int)
    signals[y_pred >= buy_threshold] = 1
    signals[y_pred <= sell_threshold] = -1
    return signals


def score_to_signal(scores: object, threshold: float, as_label: bool = False) -> object:
    """Convert ESN trading scores to buy/sell/hold signals.

    Scores below ``-threshold`` become buy, scores above ``threshold`` become
    sell, and all remaining scores become hold. Numeric output uses buy=-1,
    sell=1, hold=0.
    """

    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")
    index = scores.index if pd is not None and isinstance(scores, pd.Series) else None
    values = np.asarray(scores, dtype=float)
    numeric = np.zeros(len(values), dtype=int)
    numeric[values < -threshold] = -1
    numeric[values > threshold] = 1
    result: np.ndarray
    if as_label:
        result = np.where(numeric == -1, "buy", np.where(numeric == 1, "sell", "hold"))
    else:
        result = numeric
    if index is not None and pd is not None:
        return pd.Series(result, index=index, name="signal")
    return result


def classification_like_metrics(
    y_true: np.ndarray,
    y_signal: np.ndarray,
) -> dict[str, Any]:
    """Return class-style metrics for thresholded ESN signals."""

    y_true = np.asarray(y_true, dtype=int)
    y_signal = np.asarray(y_signal, dtype=int)
    if len(y_true) != len(y_signal):
        raise ValueError(
            f"y_true and y_signal lengths differ: {len(y_true)} != {len(y_signal)}."
        )

    cm = _confusion_counts(y_true, y_signal)
    return {
        "accuracy": float(np.mean(y_true == y_signal)) if len(y_true) else 0.0,
        "buy_precision": _precision(cm, -1),
        "buy_recall": _recall(cm, -1),
        "sell_precision": _precision(cm, 1),
        "sell_recall": _recall(cm, 1),
        "confusion_counts": cm,
    }


def _confusion_counts(y_true: np.ndarray, y_signal: np.ndarray) -> dict[str, int]:
    labels = [-1, 0, 1]
    counts: dict[str, int] = {}
    for true_label in labels:
        for pred_label in labels:
            key = f"true_{true_label}_pred_{pred_label}"
            counts[key] = int(np.sum((y_true == true_label) & (y_signal == pred_label)))
    return counts


def _precision(counts: dict[str, int], label: int) -> float:
    tp = counts[f"true_{label}_pred_{label}"]
    predicted = sum(counts[f"true_{true_label}_pred_{label}"] for true_label in [-1, 0, 1])
    return float(tp / predicted) if predicted else 0.0


def _recall(counts: dict[str, int], label: int) -> float:
    tp = counts[f"true_{label}_pred_{label}"]
    actual = sum(counts[f"true_{label}_pred_{pred_label}"] for pred_label in [-1, 0, 1])
    return float(tp / actual) if actual else 0.0
