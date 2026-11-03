# legacy_v1 copy of src/utils/scaling.py (frozen v1 recipe). Only import paths were changed.
"""Scale raw (continuous) indicator values into a fixed [-1, 1] range.

Unlike ``StandardScaler`` (mean 0, unit variance, unbounded), this scaler
maps each feature's train-time min/max to -1/+1, which is the input range
ESN reservoirs are typically tuned for. The scaler is fit on train rows only
and clipped on transform, so validation/test rows are never used to choose
the scale (no lookahead) and out-of-range values still land inside [-1, 1]
instead of exploding past it.
"""

from __future__ import annotations

import numpy as np


class MinMaxUnitScaler:
    """Fit min/max on train data, then map features to [-1, 1]."""

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxUnitScaler":
        X = np.asarray(X, dtype=float)
        self.min_ = np.nanmin(X, axis=0)
        self.max_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxUnitScaler must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        value_range = self.max_ - self.min_
        value_range = np.where(value_range == 0.0, 1.0, value_range)
        scaled = 2.0 * (X - self.min_) / value_range - 1.0
        return np.clip(scaled, -1.0, 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
