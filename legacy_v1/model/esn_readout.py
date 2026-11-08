"""ESN helpers for scoring many parameter sets fast: reservoir states over one unbroken run, a ridge readout on any rows,
thresholded signals, and the inner selection years of a training window. Used by experiments/select_esn_backtest.py.

Parameter dicts: reservoir_size, spectral_radius, sparsity, input_scale, leaking_rate, log10_alpha (ridge penalty 10**log10_alpha).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from legacy_v1.model.esn import EchoStateNetwork

WASHOUT = 30


def esn_states(X: np.ndarray, params: dict, seed: int) -> tuple[EchoStateNetwork, np.ndarray]:
    """The reservoir states of X (rows in time order, no reset) and the ESN object (for its readout design)."""
    model = EchoStateNetwork(input_dim=X.shape[1], reservoir_size=int(params["reservoir_size"]), spectral_radius=params["spectral_radius"],
                             sparsity=params["sparsity"], input_scale=params["input_scale"], leaking_rate=params["leaking_rate"],
                             ridge_alpha=10.0 ** params["log10_alpha"], washout=WASHOUT, random_state=seed)
    return model, model._collect_states(X)


def fit_readout(model: EchoStateNetwork, X: np.ndarray, states: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Ridge weights on [1, inputs, states] over `rows` (the same solve as EchoStateNetwork.fit; the bias is not penalised)."""
    design = model._extended_state(X[rows], states[rows])
    penalty = np.eye(design.shape[1]) * model.ridge_alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y[rows])


def scores(model: EchoStateNetwork, X: np.ndarray, states: np.ndarray, weights: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.clip(model._extended_state(X[rows], states[rows]) @ weights, -1.0, 1.0)


def to_signal(score: np.ndarray, threshold: float) -> np.ndarray:
    """-1 (buy) below -threshold, +1 (sell) above threshold, else 0."""
    return np.where(score < -threshold, -1, np.where(score > threshold, 1, 0)).astype(float)


def inner_folds(dates: pd.Series, n_years: int = 5, first_val_year: int = 3) -> list[tuple[int, int]]:
    """(start, end) row positions of training years first_val_year .. n_years of a window starting at dates.iloc[0]."""
    d = dates.to_numpy()
    pos = lambda k: int(np.searchsorted(d, np.datetime64(dates.iloc[0] + pd.DateOffset(years=k))))  # noqa: E731
    return [(pos(k - 1), min(pos(k), len(d))) for k in range(first_val_year, n_years + 1) if pos(k - 1) < len(d)]
