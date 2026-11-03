# legacy_v1 copy of src/esn/model.py (frozen v1 recipe). Only import paths were changed.
"""Numpy Echo State Network with fixed reservoir and ridge readout."""

from __future__ import annotations

import contextlib
import io

import numpy as np

try:
    with contextlib.redirect_stderr(io.StringIO()):
        import pandas as pd
except Exception:  # pragma: no cover - pandas availability is environment-specific
    pd = None


class EchoStateNetwork:
    """Echo State Network for continuous CPM-based trading score prediction.

    The input weights and reservoir weights are random fixed matrices. Only the
    output readout ``Wout`` is learned, using ridge regression on extended states
    of the form ``[bias, input, reservoir_state]``. This is a basic ESN without
    teacher-output feedback/Wback; add Wback later only if the experiment needs it.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scale: float = 0.5,
        leaking_rate: float = 0.3,
        ridge_alpha: float = 1.0,
        washout: int = 30,
        random_state: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive.")
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("sparsity must be between 0 and 1.")
        if not 0.0 < leaking_rate <= 1.0:
            raise ValueError("leaking_rate must be in (0, 1].")
        if ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")
        if washout < 0:
            raise ValueError("washout must be non-negative.")

        self.input_dim = int(input_dim)
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.sparsity = float(sparsity)
        self.input_scale = float(input_scale)
        self.leaking_rate = float(leaking_rate)
        self.ridge_alpha = float(ridge_alpha)
        self.washout = int(washout)
        self.random_state = int(random_state)

        # Win and W (reservoir) use independent RNG streams, and Win is drawn
        # one input-column at a time, so that adding/removing an input
        # feature only appends/removes its own column of random weights
        # instead of reshuffling every other column and the reservoir matrix
        # (both would otherwise change because np.random.default_rng draws
        # are consumed sequentially and size=(reservoir_size, input_dim)
        # fills row-major, so changing input_dim shifts every later draw).
        win_seed, reservoir_seed = np.random.SeedSequence(self.random_state).spawn(2)
        self.rng_win_ = np.random.default_rng(win_seed)
        self.rng_reservoir_ = np.random.default_rng(reservoir_seed)

        self.Win = self.rng_win_.uniform(
            low=-self.input_scale,
            high=self.input_scale,
            size=(self.input_dim, self.reservoir_size),
        ).T
        self.W = self._initialize_reservoir()
        self.state = np.zeros(self.reservoir_size, dtype=float)
        self.Wout: np.ndarray | None = None

    def fit(
        self, X: object, y: object, washout: int | None = None, sample_mask: object = None
    ) -> "EchoStateNetwork":
        """Fit the ridge readout using washout-trimmed reservoir states.

        ``sample_mask`` (optional, aligned with X/y before washout trimming)
        restricts which rows contribute to the ridge regression *after* the
        reservoir has processed the full, contiguous X sequence -- e.g. for
        CUSUM event sampling, where only a subset of days should count
        toward the fit but the recurrence still needs the true day-by-day
        sequence to produce meaningful states.
        """

        X_array = self._validate_X(X)
        y_array = self._validate_y(y)
        washout_rows = self.washout if washout is None else int(washout)
        if len(X_array) != len(y_array):
            raise ValueError(f"X and y lengths differ: {len(X_array)} != {len(y_array)}.")
        if washout_rows < 0:
            raise ValueError("washout must be non-negative.")
        if len(X_array) <= washout_rows:
            raise ValueError(
                f"Need more samples than washout ({washout_rows}); got {len(X_array)}."
            )

        states = self._collect_states(X_array)
        train_states = states[washout_rows:]
        train_X = X_array[washout_rows:]
        train_y = y_array[washout_rows:]
        if sample_mask is not None:
            mask = np.asarray(sample_mask, dtype=bool)
            if len(mask) != len(X_array):
                raise ValueError(f"sample_mask length {len(mask)} != X length {len(X_array)}.")
            mask = mask[washout_rows:]
            train_states = train_states[mask]
            train_X = train_X[mask]
            train_y = train_y[mask]
        design = self._extended_state(train_X, train_states)

        penalty = np.eye(design.shape[1], dtype=float) * self.ridge_alpha
        penalty[0, 0] = 0.0
        lhs = design.T @ design + penalty
        rhs = design.T @ train_y
        self.Wout = np.linalg.solve(lhs, rhs)
        return self

    def predict(self, X: object, washout: int = 0) -> object:
        """Predict clipped continuous trading scores for X in time order."""

        if self.Wout is None:
            raise RuntimeError("EchoStateNetwork must be fitted before predict().")
        if washout < 0:
            raise ValueError("washout must be non-negative.")

        index = X.index if pd is not None and isinstance(X, pd.DataFrame) else None
        X_array = self._validate_X(X)
        states = self._collect_states(X_array)
        if washout:
            states = states[washout:]
            X_array = X_array[washout:]
            if index is not None:
                index = index[washout:]
        scores = np.clip(self._extended_state(X_array, states) @ self.Wout, -1.0, 1.0)
        if index is not None and pd is not None:
            return pd.Series(scores, index=index, name="trading_score")
        return scores

    def _initialize_reservoir(self) -> np.ndarray:
        weights = self.rng_reservoir_.uniform(
            low=-1.0,
            high=1.0,
            size=(self.reservoir_size, self.reservoir_size),
        )
        mask = self.rng_reservoir_.random(size=weights.shape) < self.sparsity
        weights *= mask

        eigenvalues = np.linalg.eigvals(weights)
        current_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        if current_radius > 0.0:
            weights *= self.spectral_radius / current_radius
        return weights

    def _collect_states(self, X: np.ndarray) -> np.ndarray:
        self._reset_state()
        states = np.empty((len(X), self.reservoir_size), dtype=float)
        for idx, x_t in enumerate(X):
            states[idx] = self._update_state(x_t)
        return states

    def _update_state(self, x_t: np.ndarray) -> np.ndarray:
        pre_activation = self.Win @ x_t + self.W @ self.state
        new_state = (1.0 - self.leaking_rate) * self.state
        new_state += self.leaking_rate * np.tanh(pre_activation)
        self.state = new_state
        return self.state.copy()

    def _reset_state(self) -> None:
        self.state = np.zeros(self.reservoir_size, dtype=float)

    def _validate_X(self, X: object) -> np.ndarray:
        if pd is not None and isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected X with {self.input_dim} features, got {X.shape[1]}."
            )
        return X

    @staticmethod
    def _validate_y(y: object) -> np.ndarray:
        if pd is not None and isinstance(y, pd.Series):
            y = y.to_numpy()
        return np.asarray(y, dtype=float).reshape(-1)

    @staticmethod
    def _extended_state(X: np.ndarray, states: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(states), dtype=float), X, states])
