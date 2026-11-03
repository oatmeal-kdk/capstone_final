# legacy_v1 copy of src/cpm/triple_barrier.py (frozen v1 recipe). Only import paths were changed.
"""ATR-scaled triple-barrier labeling (Lopez de Prado style).

Alternative to CPM turning points for building the ESN training target. For
each row t, an upper (profit-take) and lower (stop-loss) barrier are set at
+-k * ATR[t] around Close[t], and a vertical (time) barrier caps the lookahead
at ``max_holding`` rows. Whichever barrier is touched first in
[t+1, t+max_holding] decides the label; touching neither means the vertical
barrier wins.

Label convention matches the project's existing turning_label/triangle_target
convention (buy=-1, sell=1, hold=0): the upper barrier means price rose, i.e.
entering at t would have been a good buy, so it's labeled -1; the lower
barrier means price fell, labeled +1; a time-out with no clear move is 0.

Unlike CPM's turning points (which require an unbounded forward scan to
confirm a P% reversal), every label here depends on at most ``max_holding``
future rows, which makes it straightforward to reason about causality in a
walk-forward fold: a label at row t is fully determined by rows
[t, t+max_holding].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.volatility import average_true_range


def label_triple_barrier(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    atr_window: int = 14,
    k: float = 2.0,
    max_holding: int = 10,
) -> pd.Series:
    """Return a {-1, 0, 1} Series labeled via ATR-scaled triple barriers."""

    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    close = df[close_col].astype(float).to_numpy()
    atr = average_true_range(
        df[high_col].astype(float), df[low_col].astype(float), df[close_col].astype(float), window=atr_window
    ).to_numpy()

    n = len(df)
    labels = np.full(n, np.nan, dtype=float)

    for t in range(n):
        if np.isnan(atr[t]) or atr[t] <= 0:
            continue
        entry = close[t]
        upper = entry + k * atr[t]
        lower = entry - k * atr[t]
        end = min(t + max_holding, n - 1)
        if end <= t:
            continue

        label = 0.0
        for future in range(t + 1, end + 1):
            hit_upper = high[future] >= upper
            hit_lower = low[future] <= lower
            if hit_upper and hit_lower:
                # both touched same bar: assume the adverse (stop-loss) side first
                label = 1.0
                break
            if hit_upper:
                label = -1.0
                break
            if hit_lower:
                label = 1.0
                break
        labels[t] = label

    return pd.Series(labels, index=df.index, name="barrier_target")


def label_vol_adjusted_horizon(
    df: pd.DataFrame,
    close_col: str = "Close",
    horizon: int = 10,
    vol_window: int = 20,
    threshold: float = 1.0,
) -> pd.Series:
    """Return a {-1, 0, 1} Series labeled by horizon-ahead return / trailing volatility.

    Simpler alternative to the triple barrier: one fixed forward horizon
    instead of dynamic barriers with early exit. The raw forward return is
    normalized by trailing realized volatility (scaled to the horizon), so
    the same raw % move counts as a bigger signal in a calm regime than a
    choppy one. Every label depends on at most ``horizon`` future rows --
    same bounded-lookahead, purge-with-_full_horizon_mask causal property as
    label_triple_barrier.
    """

    close = df[close_col].astype(float).to_numpy()
    n = len(df)

    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    trailing_vol = pd.Series(log_ret).rolling(vol_window).std().to_numpy() * np.sqrt(horizon)

    fwd_ret = np.full(n, np.nan)
    valid_end = n - horizon
    if valid_end > 0:
        fwd_ret[:valid_end] = close[horizon:] / close[:valid_end] - 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        score = fwd_ret / trailing_vol

    labels = np.full(n, np.nan)
    valid = ~np.isnan(score) & (trailing_vol > 0)
    labels[valid & (score > threshold)] = -1.0
    labels[valid & (score < -threshold)] = 1.0
    labels[valid & (np.abs(score) <= threshold)] = 0.0

    return pd.Series(labels, index=df.index, name="vol_adjusted_target")


def label_trend_regime(
    df: pd.DataFrame,
    close_col: str = "Close",
    horizon: int = 10,
    threshold_pct: float = 0.02,
) -> pd.Series:
    """Return a {-1, 0, 1} Series: is the next ``horizon`` days an uptrend/downtrend/sideways regime?

    Unlike label_vol_adjusted_horizon, the forward return is compared to a
    fixed percentage threshold, not normalized by trailing volatility -- a
    plainer "is price meaningfully higher/lower `horizon` days from now"
    regime call, labeling the segment rather than an individual entry/exit
    timing decision. Same buy=-1/sell=1/hold=0 convention and same
    bounded-lookahead purging (_full_horizon_mask) as the other labels here.
    """

    close = df[close_col].astype(float).to_numpy()
    n = len(df)

    fwd_ret = np.full(n, np.nan)
    valid_end = n - horizon
    if valid_end > 0:
        fwd_ret[:valid_end] = close[horizon:] / close[:valid_end] - 1.0

    labels = np.full(n, np.nan)
    valid = ~np.isnan(fwd_ret)
    labels[valid & (fwd_ret > threshold_pct)] = -1.0
    labels[valid & (fwd_ret < -threshold_pct)] = 1.0
    labels[valid & (np.abs(fwd_ret) <= threshold_pct)] = 0.0

    return pd.Series(labels, index=df.index, name="trend_regime_target")


def cusum_event_mask(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    atr_window: int = 14,
    h_mult: float = 1.0,
) -> np.ndarray:
    """Symmetric CUSUM filter (Lopez de Prado, AFML sec 2.5).

    Marks a day as an "event" once the cumulative price drift since the last
    event exceeds ``h_mult`` * ATR[t] (a volatility-adaptive threshold) in
    either direction, then resets. Intended to select the subset of "days
    something actually happened" for training/validation, instead of every
    day including quiet, uninformative ones. Only ever looks at close[t] and
    close[t-1] plus the running CUSUM state -- fully causal.
    """

    close = df[close_col].astype(float).to_numpy()
    atr = average_true_range(
        df[high_col].astype(float), df[low_col].astype(float), df[close_col].astype(float), window=atr_window
    ).to_numpy()

    n = len(df)
    events = np.zeros(n, dtype=bool)
    s_pos, s_neg = 0.0, 0.0
    for t in range(1, n):
        if np.isnan(atr[t]) or atr[t] <= 0:
            continue
        diff = close[t] - close[t - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        threshold = h_mult * atr[t]
        if s_pos > threshold or s_neg < -threshold:
            events[t] = True
            s_pos = 0.0
            s_neg = 0.0
    return events
