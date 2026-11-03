# legacy_v1 copy of src/indicators/ohlcv_features.py (frozen v1 recipe). Only import paths were changed.
"""Hand-derived OHLCV features not covered by the `ta`-library indicator set.

The 20 indicators in src/indicators/ta_features.py are almost all momentum/
trend oscillators recombining a ticker's own price. They carry little
information about volatility regime, volume surprise, overnight gaps, or
distance from a rolling high -- this module fills those specific gaps rather
than adding more correlated oscillator variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_FEATURE_COLUMNS = [
    "ohlcv_realized_vol_5",
    "ohlcv_realized_vol_20",
    "ohlcv_realized_vol_60",
    "ohlcv_vol_of_vol",
    "ohlcv_volume_zscore",
    "ohlcv_gap",
    "ohlcv_drawdown_from_252d_high",
    "ohlcv_streak",
]


def build_ohlcv_features(
    df: pd.DataFrame,
    open_col: str = "Open",
    close_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """Return ``df`` with 8 hand-derived OHLCV feature columns appended.

    All columns are plain causal rolling-window formulas (no future rows
    touched), same as build_ta_features -- but note the 252-day rolling
    windows (vol_60, drawdown_from_252d_high) need far more warmup rows
    than any of the 20 ta indicators.
    """

    required = [open_col, close_col, volume_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"build_ohlcv_features: missing required columns: {missing}")

    open_ = df[open_col].astype(float)
    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)

    result = df.copy()
    log_ret = np.log(close / close.shift(1))

    vol5 = log_ret.rolling(5).std() * np.sqrt(252.0)
    vol20 = log_ret.rolling(20).std() * np.sqrt(252.0)
    vol60 = log_ret.rolling(60).std() * np.sqrt(252.0)
    result["ohlcv_realized_vol_5"] = vol5
    result["ohlcv_realized_vol_20"] = vol20
    result["ohlcv_realized_vol_60"] = vol60
    result["ohlcv_vol_of_vol"] = (vol20 - vol20.shift(10)) / vol20.shift(10).replace(0.0, np.nan)

    volume_mean20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std()
    result["ohlcv_volume_zscore"] = (volume - volume_mean20) / volume_std20.replace(0.0, np.nan)

    result["ohlcv_gap"] = (open_ - close.shift(1)) / close.shift(1)

    rolling_high_252 = close.rolling(252).max()
    result["ohlcv_drawdown_from_252d_high"] = (close - rolling_high_252) / rolling_high_252

    direction = np.sign(close.diff()).fillna(0.0).to_numpy()
    streak = np.zeros(len(direction))
    for i in range(1, len(direction)):
        d = direction[i]
        if d == 0.0:
            streak[i] = 0.0
        elif streak[i - 1] == 0.0 or np.sign(streak[i - 1]) == d:
            streak[i] = streak[i - 1] + d
        else:
            streak[i] = d
    result["ohlcv_streak"] = streak

    return result
