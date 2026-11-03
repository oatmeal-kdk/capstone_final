# legacy_v1 copy of src/indicators/ta_features.py (frozen v1 recipe). Only import paths were changed.
"""Build 20 raw `ta`-library indicator features as an alternative ESN input set.

Unlike the GA-tuned indicators in ``src/indicators/`` (which collapse each
indicator into a signed buy/sell/hold signal), these are the *raw* continuous
outputs of standard technical-analysis indicators from the ``ta`` library.
Every indicator here was chosen because it is naturally bounded or expressed
as a percentage/ratio (RSI, oscillators, %B, ADX, etc.) rather than a raw
price/volume level (raw MACD, OBV, ATR, VWAP, ...), so it stays meaningful
across a 16-year price range instead of drifting with the price level. ATR
is the one exception among the classic price-scale indicators and is
divided by Close here to turn it into a relative (%) value.

These raw columns still need ``src/utils/scaling.py: MinMaxUnitScaler``
(fit on train only) before being fed to the ESN.
"""

from __future__ import annotations

import ta.momentum as ta_momentum
import ta.trend as ta_trend
import ta.volatility as ta_volatility
import ta.volume as ta_volume

with __import__("contextlib").redirect_stderr(__import__("io").StringIO()):
    import pandas as pd


TA_FEATURE_COLUMNS = [
    # momentum (7)
    "ta_rsi",
    "ta_stoch_k",
    "ta_stoch_d",
    "ta_stochrsi_k",
    "ta_williams_r",
    "ta_roc",
    "ta_ultimate_osc",
    # trend (6)
    "ta_adx",
    "ta_adx_pos",
    "ta_adx_neg",
    "ta_cci",
    "ta_trix",
    "ta_aroon_osc",
    # volatility (4)
    "ta_bb_pband",
    "ta_bb_wband",
    "ta_atr_pct",
    "ta_keltner_pband",
    # volume (3)
    "ta_mfi",
    "ta_cmf",
    "ta_pvo",
]


_DEFAULT_WINDOWS = {
    "rsi": 14,
    "stoch": 14,
    "stoch_smooth": 3,
    "stochrsi": 14,
    "stochrsi_smooth1": 3,
    "stochrsi_smooth2": 3,
    "williams_r": 14,
    "roc": 12,
    "uo_1": 7,
    "uo_2": 14,
    "uo_3": 28,
    "adx": 14,
    "cci": 20,
    "trix": 15,
    "aroon": 25,
    "bb": 20,
    "bb_dev": 2,
    "atr": 14,
    "keltner": 20,
    "keltner_atr": 10,
    "mfi": 14,
    "cmf": 20,
    "pvo_slow": 26,
    "pvo_fast": 12,
    "pvo_sign": 9,
}


def _scaled(name: str, window_scale: float, minimum: int = 2) -> int:
    return max(minimum, round(_DEFAULT_WINDOWS[name] * window_scale))


def build_ta_features(
    df: pd.DataFrame,
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    volume_col: str = "Volume",
    window_scale: float = 1.0,
) -> pd.DataFrame:
    """Return ``df`` with 20 raw (unscaled) `ta` indicator columns appended.

    ``window_scale`` multiplies every indicator's default window/period
    (rounded, floored at 2) so the whole feature set can be shifted
    shorter/longer as a single knob, e.g. for a small val-selected grid
    search instead of independently GA-tuning 20+ window parameters.
    """

    required = [high_col, low_col, close_col, volume_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"build_ta_features: missing required columns: {missing}")

    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    volume = df[volume_col].astype(float)

    s = lambda name: _scaled(name, window_scale)  # noqa: E731

    result = df.copy()

    # momentum
    result["ta_rsi"] = ta_momentum.rsi(close, window=s("rsi"))
    result["ta_stoch_k"] = ta_momentum.stoch(high, low, close, window=s("stoch"), smooth_window=s("stoch_smooth"))
    result["ta_stoch_d"] = ta_momentum.stoch_signal(high, low, close, window=s("stoch"), smooth_window=s("stoch_smooth"))
    result["ta_stochrsi_k"] = ta_momentum.stochrsi_k(
        close, window=s("stochrsi"), smooth1=s("stochrsi_smooth1"), smooth2=s("stochrsi_smooth2")
    )
    result["ta_williams_r"] = ta_momentum.williams_r(high, low, close, lbp=s("williams_r"))
    result["ta_roc"] = ta_momentum.roc(close, window=s("roc"))
    result["ta_ultimate_osc"] = ta_momentum.ultimate_oscillator(
        high, low, close, window1=s("uo_1"), window2=s("uo_2"), window3=s("uo_3")
    )

    # trend
    result["ta_adx"] = ta_trend.adx(high, low, close, window=s("adx"))
    result["ta_adx_pos"] = ta_trend.adx_pos(high, low, close, window=s("adx"))
    result["ta_adx_neg"] = ta_trend.adx_neg(high, low, close, window=s("adx"))
    result["ta_cci"] = ta_trend.cci(high, low, close, window=s("cci"))
    result["ta_trix"] = ta_trend.trix(close, window=s("trix"))
    result["ta_aroon_osc"] = ta_trend.aroon_up(high, low, window=s("aroon")) - ta_trend.aroon_down(
        high, low, window=s("aroon")
    )

    # volatility
    result["ta_bb_pband"] = ta_volatility.bollinger_pband(close, window=s("bb"), window_dev=_scaled("bb_dev", window_scale, minimum=1))
    result["ta_bb_wband"] = ta_volatility.bollinger_wband(close, window=s("bb"), window_dev=_scaled("bb_dev", window_scale, minimum=1))
    result["ta_atr_pct"] = ta_volatility.average_true_range(high, low, close, window=s("atr")) / close.replace(0, pd.NA)
    result["ta_keltner_pband"] = ta_volatility.keltner_channel_pband(
        high, low, close, window=s("keltner"), window_atr=s("keltner_atr")
    )

    # volume
    result["ta_mfi"] = ta_volume.money_flow_index(high, low, close, volume, window=s("mfi"))
    result["ta_cmf"] = ta_volume.chaikin_money_flow(high, low, close, volume, window=s("cmf"))
    result["ta_pvo"] = ta_momentum.pvo(volume, window_slow=s("pvo_slow"), window_fast=s("pvo_fast"), window_sign=s("pvo_sign"))

    return result
