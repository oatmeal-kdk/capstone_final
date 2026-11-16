"""Trading rules v2 (added 2026-11-16): the rule forms of the paper (Lin, Yang & Song 2011, section 2) where it gives one, and
two-part rules for the other indicators. The v1 rules (rules.py) stay as they are, for comparison.

Each rule is one "expert" (paper section 3) that turns the prices into a -1 (buy) / 0 / +1 (sell) series; the GA (optimizer.py)
tunes its parameters by the turning-point fitness (fitness.py), as in v1.

The paper's systems (section 2):
  ma_cross     2.2.1 golden / dead cross with Baba's conditions (eqs. 8-11) on z = MA(long) - MA(short). Buy while short is
               below long (z > 0) once the gap has opened beyond b x c and closed back below min(M / a, c), M = the largest
               gap since the cross: the turn is bought before the golden cross itself. Sell is the mirror on -z with a', b', c'.
               Both windows are tuned (the paper's change to Baba). Adaptation: z in % of MA(long), so c does not depend on price.
  ma_envelope  2.2.2 band of +p1 % / -p2 % (p1 != p2 allowed) around MA(window). Buy when the close breaks above the upper band
               while the MA is falling (downtrend), sell when it breaks below the lower band while the MA is rising.
               Interpretation: "in downtrend" = the MA lower than the day before.
  rsi          2.3 RSI with oblique lines. When RSI rises above `upper`, a line starts there with slope p per day; sell when RSI
               falls below it. When RSI falls below `lower`, a line starts there with slope -q; buy when RSI rises above it.
               Interpretation (the figure is not in the text): the line starts at the level on the crossing day. p = 0 is the
               v1 rule (back below the level); p > 0 sells earlier, before RSI is back at the level.
  roc          2.4 divergence of a long and a short ROC. Sell when the long ROC is above `upper` (a new high) while the short ROC
               is near the equilibrium line (-band_down <= short <= band_up); buy when the long ROC is below `lower` with the
               short ROC near it. The paper's ROC is close / close(n) x 100 (equilibrium 100); `ta` gives the % change
               (equilibrium 0), the same up to the shift.
  stoch        2.5 buy when %K crosses above %D with %K < a and %K - %D < b; sell when %K crosses below %D with %K > c and
               %D - %K < d.
  candle       2.6 one system for the four patterns: hammer / hanging man, dark cloud cover, piercing line, engulfing. Buy when a
               bullish pattern appears after a downtrend, sell when a bearish one appears after an uptrend, 0 when none (or both).
               Trend = yesterday's close against its MA(trend). The trend decides hammer (after a fall, buy) against hanging
               man (after a rise, sell): the same shape (v1 split them by colour). Piercing / dark cloud use the paper's
               measure: |yesterday's open - today's close| < c (b) x yesterday's body. Engulfing margins d-g are in units of
               yesterday's body (the paper's are in price), so they do not depend on price.

Two-part rules for indicators the paper does not use:
  trix       TRIX crosses its signal line (EMA of TRIX), buy only at or below `lower`, sell only at or above `upper`.
  aroon      buy when Aroon Up crosses above Aroon Down while Down was still >= strong the day before (the low is recent, so
             the cross comes right after a bottom); sell is the mirror. (A first version, buy on the first day of Up >= strong
             and Down <= weak, confirmed the trend too late: SPY fold 2020 fitness 1.38 after the GA, worse than no signal.)
  bollinger  v1's %B re-cross, counted only when the band width is at least w_min (a touch of a narrow band means little).
  squeeze    the Bollinger band inside the Keltner channel (ATR-based) for at least min_days, then out: buy if the close is
             above the middle line, sell if below. Replaces v1's keltner rule.
  mfi        divergence: buy when the close makes a new `lookback`-day low while MFI stays at least `gap` above its own low of
             those days, and that low was oversold (<= lower); sell is the mirror.
  cmf        CMF crosses buy_level upward (sell_level downward), counted only when PVO >= pvo_min (volume growing).
Kept from v1 (single oscillators made for level reading): adx (already ADX + DI), williams_r, cci, stochrsi, ultimate.

Signals that come from a state (roc, aroon, mfi, ma_cross) fire on its first day only, like the crosses of the other rules.
Parameter ranges are not in the paper; they are set here around the usual values. The candle margins (a up to 10, d-g up to
1.5 bodies) and the CMF levels (+-0.4) are wide enough for the GA to switch a weak pattern or side off: in the first SPY check
the GA pushed them to the old bounds (a 5, g 0.5, CMF +-0.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta.momentum as ta_momentum
import ta.trend as ta_trend
import ta.volatility as ta_volatility
import ta.volume as ta_volume

from legacy_v1.ga.rules import RULES as V1, Param, Rule, W, _candles, _hlc, band_cross


def _edge(cond: pd.Series) -> pd.Series:
    """True on the first day of each run of `cond`."""
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def _signal(buy, sell) -> np.ndarray:
    b, s = np.asarray(buy, dtype=bool), np.asarray(sell, dtype=bool)
    return np.where(b & ~s, -1, np.where(s & ~b, 1, 0)).astype(int)


def _cross_up(x: pd.Series, y) -> pd.Series:
    return (x.shift(1) <= (y.shift(1) if isinstance(y, pd.Series) else y)) & (x > y)


def _cross_down(x: pd.Series, y) -> pd.Series:
    return (x.shift(1) >= (y.shift(1) if isinstance(y, pd.Series) else y)) & (x < y)


# ---------------------------------------------------------------- the paper's systems

def ma_gap(close: pd.Series, short: int, long: int) -> pd.Series:
    """z = (MA(long) - MA(short)) / MA(long) x 100: > 0 while the short MA is below the long one."""
    ma_long = close.rolling(long).mean()
    return (ma_long - close.rolling(short).mean()) / ma_long * 100


def _baba(z: pd.Series, a: float, b: float, c: float) -> pd.Series:
    """Days that meet eqs. 8-9 on z: z > 0 since the last upward zero cross, M = max z since then, M > b c and z < min(M / a, c)."""
    pos = z > 0
    episode = (pos & ~pos.shift(1, fill_value=False)).cumsum()
    m = z.where(pos).groupby(episode).cummax()
    return _edge(pos & (m > b * c) & (z < np.minimum(m / a, c)))


def _ma_cross(df, p):
    z = ma_gap(df["Close"].astype(float), p["short"], p["long"])
    return _signal(_baba(z, p["a"], p["b"], p["c"]), _baba(-z, p["sa"], p["sb"], p["sc"]))


def _ma_envelope(df, p):
    c = df["Close"].astype(float)
    ma = c.rolling(p["window"]).mean()
    upper, lower = ma * (1 + p["p1"] / 100), ma * (1 - p["p2"] / 100)
    return _signal(_cross_up(c, upper) & (ma < ma.shift(1)), _cross_down(c, lower) & (ma > ma.shift(1)))


def _rsi(df, p):
    x = ta_momentum.rsi(df["Close"].astype(float), window=p["window"]).to_numpy()
    out = np.zeros(len(x), dtype=int)
    top = bottom = None                              # day the current oblique line started
    for t in range(1, len(x)):
        if np.isnan(x[t]) or np.isnan(x[t - 1]):
            continue
        if top is not None and x[t] < p["upper"] + p["p"] * (t - top):
            out[t], top = 1, None
        elif bottom is not None and x[t] > p["lower"] - p["q"] * (t - bottom):
            out[t], bottom = -1, None
        if x[t - 1] <= p["upper"] < x[t]:
            top, bottom = t, None
        elif x[t - 1] >= p["lower"] > x[t]:
            bottom, top = t, None
    return out


def _roc(df, p):
    c = df["Close"].astype(float)
    short, long = ta_momentum.roc(c, window=p["short"]), ta_momentum.roc(c, window=p["long"])
    near = (short <= p["band_up"]) & (short >= -p["band_down"])
    return _signal(_edge((long < p["lower"]) & near), _edge((long > p["upper"]) & near))


def _stoch(df, p):
    h, l, c = _hlc(df)
    k = ta_momentum.stoch(h, l, c, window=p["window"], smooth_window=p["smooth"])
    d = ta_momentum.stoch_signal(h, l, c, window=p["window"], smooth_window=p["smooth"])
    buy = _cross_up(k, d) & (k < p["a"]) & (k - d < p["b"])
    sell = _cross_down(k, d) & (k > p["c"]) & (d - k < p["d"])
    return _signal(buy, sell)


def candle_patterns(df: pd.DataFrame, p: dict) -> tuple[pd.Series, pd.Series]:
    """(bullish, bearish) days of the four patterns, before the trend condition."""
    k = _candles(df)
    o, c = k["o"], k["c"]
    o1, c1, body1 = o.shift(1), c.shift(1), k["prev_body"]
    both = k["valid"] & k["prev_valid"]
    hammer = (k["lower"] / k["body"] >= p["a"]) & (k["lower"] > k["upper"]) & (k["lower"] > (k["lower"] + k["upper"]) / 2) & k["valid"]
    piercing = k["prev_black"] & k["white"] & (o <= c1) & (c < o1) & ((o1 - c) < p["c"] * body1) & both
    dark_cloud = k["prev_white"] & k["black"] & (o >= c1) & (c > o1) & ((c - o1) < p["b"] * body1) & both
    bull_engulf = k["prev_black"] & k["white"] & (o <= c1 - p["d"] * body1) & (c >= o1 + p["e"] * body1) & both
    bear_engulf = k["prev_white"] & k["black"] & (o >= c1 + p["f"] * body1) & (c <= o1 - p["g"] * body1) & both
    return hammer | piercing | bull_engulf, hammer | dark_cloud | bear_engulf


def _candle(df, p):
    c = df["Close"].astype(float)
    ma = c.rolling(p["trend"]).mean()
    down, up = (c < ma).shift(1, fill_value=False), (c > ma).shift(1, fill_value=False)
    bullish, bearish = candle_patterns(df, p)
    return _signal(bullish & down, bearish & up)


# ---------------------------------------------------------------- two-part rules

def _trix(df, p):
    x = ta_trend.trix(df["Close"].astype(float), window=p["window"])
    s = x.ewm(span=p["signal"], adjust=False).mean()
    return _signal(_cross_up(x, s) & (x <= p["lower"]), _cross_down(x, s) & (x >= p["upper"]))


def _aroon(df, p):
    h, l, _ = _hlc(df)
    up, down = ta_trend.aroon_up(h, l, window=p["window"]), ta_trend.aroon_down(h, l, window=p["window"])
    return _signal(_cross_up(up, down) & (down.shift(1) >= p["strong"]), _cross_up(down, up) & (up.shift(1) >= p["strong"]))


def _bollinger(df, p):
    c = df["Close"].astype(float)
    x = ta_volatility.bollinger_pband(c, window=p["window"], window_dev=p["dev"])
    width = ta_volatility.bollinger_wband(c, window=p["window"], window_dev=p["dev"])
    out = band_cross(x, p["lower"], p["upper"])
    out[(width < p["w_min"]).to_numpy()] = 0
    return out


def keltner_atr(df: pd.DataFrame, window: int, atr_window: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """(middle = MA(window), lower, upper = middle -/+ mult x ATR(atr_window), ATR)."""
    h, l, c = _hlc(df)
    mid = c.rolling(window).mean()
    atr = ta_volatility.average_true_range(h, l, c, window=atr_window)
    return mid, mid - mult * atr, mid + mult * atr, atr


def _squeeze(df, p):
    c = df["Close"].astype(float)
    mid, kc_lo, kc_up, _ = keltner_atr(df, p["window"], p["atr_window"], p["mult"])
    sd = c.rolling(p["window"]).std(ddof=0)
    on = ((mid + p["dev"] * sd) < kc_up) & ((mid - p["dev"] * sd) > kc_lo)
    run = on.astype(int).groupby((~on).cumsum()).cumsum()           # consecutive squeeze days
    release = ~on & (run.shift(1) >= p["min_days"])
    return _signal(release & (c > mid), release & (c < mid))


def _mfi(df, p):
    h, l, c = _hlc(df)
    x = ta_volume.money_flow_index(h, l, c, df["Volume"].astype(float), window=p["window"])
    n = p["lookback"]
    low_c, high_c = c.shift(1).rolling(n).min(), c.shift(1).rolling(n).max()
    low_x, high_x = x.shift(1).rolling(n).min(), x.shift(1).rolling(n).max()
    buy = (c < low_c) & (low_x <= p["lower"]) & (x >= low_x + p["gap"])
    sell = (c > high_c) & (high_x >= p["upper"]) & (x <= high_x - p["gap"])
    return _signal(_edge(buy), _edge(sell))


def _cmf(df, p):
    h, l, c = _hlc(df)
    v = df["Volume"].astype(float)
    x = ta_volume.chaikin_money_flow(h, l, c, v, window=p["window"])
    growing = ta_momentum.pvo(v, window_slow=26, window_fast=12, window_sign=9) >= p["pvo_min"]
    return _signal(_cross_up(x, p["buy_level"]) & growing, _cross_down(x, p["sell_level"]) & growing)


RULES_V2: dict[str, Rule] = {r.name: r for r in [
    Rule("ma_cross", ("ta_ma_gap",),
         (W(3, 50, 10, "short"), W(20, 150, 50, "long"), Param("a", 1.0, 10.0, 2.0), Param("b", 0.0, 5.0, 1.0), Param("c", 0.5, 10.0, 2.0),
          Param("sa", 1.0, 10.0, 2.0), Param("sb", 0.0, 5.0, 1.0), Param("sc", 0.5, 10.0, 2.0)),
         _ma_cross, increasing=(("short", "long"),)),
    Rule("ma_envelope", ("ta_ma_dev",), (W(5, 60, 20), Param("p1", 0.5, 10.0, 3.0), Param("p2", 0.5, 10.0, 3.0)), _ma_envelope),
    Rule("rsi", ("ta_rsi",), (W(5, 30, 14), Param("lower", 10, 45, 30), Param("upper", 55, 90, 70),
                              Param("p", 0.0, 5.0, 1.0), Param("q", 0.0, 5.0, 1.0)), _rsi),
    Rule("roc", ("ta_roc_short", "ta_roc_long"),
         (W(3, 20, 10, "short"), W(20, 120, 60, "long"), Param("upper", 2.0, 40.0, 8.0), Param("lower", -40.0, -2.0, -8.0),
          Param("band_up", 0.0, 5.0, 2.0), Param("band_down", 0.0, 5.0, 2.0)),
         _roc, increasing=(("short", "long"),)),
    Rule("stoch", ("ta_stoch_k", "ta_stoch_d"),
         (W(5, 30, 14), W(2, 10, 3, "smooth"), Param("a", 5, 50, 20), Param("b", 1, 30, 15), Param("c", 50, 95, 80), Param("d", 1, 30, 15)),
         _stoch),
    Rule("candle", ("candle",),
         (Param("a", 1.0, 10.0, 2.0), Param("b", 0.0, 1.0, 0.5), Param("c", 0.0, 1.0, 0.5), Param("d", 0.0, 1.5, 0.0),
          Param("e", 0.0, 1.5, 0.0), Param("f", 0.0, 1.5, 0.0), Param("g", 0.0, 1.5, 0.0), W(3, 30, 10, "trend")),
         _candle),
    Rule("trix", ("ta_trix",), (W(5, 40, 15), W(3, 20, 9, "signal"), Param("lower", -0.4, 0.1, 0.0), Param("upper", -0.1, 0.4, 0.0)), _trix),
    Rule("aroon", ("ta_aroon_osc",), (W(10, 50, 25), Param("strong", 50, 100, 70)), _aroon),
    Rule("bollinger", ("ta_bb_pband", "ta_bb_wband"),
         (W(10, 50, 20), Param("dev", 1.0, 3.0, 2.0), Param("lower", -0.3, 0.3, 0.0), Param("upper", 0.7, 1.3, 1.0), Param("w_min", 0.0, 15.0, 0.0)),
         _bollinger),
    Rule("squeeze", ("ta_keltner_pband", "ta_atr_pct"),
         (W(10, 50, 20), Param("dev", 1.0, 3.0, 2.0), W(5, 30, 14, "atr_window"), Param("mult", 1.0, 3.0, 1.5), W(1, 20, 5, "min_days")),
         _squeeze),
    Rule("mfi", ("ta_mfi",), (W(5, 30, 14), W(5, 60, 20, "lookback"), Param("lower", 5, 45, 30), Param("upper", 55, 95, 70),
                              Param("gap", 0.0, 20.0, 5.0)), _mfi),
    Rule("cmf", ("ta_cmf", "ta_pvo"), (W(5, 50, 20), Param("buy_level", -0.4, 0.2, 0.0), Param("sell_level", -0.2, 0.4, 0.0),
                                       Param("pvo_min", -10.0, 20.0, 0.0)), _cmf),
    *(V1[name] for name in ("adx", "williams_r", "cci", "stochrsi", "ultimate")),
]}

PAPER_RULES = ("ma_cross", "ma_envelope", "rsi", "roc", "stoch", "candle")
