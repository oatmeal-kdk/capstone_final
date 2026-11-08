"""Trading rules on the ta indicators of ta_features.py, each with its parameter space for the GA.

Each rule turns one indicator into a -1 (buy) / 0 / +1 (sell) series; the GA (optimizer.py) searches its parameters for the best
turning-point fitness (fitness.py). Defaults: the ta library's periods (as in ta_features.py) and the textbook levels.

Most rules use the same reversal trigger, `band_cross`: buy on the day the indicator comes back UP through the lower level
(leaving the oversold zone), sell on the day it comes back DOWN through the upper level. It fires once per excursion, not on every
day spent beyond the level, and it waits for the turn, in the spirit of the paper's oblique lines on RSI.
  stoch  buy when %K crosses above %D with %K at or below the lower level (today or yesterday), sell the mirror (paper 2.5).
  adx    buy when +DI crosses above -DI while ADX >= adx_min, sell when -DI crosses above +DI (a trend-start rule).

14 rules cover 17 of the 20 ta_features columns (stoch: %K and %D; adx: ADX, +DI and -DI). Not covered, because they carry no
direction and cannot say buy or sell on their own: ta_bb_wband (band width), ta_atr_pct (ATR / close), ta_pvo (volume).
CANDLE_RULES adds the 5 candle patterns of the paper (section 2.6); ALL_RULES = the 19 rules together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import ta.momentum as ta_momentum
import ta.trend as ta_trend
import ta.volatility as ta_volatility
import ta.volume as ta_volume


@dataclass(frozen=True)
class Param:
    name: str
    low: float
    high: float
    default: float
    integer: bool = False


@dataclass(frozen=True)
class Rule:
    name: str
    columns: tuple[str, ...]                      # the ta_features columns the rule stands for
    params: tuple[Param, ...]
    signal: Callable[[pd.DataFrame, dict], np.ndarray]
    increasing: tuple[tuple[str, ...], ...] = field(default=())    # parameter groups that must be strictly increasing

    def default(self) -> list[float]:
        return [p.default for p in self.params]

    def repair(self, genes: Sequence[float]) -> list[float]:
        """Clip into the bounds, round the integer genes, and restore the required order (e.g. short < middle < long window)."""
        out = []
        for p, g in zip(self.params, genes):
            g = float(np.clip(g, p.low, p.high))
            out.append(float(round(g)) if p.integer else g)
        names = [p.name for p in self.params]
        for group in self.increasing:                       # integer windows: each at least 1 above the one before
            pos = [names.index(n) for n in group]
            for a, b in zip(pos, pos[1:]):                  # push up, within each gene's own upper bound
                out[b] = min(max(out[b], out[a] + 1), self.params[b].high)
            for a, b in reversed(list(zip(pos, pos[1:]))):  # if an upper bound stopped that, push the lower ones down
                out[a] = max(min(out[a], out[b] - 1), self.params[a].low)
        return out

    def as_dict(self, genes: Sequence[float]) -> dict:
        return {p.name: (int(g) if p.integer else round(float(g), 4)) for p, g in zip(self.params, self.repair(genes))}


def band_cross(x: pd.Series, lower: float, upper: float) -> np.ndarray:
    """-1 on the day x comes back up through `lower`, +1 on the day it comes back down through `upper`, else 0."""
    prev = x.shift(1)
    buy = (prev < lower) & (x >= lower)
    sell = (prev > upper) & (x <= upper)
    return np.where(buy, -1, np.where(sell, 1, 0)).astype(int)


def _hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    return df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)


def _rsi(df, p):
    return band_cross(ta_momentum.rsi(df["Close"].astype(float), window=p["window"]), p["lower"], p["upper"])


def _stoch(df, p):
    h, l, c = _hlc(df)
    k = ta_momentum.stoch(h, l, c, window=p["window"], smooth_window=p["smooth"])
    d = ta_momentum.stoch_signal(h, l, c, window=p["window"], smooth_window=p["smooth"])
    pk, pd_ = k.shift(1), d.shift(1)
    buy = (pk <= pd_) & (k > d) & ((k <= p["lower"]) | (pk <= p["lower"]))
    sell = (pk >= pd_) & (k < d) & ((k >= p["upper"]) | (pk >= p["upper"]))
    return np.where(buy, -1, np.where(sell, 1, 0)).astype(int)


def _stochrsi(df, p):
    x = ta_momentum.stochrsi_k(df["Close"].astype(float), window=p["window"], smooth1=p["smooth1"], smooth2=p["smooth2"])
    return band_cross(x, p["lower"], p["upper"])


def _williams_r(df, p):
    h, l, c = _hlc(df)
    return band_cross(ta_momentum.williams_r(h, l, c, lbp=p["window"]), p["lower"], p["upper"])


def _roc(df, p):
    return band_cross(ta_momentum.roc(df["Close"].astype(float), window=p["window"]), p["lower"], p["upper"])


def _ultimate(df, p):
    h, l, c = _hlc(df)
    x = ta_momentum.ultimate_oscillator(h, l, c, window1=p["window1"], window2=p["window2"], window3=p["window3"])
    return band_cross(x, p["lower"], p["upper"])


def _adx(df, p):
    h, l, c = _hlc(df)
    adx = ta_trend.adx(h, l, c, window=p["window"])
    pos, neg = ta_trend.adx_pos(h, l, c, window=p["window"]), ta_trend.adx_neg(h, l, c, window=p["window"])
    diff, prev = pos - neg, (pos - neg).shift(1)
    strong = adx >= p["adx_min"]
    buy = (prev <= 0) & (diff > 0) & strong
    sell = (prev >= 0) & (diff < 0) & strong
    return np.where(buy, -1, np.where(sell, 1, 0)).astype(int)


def _cci(df, p):
    h, l, c = _hlc(df)
    return band_cross(ta_trend.cci(h, l, c, window=p["window"]), p["lower"], p["upper"])


def _trix(df, p):
    return band_cross(ta_trend.trix(df["Close"].astype(float), window=p["window"]), p["lower"], p["upper"])


def _aroon(df, p):
    h, l, _ = _hlc(df)
    x = ta_trend.aroon_up(h, l, window=p["window"]) - ta_trend.aroon_down(h, l, window=p["window"])
    return band_cross(x, p["lower"], p["upper"])


def _bollinger(df, p):
    x = ta_volatility.bollinger_pband(df["Close"].astype(float), window=p["window"], window_dev=p["dev"])
    return band_cross(x, p["lower"], p["upper"])


def _keltner(df, p):
    h, l, c = _hlc(df)
    return band_cross(ta_volatility.keltner_channel_pband(h, l, c, window=p["window"]), p["lower"], p["upper"])


def _mfi(df, p):
    h, l, c = _hlc(df)
    x = ta_volume.money_flow_index(h, l, c, df["Volume"].astype(float), window=p["window"])
    return band_cross(x, p["lower"], p["upper"])


def _cmf(df, p):
    h, l, c = _hlc(df)
    x = ta_volume.chaikin_money_flow(h, l, c, df["Volume"].astype(float), window=p["window"])
    return band_cross(x, p["lower"], p["upper"])


def _candles(df: pd.DataFrame) -> dict:
    """Body, wicks and colours of today's and yesterday's candles (ported from capstone2 src/indicators/candle.py)."""
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    body = (c - o).abs()
    prev_body = body.shift(1)
    valid = o.notna() & h.notna() & l.notna() & c.notna() & (h >= top) & (l <= bottom) & (body > 0)
    return {"o": o, "c": c, "body": body.mask(body == 0, 1e-12), "prev_body": prev_body.mask(prev_body == 0, 1e-12),
            "upper": h - top, "lower": bottom - l, "white": c > o, "black": c < o,
            "prev_white": c.shift(1) > o.shift(1), "prev_black": c.shift(1) < o.shift(1),
            "valid": valid, "prev_valid": valid.shift(1, fill_value=False)}


def _candle_hammer(df, p):
    """Hammer / hanging man: lower wick >= a x body, longer than the upper wick and over half of both wicks. White -1, black +1."""
    k = _candles(df)
    shape = (k["lower"] / k["body"] >= p["a"]) & (k["lower"] > k["upper"]) & (k["lower"] > (k["lower"] + k["upper"]) / 2) & k["valid"]
    return np.where(shape & k["white"], -1, np.where(shape & k["black"], 1, 0)).astype(int)


def _candle_dark_cloud(df, p):
    """Dark cloud cover: white then black, open >= yesterday's close, close at least b x yesterday's body below yesterday's
    close but above yesterday's open. +1."""
    k = _candles(df)
    o, c = k["o"], k["c"]
    hit = (k["prev_white"] & k["black"] & (o >= c.shift(1)) & (c <= c.shift(1) - p["b"] * k["prev_body"]) & (c > o.shift(1))
           & k["valid"] & k["prev_valid"])
    return np.where(hit, 1, 0).astype(int)


def _candle_piercing(df, p):
    """Piercing line: black then white, open <= yesterday's close, close at least c x yesterday's body above yesterday's close
    but below yesterday's open. -1."""
    k = _candles(df)
    o, c = k["o"], k["c"]
    hit = (k["prev_black"] & k["white"] & (o <= c.shift(1)) & (c >= c.shift(1) + p["c"] * k["prev_body"]) & (c < o.shift(1))
           & k["valid"] & k["prev_valid"])
    return np.where(hit, -1, 0).astype(int)


def _candle_bullish_engulfing(df, p):
    """Bullish engulfing: black then white, open <= yesterday's close - d x body, close >= yesterday's open + e x body. -1."""
    k = _candles(df)
    o, c = k["o"], k["c"]
    hit = (k["prev_black"] & k["white"] & (o <= c.shift(1) - p["d"] * k["prev_body"]) & (c >= o.shift(1) + p["e"] * k["prev_body"])
           & k["valid"] & k["prev_valid"])
    return np.where(hit, -1, 0).astype(int)


def _candle_bearish_engulfing(df, p):
    """Bearish engulfing: white then black, open >= yesterday's close + f x body, close <= yesterday's open - g x body. +1."""
    k = _candles(df)
    o, c = k["o"], k["c"]
    hit = (k["prev_white"] & k["black"] & (o >= c.shift(1) + p["f"] * k["prev_body"]) & (c <= o.shift(1) - p["g"] * k["prev_body"])
           & k["valid"] & k["prev_valid"])
    return np.where(hit, 1, 0).astype(int)


W = lambda low, high, default, name="window": Param(name, low, high, default, integer=True)  # noqa: E731

RULES: dict[str, Rule] = {r.name: r for r in [
    Rule("rsi", ("ta_rsi",), (W(5, 30, 14), Param("lower", 10, 45, 30), Param("upper", 55, 90, 70)), _rsi),
    Rule("stoch", ("ta_stoch_k", "ta_stoch_d"),
         (W(5, 30, 14), W(2, 10, 3, "smooth"), Param("lower", 5, 40, 20), Param("upper", 60, 95, 80)), _stoch),
    Rule("stochrsi", ("ta_stochrsi_k",),
         (W(5, 30, 14), W(2, 10, 3, "smooth1"), W(2, 10, 3, "smooth2"), Param("lower", 0.0, 0.4, 0.2), Param("upper", 0.6, 1.0, 0.8)),
         _stochrsi),
    Rule("williams_r", ("ta_williams_r",), (W(5, 30, 14), Param("lower", -95, -60, -80), Param("upper", -40, -5, -20)), _williams_r),
    Rule("roc", ("ta_roc",), (W(3, 30, 12), Param("lower", -15, -0.5, -5), Param("upper", 0.5, 15, 5)), _roc),
    Rule("ultimate", ("ta_ultimate_osc",),
         (W(3, 15, 7, "window1"), W(7, 30, 14, "window2"), W(14, 60, 28, "window3"), Param("lower", 20, 45, 30), Param("upper", 55, 80, 70)),
         _ultimate, increasing=(("window1", "window2", "window3"),)),
    Rule("adx", ("ta_adx", "ta_adx_pos", "ta_adx_neg"), (W(5, 30, 14), Param("adx_min", 10, 40, 20)), _adx),
    Rule("cci", ("ta_cci",), (W(5, 50, 20), Param("lower", -250, -50, -100), Param("upper", 50, 250, 100)), _cci),
    Rule("trix", ("ta_trix",), (W(5, 40, 15), Param("lower", -0.4, 0.0, -0.1), Param("upper", 0.0, 0.4, 0.1)), _trix),
    Rule("aroon", ("ta_aroon_osc",), (W(10, 50, 25), Param("lower", -95, -20, -50), Param("upper", 20, 95, 50)), _aroon),
    Rule("bollinger", ("ta_bb_pband",),
         (W(10, 50, 20), Param("dev", 1.0, 3.0, 2.0), Param("lower", -0.3, 0.3, 0.0), Param("upper", 0.7, 1.3, 1.0)), _bollinger),
    Rule("keltner", ("ta_keltner_pband",), (W(10, 50, 20), Param("lower", -0.3, 0.3, 0.0), Param("upper", 0.7, 1.3, 1.0)), _keltner),
    Rule("mfi", ("ta_mfi",), (W(5, 30, 14), Param("lower", 5, 40, 20), Param("upper", 60, 95, 80)), _mfi),
    Rule("cmf", ("ta_cmf",), (W(5, 50, 20), Param("lower", -0.3, 0.0, -0.1), Param("upper", 0.0, 0.4, 0.1)), _cmf),
]}

# Candle patterns (paper 2.6, capstone2 src/indicators/candle.py), one rule per pattern with only the parameters it uses.
# Bounds as in capstone2; defaults: hammer wick 2x body, the textbook half-body penetration, and no extra engulfing margin.
CANDLE_RULES: dict[str, Rule] = {r.name: r for r in [
    Rule("candle_hammer", ("candle_hammer_hanging_man",), (Param("a", 1.0, 5.0, 2.0),), _candle_hammer),
    Rule("candle_dark_cloud", ("candle_dark_cloud_cover",), (Param("b", 0.0, 1.0, 0.5),), _candle_dark_cloud),
    Rule("candle_piercing", ("candle_piercing_line",), (Param("c", 0.0, 1.0, 0.5),), _candle_piercing),
    Rule("candle_bullish_engulfing", ("candle_bullish_engulfing",), (Param("d", 0.0, 0.5, 0.0), Param("e", 0.0, 0.5, 0.0)),
         _candle_bullish_engulfing),
    Rule("candle_bearish_engulfing", ("candle_bearish_engulfing",), (Param("f", 0.0, 0.5, 0.0), Param("g", 0.0, 0.5, 0.0)),
         _candle_bearish_engulfing),
]}

ALL_RULES: dict[str, Rule] = {**RULES, **CANDLE_RULES}

NOT_COVERED = ("ta_bb_wband", "ta_atr_pct", "ta_pvo")
