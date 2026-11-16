"""ESN inputs of the paper (Lin, Yang & Song 2011, section 3): the -1/0/+1 signals of the GA-tuned trading rules.

For one walk-forward fold, every rule of ALL_RULES (14 indicator rules + 5 candle patterns) gets its parameters from the GA
(optimizer.py) by the turning-point fitness (fitness.py) on the training window's CPM turning points only; then all rules are run
with those parameters from the training start to the test end. The indicators are causal (rolling windows), so a signal of
day t uses prices up to day t only; the parameters use the training window only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import ta.momentum as ta_momentum
import ta.trend as ta_trend
import ta.volatility as ta_volatility
import ta.volume as ta_volume

from legacy_v1.ga.fitness import turning_point_fitness
from legacy_v1.ga.optimizer import GAConfig, optimize
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS
from legacy_v1.ga.rules import ALL_RULES, Rule
from legacy_v1.ga.rules_v2 import RULES_V2, keltner_atr, ma_gap


def tune_rule(rule: Rule, frame: pd.DataFrame, close: np.ndarray, points: Sequence[dict], config: GAConfig) -> tuple[dict, np.ndarray]:
    """GA on one rule: (result of optimize, the signal of the best parameters over `frame`). Point indices refer to `frame` rows."""
    cache: dict[tuple, np.ndarray] = {}

    def signal(params: dict) -> np.ndarray:
        key = tuple(params.values())
        if key not in cache:
            cache[key] = np.asarray(rule.signal(frame, params), dtype=float)
        return cache[key]

    result = optimize(rule, lambda params: turning_point_fitness(close, signal(params), points)["normalized"], config)
    return result, signal(result["params"])


def _stored(rule_params: dict[str, dict], names: Sequence[str], rule_set: dict[str, Rule] = ALL_RULES) -> dict[str, dict]:
    """{rule: params} for `names`, read by parameter name and clipped / rounded as in the GA."""
    missing = [name for name in names if name not in rule_params]
    if missing:
        raise KeyError(f"stored parameters lack rules {missing}")
    return {name: rule_set[name].as_dict([rule_params[name][p.name] for p in rule_set[name].params]) for name in names}


def signals_from_params(df: pd.DataFrame, train_start: int, end: int, rule_params: dict[str, dict],
                        rules: Sequence[str] | None = None, rule_set: dict[str, Rule] = ALL_RULES) -> np.ndarray:
    """Signals of rows train_start .. end-1 from stored parameters ({rule: params}, e.g. one fold of data/params/<ticker>.json),
    one column per rule in the order of `rules` (default: every rule of `rule_set`, ALL_RULES or rules_v2.RULES_V2).
    Rows from `end` on are never read."""
    frame = df.iloc[:end]
    names = list(rules or rule_set)
    params = _stored(rule_params, names, rule_set)                  # by name, then clipped / rounded as in the GA
    return np.column_stack([np.asarray(rule_set[name].signal(frame, params[name]), dtype=float)[train_start:end] for name in names])


def ta_values_from_params(df: pd.DataFrame, train_start: int, end: int, rule_params: dict[str, dict]) -> np.ndarray:
    """The 20 ta indicator values (TA_FEATURE_COLUMNS order, raw, unscaled) of rows train_start .. end-1, computed with the
    indicator parameters the GA found for the 14 indicator rules (windows, smoothing, Bollinger dev); the rules' buy / sell
    thresholds are not used. ta_bb_wband takes the bollinger rule's window and dev; ta_atr_pct (14), ta_pvo (26 / 12 / 9) and the
    Keltner ATR window (10) have no rule and keep their defaults. Rows from `end` on are never read. Long GA windows can leave
    the first rows of the series NaN (warm-up)."""
    p = _stored(rule_params, ["rsi", "stoch", "stochrsi", "williams_r", "roc", "ultimate", "adx", "cci", "trix", "aroon",
                              "bollinger", "keltner", "mfi", "cmf"])
    frame = df.iloc[:end]
    h, l, c, v = (frame[col].astype(float) for col in ("High", "Low", "Close", "Volume"))
    out = {
        "ta_rsi": ta_momentum.rsi(c, window=p["rsi"]["window"]),
        "ta_stoch_k": ta_momentum.stoch(h, l, c, window=p["stoch"]["window"], smooth_window=p["stoch"]["smooth"]),
        "ta_stoch_d": ta_momentum.stoch_signal(h, l, c, window=p["stoch"]["window"], smooth_window=p["stoch"]["smooth"]),
        "ta_stochrsi_k": ta_momentum.stochrsi_k(c, window=p["stochrsi"]["window"], smooth1=p["stochrsi"]["smooth1"],
                                               smooth2=p["stochrsi"]["smooth2"]),
        "ta_williams_r": ta_momentum.williams_r(h, l, c, lbp=p["williams_r"]["window"]),
        "ta_roc": ta_momentum.roc(c, window=p["roc"]["window"]),
        "ta_ultimate_osc": ta_momentum.ultimate_oscillator(h, l, c, window1=p["ultimate"]["window1"], window2=p["ultimate"]["window2"],
                                                           window3=p["ultimate"]["window3"]),
        "ta_adx": ta_trend.adx(h, l, c, window=p["adx"]["window"]),
        "ta_adx_pos": ta_trend.adx_pos(h, l, c, window=p["adx"]["window"]),
        "ta_adx_neg": ta_trend.adx_neg(h, l, c, window=p["adx"]["window"]),
        "ta_cci": ta_trend.cci(h, l, c, window=p["cci"]["window"]),
        "ta_trix": ta_trend.trix(c, window=p["trix"]["window"]),
        "ta_aroon_osc": ta_trend.aroon_up(h, l, window=p["aroon"]["window"]) - ta_trend.aroon_down(h, l, window=p["aroon"]["window"]),
        "ta_bb_pband": ta_volatility.bollinger_pband(c, window=p["bollinger"]["window"], window_dev=p["bollinger"]["dev"]),
        "ta_bb_wband": ta_volatility.bollinger_wband(c, window=p["bollinger"]["window"], window_dev=p["bollinger"]["dev"]),
        "ta_atr_pct": ta_volatility.average_true_range(h, l, c, window=14) / c.replace(0, np.nan),
        "ta_keltner_pband": ta_volatility.keltner_channel_pband(h, l, c, window=p["keltner"]["window"], window_atr=10),
        "ta_mfi": ta_volume.money_flow_index(h, l, c, v, window=p["mfi"]["window"]),
        "ta_cmf": ta_volume.chaikin_money_flow(h, l, c, v, window=p["cmf"]["window"]),
        "ta_pvo": ta_momentum.pvo(v, window_slow=26, window_fast=12, window_sign=9),
    }
    return np.column_stack([np.asarray(out[col], dtype=float)[train_start:end] for col in TA_FEATURE_COLUMNS])


TA_V2_COLUMNS = [
    "ta_rsi", "ta_stoch_k", "ta_stoch_d", "ta_stochrsi_k", "ta_williams_r", "ta_roc_short", "ta_roc_long", "ta_ultimate_osc",
    "ta_adx", "ta_adx_pos", "ta_adx_neg", "ta_cci", "ta_trix", "ta_aroon_osc", "ta_ma_gap", "ta_ma_dev",
    "ta_bb_pband", "ta_bb_wband", "ta_atr_pct", "ta_keltner_pband",
    "ta_mfi", "ta_cmf", "ta_pvo",
]


def ta_values_v2(df: pd.DataFrame, train_start: int, end: int, rule_params: dict[str, dict]) -> np.ndarray:
    """The 23 indicator values of TA_V2_COLUMNS (raw, unscaled) of rows train_start .. end-1, computed with the periods the GA
    found for the rules_v2 rules; thresholds are not used. Against v1's 20: ROC as the roc rule's short and long ROC, plus
    ta_ma_gap (ma_cross: (MA long - MA short) / MA long in %) and ta_ma_dev (ma_envelope: close / MA - 1 in %); the Keltner
    position and ATR % from the squeeze rule (ATR-based channel); ta_bb_wband from the bollinger rule; ta_pvo at 26 / 12 / 9.
    Rows from `end` on are never read; long windows leave the first rows NaN (warm-up)."""
    p = _stored(rule_params, [n for n in RULES_V2 if n != "candle"], RULES_V2)
    frame = df.iloc[:end]
    h, l, c, v = (frame[col].astype(float) for col in ("High", "Low", "Close", "Volume"))
    env = c.rolling(p["ma_envelope"]["window"]).mean()
    _, kc_lo, kc_up, atr = keltner_atr(frame, p["squeeze"]["window"], p["squeeze"]["atr_window"], p["squeeze"]["mult"])
    u = p["ultimate"]
    out = {
        "ta_rsi": ta_momentum.rsi(c, window=p["rsi"]["window"]),
        "ta_stoch_k": ta_momentum.stoch(h, l, c, window=p["stoch"]["window"], smooth_window=p["stoch"]["smooth"]),
        "ta_stoch_d": ta_momentum.stoch_signal(h, l, c, window=p["stoch"]["window"], smooth_window=p["stoch"]["smooth"]),
        "ta_stochrsi_k": ta_momentum.stochrsi_k(c, window=p["stochrsi"]["window"], smooth1=p["stochrsi"]["smooth1"],
                                               smooth2=p["stochrsi"]["smooth2"]),
        "ta_williams_r": ta_momentum.williams_r(h, l, c, lbp=p["williams_r"]["window"]),
        "ta_roc_short": ta_momentum.roc(c, window=p["roc"]["short"]),
        "ta_roc_long": ta_momentum.roc(c, window=p["roc"]["long"]),
        "ta_ultimate_osc": ta_momentum.ultimate_oscillator(h, l, c, window1=u["window1"], window2=u["window2"], window3=u["window3"]),
        "ta_adx": ta_trend.adx(h, l, c, window=p["adx"]["window"]),
        "ta_adx_pos": ta_trend.adx_pos(h, l, c, window=p["adx"]["window"]),
        "ta_adx_neg": ta_trend.adx_neg(h, l, c, window=p["adx"]["window"]),
        "ta_cci": ta_trend.cci(h, l, c, window=p["cci"]["window"]),
        "ta_trix": ta_trend.trix(c, window=p["trix"]["window"]),
        "ta_aroon_osc": ta_trend.aroon_up(h, l, window=p["aroon"]["window"]) - ta_trend.aroon_down(h, l, window=p["aroon"]["window"]),
        "ta_ma_gap": ma_gap(c, p["ma_cross"]["short"], p["ma_cross"]["long"]),
        "ta_ma_dev": (c / env - 1) * 100,
        "ta_bb_pband": ta_volatility.bollinger_pband(c, window=p["bollinger"]["window"], window_dev=p["bollinger"]["dev"]),
        "ta_bb_wband": ta_volatility.bollinger_wband(c, window=p["bollinger"]["window"], window_dev=p["bollinger"]["dev"]),
        "ta_atr_pct": atr / c.replace(0, np.nan),
        "ta_keltner_pband": (c - kc_lo) / (kc_up - kc_lo).replace(0, np.nan),
        "ta_mfi": ta_volume.money_flow_index(h, l, c, v, window=p["mfi"]["window"]),
        "ta_cmf": ta_volume.chaikin_money_flow(h, l, c, v, window=p["cmf"]["window"]),
        "ta_pvo": ta_momentum.pvo(v, window_slow=26, window_fast=12, window_sign=9),
    }
    return np.column_stack([np.asarray(out[col], dtype=float)[train_start:end] for col in TA_V2_COLUMNS])


def ga_signal_inputs(df: pd.DataFrame, train_start: int, end: int, points: Sequence[dict], config: GAConfig,
                     rules: Sequence[str] | None = None) -> tuple[np.ndarray, dict]:
    """(signals of rows train_start .. end-1, one column per rule in the order of `rules`; {rule: {params, train_fitness,
    default_fitness}}). `points` are the training window's turning points (indices into df); rows from `end` on are never read."""
    frame = df.iloc[:end]
    close = frame["Close"].astype(float).to_numpy()
    names = list(rules or ALL_RULES)
    columns, info = [], {}
    for name in names:
        result, signal = tune_rule(ALL_RULES[name], frame, close, points, config)
        columns.append(signal[train_start:end])
        info[name] = {"params": result["params"], "train_fitness": result["objective"], "default_fitness": result["default_objective"]}
    return np.column_stack(columns), info
