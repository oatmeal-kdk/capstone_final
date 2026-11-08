"""The 5 candle patterns of the paper (Lin, Yang & Song 2011, 2.6) as ESN input columns, added 2026-11-08.

Each column is the -1 (buy) / 0 / +1 (sell) signal of one pattern of legacy_v1/ga/rules.py CANDLE_RULES at its default parameters
(hammer wick 2x body, half-body penetration, no extra engulfing margin). A pattern on day t uses days t-1 and t only.
The columns are already in [-1, 1] and must NOT be min-max scaled: a sell-only pattern (0/+1) would turn its 0 into -1, a buy.
"""

from __future__ import annotations

import pandas as pd

from legacy_v1.ga.rules import CANDLE_RULES

CANDLE_COLUMNS = [f"{name}" for name in CANDLE_RULES]       # candle_hammer, candle_dark_cloud, candle_piercing, ...


def build_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the 5 candle pattern columns appended (needs Open, High, Low, Close)."""
    result = df.copy()
    for name, rule in CANDLE_RULES.items():
        result[name] = rule.signal(result, rule.as_dict(rule.default())).astype(float)
    return result
