# legacy_v1 copy of src/indicators/relative_series.py (frozen v1 recipe). Only import paths were changed.
"""Build a synthetic "relative price" OHLCV series: ticker vs. a benchmark.

Feeds the exact same downstream pipeline (build_ta_features, label_triple_barrier,
run_esn_ta_walkforward.py) that already works for absolute price series, just
with Close replaced by the ticker/benchmark price ratio -- so the resulting
signal answers "is this ticker outperforming its benchmark" instead of "is
this ticker going up".

A ratio has no real intraday high/low, so Open/High/Low are all set equal to
Close (this degrades range-dependent indicators like ATR/Stochastic to a
close-to-close proxy -- an accepted simplification, not a bug). Volume stays
the ticker's own volume, since volume-based indicators (MFI/CMF/PVO) should
reflect whether the ticker itself is being actively traded, not the benchmark.
"""

from __future__ import annotations

import pandas as pd


def build_relative_series(
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    date_col: str = "Date",
    close_col: str = "Close",
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """Return a synthetic OHLCV dataframe of ticker_close / benchmark_close."""

    merged = ticker_df[[date_col, close_col, volume_col]].merge(
        benchmark_df[[date_col, close_col]],
        on=date_col,
        how="inner",
        suffixes=("", "_bench"),
    )
    ratio = merged[close_col] / merged[f"{close_col}_bench"]

    # A handful of ta indicators (Keltner %B, CMF, ...) divide by (High - Low)
    # or similar range terms; an exact High == Low == Close ratio makes that
    # range identically zero every row, producing inf/degenerate columns. A
    # tiny relative spread (0.01%) avoids the exact-zero denominator while
    # staying negligible next to real day-to-day ratio movement.
    result = pd.DataFrame({
        date_col: merged[date_col],
        "Open": ratio,
        "High": ratio * 1.0001,
        "Low": ratio * 0.9999,
        "Close": ratio,
        volume_col: merged[volume_col],
    })
    return result
