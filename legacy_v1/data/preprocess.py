# legacy_v1 copy of src/data/preprocess.py (frozen v1 recipe). Only import paths were changed.
"""OHLCV preprocessing."""

from __future__ import annotations

import pandas as pd


def preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data for CPM and indicator pipelines."""

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {missing}")

    clean = df[required].copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean = clean.dropna(subset=["Date"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["Close"])
    clean = clean.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    clean = clean.reset_index(drop=True)
    clean["Date"] = clean["Date"].dt.strftime("%Y-%m-%d")
    return clean
