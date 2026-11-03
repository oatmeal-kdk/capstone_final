# legacy_v1 copy of src/data/download.py (frozen v1 recipe). Only import paths were changed.
"""OHLCV data download using yfinance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from legacy_v1.utils.io import save_dataframe


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns when present."""

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns:
            names = [str(part) for part in col if part not in ("", None)]
            flattened.append(names[0] if names else "")
        df = df.copy()
        df.columns = flattened
    return df


def download_ohlcv(
    ticker: str,
    interval: str = "1d",
    start: str = "2010-01-01",
    end: str | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from yfinance and optionally save it to CSV."""

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}, interval={interval}")

    df = _flatten_yfinance_columns(df).reset_index()
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data missing required columns: {missing}")

    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

    if output_path is not None:
        save_dataframe(df, output_path)

    return df
