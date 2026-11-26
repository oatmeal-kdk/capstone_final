"""Make the repository root importable (python -m pytest from the root works without it; plain pytest needs it)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """400 business days of a seeded random walk with consistent Open/High/Low/Close/Volume (no network, no data files)."""
    rng = np.random.default_rng(0)
    n = 400
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    dates = pd.bdate_range("2015-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})
