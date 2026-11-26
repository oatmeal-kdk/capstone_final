# legacy_v1 copy of src/pipeline/production_config.py (frozen v1 recipe). Only import paths were changed.
"""Fixed recipe shared by every ticker: walk-forward folds, label, costs, overlay, and the ticker universe.

History (capstone2, kept for context): an earlier design let each ticker auto-calibrate its own (window_scale, k,
max_holding) by a per-fold grid search. A plain fixed-parameter baseline beat it in 0/6 SPY+GLD x 3-seed trials, so every
ticker now shares the recipe below; the one change that reliably helped there was the volatility-targeted risk overlay.
ESN architecture hyperparameters (reservoir_size, spectral_radius, ...) are not set here: they live in ESN_CONFIG in
legacy_v1/pipeline/run_walkforward.py.

Ticker universe: loaded from every data/universe/*_screening_results*.csv (written by legacy_v1/pipeline/screen_universe.py,
one file per index: S&P 500, S&P 400, KOSPI, KOSDAQ). Each row has a status: "active" (Sharpe >= 0.40 on the fixed recipe,
no overlay), "weak" (0.20 - 0.40) or "excluded" (below 0.20), exposed as ACTIVE_TICKERS, WEAK_TICKERS and EXCLUDED_TICKERS.
To add an index, run screen_universe.py on its ticker list and put the resulting *_screening_results.csv in data/universe/;
no code change is needed here.

The experiment scripts take most settings from their own command-line options; they read this module for the fee and
slippage rates (FEE_RATE, SLIPPAGE_RATE), TRAIN_YEARS, the risk-overlay settings and the label settings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UNIVERSE_DIR = _PROJECT_ROOT / "data" / "universe"

# --- Universal recipe (identical for every active ticker) ---

TRAIN_YEARS = 5
VAL_YEARS = 1
STEP_YEARS = 1

TARGET_METHOD = "cpm_causal"  # since 2026-11-08 (was "triple_barrier"); P, T by the Pareto knee on each fold's training rows
TRIPLE_BARRIER_K = 2.0
TRIPLE_BARRIER_MAX_HOLDING = 10
WINDOW_SCALE = 1.0  # ta_features.py default window sizes, unscaled

ENSEMBLE_SIZE = 10  # reservoir-seed ensemble; reduces the seed variance found 2026-09

RISK_OVERLAY_TARGET_VOL = 0.10  # swept {0.05,0.08,0.10,0.15,0.20} on SPY x 3 seeds; 10% peaked on Sharpe in 3/3
RISK_OVERLAY_REBALANCE_BAND = 0.10

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

# Relative-strength (vs a benchmark, usually SPY) robustness for the handful
# of tickers actually run through legacy_v1/relative/relative_series.py -- not
# derivable from the active-ticker screening CSVs, so tracked here by hand.
# Anything not listed is simply untested, not confirmed absent.
RELATIVE_STRENGTH_STATUS: dict[str, str] = {
    "AAPL": "positive_robust",
    "MSFT": "positive_robust",
    "NVDA": "positive_robust",
    "XLK": "positive_robust",
    "CAT": "positive_robust",
    "GOOGL": "positive_weak",
    "JPM": "positive_weak",
    "JNJ": "inverted",
}


def _load_tickers_with_status(status: str) -> list[str]:
    frames = [pd.read_csv(p) for p in sorted(_UNIVERSE_DIR.glob("*_screening_results*.csv"))]
    if not frames:
        return []
    combined = pd.concat(frames, ignore_index=True)
    matched = combined[combined["status"] == status]["symbol"].dropna().unique().tolist()
    return sorted(matched)


ACTIVE_TICKERS: list[str] = _load_tickers_with_status("active")
WEAK_TICKERS: list[str] = _load_tickers_with_status("weak")
EXCLUDED_TICKERS: list[str] = _load_tickers_with_status("excluded")


def get_relative_strength_status(symbol: str) -> str:
    return RELATIVE_STRENGTH_STATUS.get(symbol, "not_tested")
