# legacy_v1 copy of src/pipeline/production_config.py (frozen v1 recipe). Only import paths were changed.
"""Locked production configuration.

The original design (Phase 9, see capstone_full_summary_v2.md artifact) had
each ticker auto-calibrate its own (window_scale, k, max_holding) via a
27-combo per-fold grid search, with a ticker-specific selection method
(single_fold / expanding_aggregate / expanding_aggregate_calmar). That
approach did not survive multi-seed robustness testing done in this
session: a plain fixed-parameter baseline beat the auto-calibrated version
in 0/6 SPY+GLD x 3-seed trials. The only change that reliably won (6/6,
later confirmed on 10/12 tickers total) was adding the volatility-targeted
risk overlay on top of the fixed-parameter baseline.

So every ticker below shares the exact same recipe -- there is no per-ticker
tuning left. ESN architecture hyperparameters (reservoir_size, spectral_radius,
etc.) were never part of the tuning question and still live in ESN_CONFIG in
run_esn_ta_walkforward.py / run_esn_ta_walkforward_auto.py.

Ticker universe (2026-09 revision): auto-loaded from every
data/universe/*_active.csv, each produced by a src/pipeline/screen_universe.py
run against one index (S&P 500, S&P 400 MidCap, ...). Only status=="active"
(Sharpe >= 0.40 on the fixed recipe, no overlay) ships -- "weak" (0.20-0.40)
tickers were dropped from the live universe entirely: only tickers with a
clearly meaningful signal are surfaced, not "thin but technically positive"
ones. To grow the universe, run screen_universe.py against a new ticker list
and drop its *_active.csv into data/universe/ -- no code change needed here.
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

TARGET_METHOD = "triple_barrier"
TRIPLE_BARRIER_K = 2.0
TRIPLE_BARRIER_MAX_HOLDING = 10
WINDOW_SCALE = 1.0  # ta_features.py default window sizes, unscaled

ENSEMBLE_SIZE = 10  # reservoir-seed ensemble; reduces the seed variance found 2026-09

RISK_OVERLAY_TARGET_VOL = 0.10  # swept {0.05,0.08,0.10,0.15,0.20} on SPY x 3 seeds; 10% peaked on Sharpe in 3/3
RISK_OVERLAY_REBALANCE_BAND = 0.10

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0005

# Relative-strength (vs a benchmark, usually SPY) robustness for the handful
# of tickers actually run through src/indicators/relative_series.py -- not
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
