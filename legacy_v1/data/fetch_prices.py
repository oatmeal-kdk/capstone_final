"""Download daily prices with yfinance and write the pipeline's input files. Added 2026-11-26 for reproducibility.

    python -m legacy_v1.data.fetch_prices [--tickers SPY QQQ AAPL GLD] [--start 2010-01-01] [--end YYYY-MM-DD]

Writes, per ticker (the paths every script of this repository reads):
  data/processed/<ticker>/1d/<ticker>_clean.csv     Date, Open, High, Low, Close, Volume (download_ohlcv + preprocess_ohlcv)
  data/processed/<ticker>/1d/<ticker>_with_tp.csv   the same + turning_label, cpm_point_type

End date: without --end, the download stops at the last day of the ticker's stored GA parameters (the last fold's test_end in
data/params/<ticker>.json, see --match-params), so the walk-forward folds and the last test year are the ones the reported
results used. Tickers without a parameter file are downloaded up to today.

The two CPM columns are turning points of one CPM run on the whole downloaded series, with P and T chosen by the Pareto knee
of run_walkforward.py's CPM_GRID (as capstone2's prepare_ticker_data.py did). They see the whole period, so they are read only
by the look-ahead reference label (run_walkforward.py --target-method cpm --allow-lookahead) and by some plots; the default
label cpm_causal rebuilds its turning points per fold from the training rows and never reads them.

The capstone2 copies the reported results were computed on are not in the repository. Prices are not adjusted
(auto_adjust=False, as download_ohlcv), but Yahoo can revise past rows, so a fresh download may differ slightly from them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from legacy_v1.data.download import download_ohlcv
from legacy_v1.data.preprocess import preprocess_ohlcv
from legacy_v1.labels.cpm import add_turning_labels, grid_search_cpm, select_best_cpm_params
from legacy_v1.pipeline.run_walkforward import CPM_GRID
from legacy_v1.utils.io import save_dataframe


def stored_end(ticker: str, params_dir: Path) -> str | None:
    """The day after the last test day in <params_dir>/<ticker>.json (yfinance's end is exclusive), or None."""
    path = params_dir / f"{ticker}.json"
    if not path.exists():
        return None
    folds = json.loads(path.read_text(encoding="utf-8"))["folds"]
    last = max(pd.Timestamp(f["test_end"]) for f in folds.values())
    return str((last + pd.Timedelta(days=1)).date())


def fetch(ticker: str, start: str, end: str | None, root: Path) -> Path:
    clean = preprocess_ohlcv(download_ohlcv(ticker, interval="1d", start=start, end=end))
    folder = root / ticker / "1d"
    save_dataframe(clean, folder / f"{ticker}_clean.csv")
    best = select_best_cpm_params(grid_search_cpm(clean["Close"], CPM_GRID["p_values"], CPM_GRID["t_values"]),
                                  CPM_GRID["selection_method"])
    with_tp = add_turning_labels(clean, P=float(best["P"]), T=int(best["T"]))
    out = folder / f"{ticker}_with_tp.csv"
    save_dataframe(with_tp, out)
    print(f"{ticker}: {len(clean)} rows {clean['Date'].iloc[0]} ~ {clean['Date'].iloc[-1]}, "
          f"CPM P={best['P']} T={int(best['T'])} -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "GLD"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None, help="Exclusive end date for every ticker. Default: see --match-params.")
    parser.add_argument("--match-params", default="data/params",
                        help="Folder of save_ga_params.py files whose last test day sets the default end per ticker.")
    parser.add_argument("--root", default="data/processed")
    args = parser.parse_args()
    for ticker in args.tickers:
        fetch(ticker, args.start, args.end or stored_end(ticker, Path(args.match_params)), Path(args.root))


if __name__ == "__main__":
    main()
