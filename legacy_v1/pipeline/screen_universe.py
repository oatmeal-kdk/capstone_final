# legacy_v1 copy of src/pipeline/screen_universe.py (frozen v1 recipe). Only import paths were changed.
"""Automated ticker screening against the locked production recipe.

For each ticker: download+preprocess OHLCV (skip if already cached), run the
fixed-parameter triple-barrier walk-forward backtest (src/pipeline/
run_esn_ta_walkforward.py, the recipe locked in production_config.py), and
classify as active/weak/excluded from the resulting Sharpe ratio.

The risk overlay is deliberately NOT run here -- it's a serving-time
position-sizing layer (see production_config.py's docstring/history), not a
viability test, so screening only needs the cheaper no-overlay backtest.

Thresholds were calibrated by hand against the ~25 tickers classified
earlier in this project (results/_universe_*, results/_gate_*): Sharpe < 0.20
-> excluded, 0.20-0.40 -> weak (kept but low confidence), >= 0.40 -> active.
This won't perfectly reproduce every earlier by-hand judgment call (e.g. XLE
was manually excluded on sector-level reasoning despite Sharpe 0.34, which
this rule alone would call "weak") -- that's an accepted trade-off for
running this unattended across hundreds of tickers instead of one at a time.

Usage:
    python -m legacy_v1.pipeline.screen_universe --tickers AAPL MSFT ...
    python -m legacy_v1.pipeline.screen_universe --csv data/raw/sp500_constituents.csv --symbol-col Symbol
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from legacy_v1.data.download import download_ohlcv
from legacy_v1.data.preprocess import preprocess_ohlcv
from legacy_v1.pipeline import production_config as cfg
from legacy_v1.utils.io import save_dataframe

MIN_ROWS = 1500  # need enough history for train(5y)+val(1y)+test(1y+) to form at least one fold


def screen_one(symbol: str, start: str = "2010-01-01", ensemble_size: int = 1) -> dict:
    tp_path = Path(f"data/processed/{symbol}/1d/{symbol}_with_tp.csv")
    if not tp_path.exists():
        try:
            raw = download_ohlcv(symbol, start=start)
        except Exception as exc:  # yfinance failures vary widely; keep screening the rest
            return {"symbol": symbol, "status": "error", "error": f"download: {exc}"}
        clean = preprocess_ohlcv(raw)
        if len(clean) < MIN_ROWS:
            return {"symbol": symbol, "status": "error", "error": f"only {len(clean)} rows (<{MIN_ROWS})"}
        save_dataframe(clean, tp_path)

    out_dir = Path(f"results/_screen/{symbol}")
    cmd = [
        sys.executable, "-m", "legacy_v1.pipeline.run_walkforward",
        "--tp-path", str(tp_path),
        "--target-method", "triple_barrier",
        "--train-years", str(cfg.TRAIN_YEARS),
        "--fee-rate", str(cfg.FEE_RATE),
        "--slippage-rate", str(cfg.SLIPPAGE_RATE),
        "--ensemble-size", str(ensemble_size),
        "--output-dir", str(out_dir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60 + 15 * max(1, ensemble_size))
    except subprocess.TimeoutExpired:
        return {"symbol": symbol, "status": "error", "error": "timeout"}

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return {"symbol": symbol, "status": "error", "error": result.stderr[-500:] or "no summary.json produced"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    strat = summary["strategy"]
    sharpe = float(strat["sharpe_ratio"])

    if sharpe < 0.20:
        status = "excluded"
    elif sharpe < 0.40:
        status = "weak"
    else:
        status = "active"

    return {
        "symbol": symbol,
        "status": status,
        "sharpe": sharpe,
        "mdd": float(strat["max_drawdown"]),
        "annualized_return": float(strat["annualized_return"]),
        "num_trades": int(strat["num_trades"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker list.")
    parser.add_argument("--csv", default=None, help="CSV with a ticker symbol column (e.g. S&P 500 constituents).")
    parser.add_argument("--symbol-col", default="Symbol")
    parser.add_argument("--output", default="results/screening_results.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only screen the first N tickers (smoke testing).")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Average predictions across this many reservoir seeds (reduces seed variance).")
    args = parser.parse_args()

    if args.tickers:
        symbols = args.tickers
    elif args.csv:
        symbols = pd.read_csv(args.csv)[args.symbol_col].dropna().unique().tolist()
    else:
        raise ValueError("Provide either --tickers or --csv.")
    if args.limit:
        symbols = symbols[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, symbol in enumerate(symbols, start=1):
        result = screen_one(symbol, ensemble_size=args.ensemble_size)
        rows.append(result)
        print(f"[{i}/{len(symbols)}] {symbol}: {result.get('status')} "
              f"(sharpe={result.get('sharpe', 'n/a')})", flush=True)
        # write incrementally so a long unattended run can be inspected/resumed
        pd.DataFrame(rows).to_csv(output_path, index=False)

    df = pd.DataFrame(rows)
    print(f"\nDone: {len(df)} tickers screened.")
    print(df["status"].value_counts())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
