"""How much of the v1 result came from trading at the signal day's own close?

    python -m legacy_v1.experiments.compare_execution [--per-market 120] [--jobs 6]

Each stock is run twice with the locked recipe (no overlay, one reservoir seed, costs 0.05% + 0.05%, as in the old screening):
same_close (the original fill) and next_open (the signal of day t is traded at the open of day t+1). Nothing else changes, so the
difference is the execution assumption alone. SPY, QQQ, AAPL and GLD (the Phase 9-12 tickers) are also run with the risk overlay.

The stocks are a random sample (fixed seed) of everything the old screening could run, NOT the 'active' list, so the comparison
is not limited to stocks chosen for their results. Buy-and-hold of the same stock over the same dates is reported next to it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.pipeline import production_config as cfg

OUT = Path("results/legacy_v1/execution")
CORE = ["SPY", "QQQ", "AAPL", "GLD"]
LISTS = {"US": ["sp500", "sp400"], "KR": ["kospi", "kosdaq"]}


def run(symbol: str, execution: str, overlay: bool) -> dict:
    path = Path(f"data/processed/{symbol}/1d/{symbol}_with_tp.csv")
    out = OUT / f"{symbol}_{execution}{'_overlay' if overlay else ''}"
    cmd = [sys.executable, "-W", "ignore", "-m", "legacy_v1.pipeline.run_walkforward", "--tp-path", str(path), "--target-method", "triple_barrier",
           "--train-years", str(cfg.TRAIN_YEARS), "--fee-rate", str(cfg.FEE_RATE), "--slippage-rate", str(cfg.SLIPPAGE_RATE), "--execution", execution,
           "--output-dir", str(out),
           "--train-window", "expanding"]                 # run with the v1 window; the CLI default became rolling on 2026-11-04
    if overlay:
        cmd += ["--risk-overlay", "--target-vol", str(cfg.RISK_OVERLAY_TARGET_VOL), "--rebalance-band", str(cfg.RISK_OVERLAY_REBALANCE_BAND)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if not (out / "equity_curve.csv").exists():
        return {"symbol": symbol, "execution": execution, "overlay": overlay, "error": result.stderr[-300:]}
    equity = pd.read_csv(out / "equity_curve.csv")
    strategy, hold = equity["portfolio_value"].pct_change().fillna(0.0), equity["close"].pct_change().fillna(0.0)
    years = len(equity) / 252
    sharpe = lambda r: float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    cagr = lambda r: float((1 + r).prod() ** (1 / years) - 1)
    mdd = lambda r: float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return {"symbol": symbol, "execution": execution, "overlay": overlay, "start": equity["date"].iloc[0], "end": equity["date"].iloc[-1],
            "sharpe": sharpe(strategy), "cagr": cagr(strategy), "mdd": mdd(strategy), "exposure": float((equity["position"] > 0).mean()),
            "trades": int(summary["strategy"]["num_trades"]), "hold_sharpe": sharpe(hold), "hold_cagr": cagr(hold), "hold_mdd": mdd(hold)}


def sample(per_market: int, seed: int = 0) -> dict[str, list[str]]:
    picked = {}
    for market, lists in LISTS.items():
        frame = pd.concat([pd.read_csv(f"data/universe/{name}_screening_results.csv") for name in lists])
        frame = frame[(frame["status"] != "error")].drop_duplicates("symbol")
        frame = frame[[Path(f"data/processed/{s}/1d/{s}_with_tp.csv").exists() for s in frame["symbol"]]]
        picked[market] = frame["symbol"].sample(min(per_market, len(frame)), random_state=seed).tolist()
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-market", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picked = sample(args.per_market)
    jobs = [(s, e, False, m) for m, symbols in picked.items() for s in symbols for e in ("same_close", "next_open")]
    jobs += [(s, e, o, "core") for s in CORE for e in ("same_close", "next_open") for o in (False, True)]
    rows = Parallel(n_jobs=args.jobs)(delayed(run)(s, e, o) for s, e, o, _ in jobs)
    table = pd.DataFrame([{**r, "group": g} for r, (_, _, _, g) in zip(rows, jobs)])
    table.to_csv(OUT.parent / "execution_compare.csv", index=False)

    ok = table[table["error"].isna()] if "error" in table else table
    pd.set_option("display.width", 200)
    for group in ("US", "KR"):
        part = ok[(ok["group"] == group) & ~ok["overlay"]]
        wide = part.pivot_table(index="symbol", columns="execution", values=["sharpe", "cagr", "mdd"]).dropna()
        hold = part.groupby("symbol")[["hold_sharpe", "hold_cagr"]].first().reindex(wide.index)
        print(f"\n== {group}: {len(wide)} stocks (random sample, not the 'active' list), median ==")
        print(pd.DataFrame({"same_close": [wide[("sharpe", "same_close")].median(), wide[("cagr", "same_close")].median(), wide[("mdd", "same_close")].median()],
                            "next_open": [wide[("sharpe", "next_open")].median(), wide[("cagr", "next_open")].median(), wide[("mdd", "next_open")].median()],
                            "buy_and_hold": [hold["hold_sharpe"].median(), hold["hold_cagr"].median(), np.nan]},
                           index=["sharpe", "cagr", "mdd"]).round(3).to_string())
        drop = wide[("sharpe", "next_open")] - wide[("sharpe", "same_close")]
        print(f"Sharpe change next_open - same_close: median {drop.median():+.3f}, worse for {(drop < 0).mean():.0%} of stocks; "
              f"beats buy-and-hold on Sharpe: same_close {(wide[('sharpe', 'same_close')] > hold['hold_sharpe']).mean():.0%}, "
              f"next_open {(wide[('sharpe', 'next_open')] > hold['hold_sharpe']).mean():.0%}")
    core = ok[ok["group"] == "core"]
    print("\n== Phase 9-12 tickers ==")
    print(core.pivot_table(index=["symbol", "overlay"], columns="execution", values=["sharpe", "cagr", "mdd"]).round(3).to_string())
    print(core.groupby("symbol")[["hold_sharpe", "hold_cagr", "hold_mdd"]].first().round(3).to_string())
    errors = table[table["error"].notna()] if "error" in table else table.iloc[0:0]
    print(f"\n{len(errors)} runs failed" + (f": {errors['symbol'].tolist()[:10]}" if len(errors) else ""))


if __name__ == "__main__":
    main()
