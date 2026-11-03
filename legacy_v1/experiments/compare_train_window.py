"""Expanding vs rolling training window in the v1 recipe (triple barrier), with and without the risk overlay.

    python -m legacy_v1.experiments.compare_train_window [--per-market 30] [--jobs 6]

Declared before running (2026-11-03); nothing below was changed after seeing results.
  Compared    expanding  every row from the start of the data up to the validation year (the v1 default: 5 years in the first
                         fold, one more year in every later fold)
              rolling    only the last 5 years before the validation year (--rolling-train)
  Fixed       triple barrier label, ta 20 indicators, ESN (single seed), threshold on validation accuracy, 1y validation, 1y test
              steps, costs 0.05% + 0.05%, fills at the next day's open. Test dates are the same for both windows.
  Stocks      the 64 of compare_labels.py: SPY, QQQ, AAPL, GLD and 30 random US + 30 random KR stocks (seed 0).
  Verdict     legacy_v1/experiments/judge.py: median over stocks of Sharpe(rolling) - Sharpe(expanding), 95% calendar-month
              block-bootstrap interval, separately without and with the overlay. Also shown: each window against holding.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.experiments.compare_execution import CORE, sample
from legacy_v1.experiments.judge import judge
from legacy_v1.pipeline import production_config as cfg

OUT = Path("results/legacy_v1/train_window")
WINDOWS = ("expanding", "rolling")


def run(symbol: str, window: str, overlay: bool) -> dict:
    out = OUT / f"{symbol}_{window}{'_overlay' if overlay else ''}"
    cmd = [sys.executable, "-W", "ignore", "-m", "legacy_v1.pipeline.run_walkforward", "--tp-path", f"data/processed/{symbol}/1d/{symbol}_with_tp.csv",
           "--target-method", "triple_barrier", "--train-years", str(cfg.TRAIN_YEARS), "--fee-rate", str(cfg.FEE_RATE),
           "--slippage-rate", str(cfg.SLIPPAGE_RATE), "--execution", "next_open", "--output-dir", str(out)]
    if window == "rolling":
        cmd.append("--rolling-train")
    if overlay:
        cmd += ["--risk-overlay", "--target-vol", str(cfg.RISK_OVERLAY_TARGET_VOL), "--rebalance-band", str(cfg.RISK_OVERLAY_REBALANCE_BAND)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not (out / "equity_curve.csv").exists():
        return {"symbol": symbol, "window": window, "overlay": overlay, "error": result.stderr.strip().splitlines()[-1][:200] if result.stderr else "no output"}
    return {"symbol": symbol, "window": window, "overlay": overlay, "error": None}


def daily_returns(symbol: str, window: str, overlay: bool) -> tuple[pd.Series, pd.Series]:
    """(strategy, hold) daily returns of one run, indexed by date."""
    equity = pd.read_csv(OUT / f"{symbol}_{window}{'_overlay' if overlay else ''}" / "equity_curve.csv", parse_dates=["date"]).set_index("date")
    return equity["portfolio_value"].pct_change().fillna(0.0), equity["close"].pct_change().fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-market", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picked = sample(args.per_market)
    groups = {s: "core" for s in CORE} | {s: m for m, symbols in picked.items() for s in symbols if s not in CORE}
    jobs = [(s, w, o) for s in groups for w in WINDOWS for o in (False, True)]
    runs = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(run)(s, w, o) for s, w, o in jobs))
    failed = set(runs.loc[runs["error"].notna(), "symbol"])
    symbols = [s for s in groups if s not in failed]               # a stock is compared only if all four of its runs finished

    rows, verdicts = [], []
    for overlay in (False, True):
        frames = {}
        for window in WINDOWS:
            pairs = {s: daily_returns(s, window, overlay) for s in symbols}
            frames[window] = pd.DataFrame({s: p[0] for s, p in pairs.items()})
            frames["hold"] = pd.DataFrame({s: p[1] for s, p in pairs.items()})
        for s in symbols:
            for name in (*WINDOWS, "hold"):
                r = frames[name][s].dropna()
                rows.append({"symbol": s, "group": groups[s], "overlay": overlay, "window": name,
                             "sharpe": float(r.mean() / r.std() * 252 ** 0.5) if r.std() > 0 else float("nan")})
        for strategy, baseline in (("rolling", "expanding"), ("expanding", "hold"), ("rolling", "hold")):
            verdicts.append({"overlay": overlay, "strategy": strategy, "baseline": baseline, **judge(frames[strategy], frames[baseline])})

    table = pd.DataFrame(rows)
    table.to_csv(OUT.parent / "train_window_compare.csv", index=False)
    verdict_table = pd.DataFrame(verdicts)
    verdict_table.to_csv(OUT.parent / "train_window_verdicts.csv", index=False)

    pd.set_option("display.width", 220)
    wide = table.pivot_table(index=["group", "symbol"], columns=["overlay", "window"], values="sharpe")
    print("== core: Sharpe by stock (overlay False / True) ==")
    print(wide.loc["core"].round(3).to_string())
    print("\n== median Sharpe by group ==")
    print(wide.groupby(level="group").median().round(3).to_string())
    print("\n== verdicts (median Sharpe difference, 95% month-block interval) ==")
    print(verdict_table[["overlay", "strategy", "baseline", "stocks", "median_sharpe_diff", "ci_lo", "ci_hi", "verdict", "share_stocks_ahead",
                         "years_ahead", "years"]].round(3).to_string(index=False))
    print(f"\n{len(failed)} stocks dropped for a failed run" + (f": {sorted(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
