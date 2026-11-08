"""Ridge readout on [1, 20 inputs, 100 states] (121 columns, the v1 readout) vs on [1, 100 states] only (101 columns).

    python -m legacy_v1.experiments.compare_readout [--per-market 30] [--jobs 6]

Declared before running (2026-11-07); nothing below was changed after seeing results.
  Compared    input_state  ridge on today's 20 inputs and the 100 reservoir states (--readout input_state, the default)
              state        ridge on the 100 reservoir states only (--readout state)
  Fixed       triple barrier label, ta 20 indicators, ESN (single seed), rolling 5y training window, threshold on validation
              accuracy, 1y validation, 1y test steps, fills at the next day's open, costs 0.05% + 0.05%, no overlay.
              Test dates are the same for both readouts.
  Stocks      the 64 of compare_labels.py: SPY, QQQ, AAPL, GLD and 30 random US + 30 random KR stocks (seed 0).
  Verdict     legacy_v1/experiments/judge.py: median over stocks of Sharpe(state) - Sharpe(input_state), 95% calendar-month
              block-bootstrap interval. Also shown: each readout against holding, and the test-period correlation of the
              score with the label (pooled over stocks' days, median over stocks).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.experiments.compare_execution import CORE, sample
from legacy_v1.experiments.judge import judge
from legacy_v1.pipeline import production_config as cfg

OUT = Path("results/legacy_v1/readout")
READOUTS = ("input_state", "state")


def run(symbol: str, readout: str) -> dict:
    out = OUT / f"{symbol}_{readout}"
    cmd = [sys.executable, "-W", "ignore", "-m", "legacy_v1.pipeline.run_walkforward", "--inputs", "ta", "--tp-path", f"data/processed/{symbol}/1d/{symbol}_with_tp.csv",
           "--target-method", "triple_barrier", "--train-years", str(cfg.TRAIN_YEARS), "--train-window", "rolling", "--fee-rate", str(cfg.FEE_RATE),
           "--slippage-rate", str(cfg.SLIPPAGE_RATE), "--execution", "next_open", "--readout", readout, "--output-dir", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not (out / "equity_curve.csv").exists():
        return {"symbol": symbol, "readout": readout, "error": result.stderr.strip().splitlines()[-1][:200] if result.stderr else "no output"}
    return {"symbol": symbol, "readout": readout, "error": None}


def load(symbol: str, readout: str) -> tuple[pd.Series, pd.Series, float]:
    """(strategy, hold) daily returns indexed by date, and the test-period score-label correlation."""
    folder = OUT / f"{symbol}_{readout}"
    equity = pd.read_csv(folder / "equity_curve.csv", parse_dates=["date"]).set_index("date")
    pred = pd.read_csv(folder / "walkforward_predictions.csv").dropna(subset=["target"])
    corr = float(np.corrcoef(pred["trading_score"], pred["target"])[0, 1]) if pred["trading_score"].std() > 0 else float("nan")
    return equity["portfolio_value"].pct_change().fillna(0.0), equity["close"].pct_change().fillna(0.0), corr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-market", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picked = sample(args.per_market)
    groups = {s: "core" for s in CORE} | {s: m for m, symbols in picked.items() for s in symbols if s not in CORE}
    runs = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(run)(s, r) for s in groups for r in READOUTS))
    failed = set(runs.loc[runs["error"].notna(), "symbol"])
    symbols = [s for s in groups if s not in failed]               # a stock is compared only if both of its runs finished

    frames, rows = {}, []
    loaded = {(s, r): load(s, r) for s in symbols for r in READOUTS}
    for r in READOUTS:
        frames[r] = pd.DataFrame({s: loaded[(s, r)][0] for s in symbols})
    frames["hold"] = pd.DataFrame({s: loaded[(s, READOUTS[0])][1] for s in symbols})
    for s in symbols:
        for name in (*READOUTS, "hold"):
            r = frames[name][s].dropna()
            rows.append({"symbol": s, "group": groups[s], "readout": name, "corr": loaded[(s, name)][2] if name != "hold" else np.nan,
                         "sharpe": float(r.mean() / r.std() * 252 ** 0.5) if r.std() > 0 else float("nan")})
    verdicts = [{"strategy": a, "baseline": b, **judge(frames[a], frames[b])} for a, b in (("state", "input_state"), ("input_state", "hold"), ("state", "hold"))]

    table = pd.DataFrame(rows)
    table.to_csv(OUT.parent / "readout_compare.csv", index=False)
    verdict_table = pd.DataFrame(verdicts)
    verdict_table.to_csv(OUT.parent / "readout_verdicts.csv", index=False)

    pd.set_option("display.width", 220)
    print("== core: Sharpe and score-label correlation ==")
    print(table[table["group"] == "core"].pivot_table(index="symbol", columns="readout", values=["sharpe", "corr"]).round(3).to_string())
    print("\n== median by group ==")
    print(table.pivot_table(index="group", columns="readout", values=["sharpe", "corr"], aggfunc="median").round(3).to_string())
    print("\n== verdicts (median Sharpe difference, 95% month-block interval) ==")
    print(verdict_table[["strategy", "baseline", "stocks", "median_sharpe_diff", "ci_lo", "ci_hi", "verdict", "share_stocks_ahead"]].round(3).to_string(index=False))
    print(f"\n{len(failed)} stocks dropped for a failed run" + (f": {sorted(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
