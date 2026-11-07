"""Threshold rule for the single-stock v1 pipeline: validation accuracy (v1) vs a training-score quantile vs a fixed 0.

    python -m legacy_v1.experiments.compare_thresholds [--label triple_barrier|cpm_causal] [--per-market 30] [--jobs 6]

Declared before running (2026-11-07); nothing below was changed after seeing results.
  Compared    accuracy  the grid value {0.1..0.9} with the best 3-class validation accuracy (--threshold-rule accuracy, v1)
              quantile  the 80th percentile of |training score| (after washout), so a buy or sell signal is on for about 20% of
                        training days; no labels used (--threshold-rule quantile --signal-share 0.2)
              zero      threshold 0, the sign of the score (--threshold-rule zero)
  Label       triple barrier (default) or cpm_causal (--label, added 2026-11-07 after the triple barrier run; same design).
  Fixed       ta 20 indicators, ESN (single seed), ridge on [1, inputs, states], rolling 5y training window,
              1y validation, 1y test steps, the v1 rule (buy below -thr, sell above thr, otherwise keep the position), fills at the
              next day's open, costs 0.05% + 0.05%, no overlay. Test dates are the same for all rules.
  Stocks      the 64 of compare_labels.py: SPY, QQQ, AAPL, GLD and 30 random US + 30 random KR stocks (seed 0).
  Verdict     legacy_v1/experiments/judge.py: median over stocks of Sharpe(rule) - Sharpe(accuracy) for quantile and zero, 95%
              calendar-month block-bootstrap interval. Also: each rule against holding, the median selected threshold, the share
              of test days in the market, and trades per year.
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

from legacy_v1.experiments.compare_execution import CORE, sample
from legacy_v1.experiments.judge import judge
from legacy_v1.pipeline import production_config as cfg

OUT = Path("results/legacy_v1/thresholds")          # cpm_causal: results/legacy_v1/thresholds_cpm_causal
RULES = ("accuracy", "quantile", "zero")


def run(symbol: str, rule: str, label: str, root: Path) -> dict:
    out = root / f"{symbol}_{rule}"
    cmd = [sys.executable, "-W", "ignore", "-m", "legacy_v1.pipeline.run_walkforward", "--tp-path", f"data/processed/{symbol}/1d/{symbol}_with_tp.csv",
           "--target-method", label, "--train-years", str(cfg.TRAIN_YEARS), "--train-window", "rolling", "--fee-rate", str(cfg.FEE_RATE),
           "--slippage-rate", str(cfg.SLIPPAGE_RATE), "--execution", "next_open", "--threshold-rule", rule, "--signal-share", "0.2", "--output-dir", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not (out / "equity_curve.csv").exists():
        return {"symbol": symbol, "rule": rule, "error": result.stderr.strip().splitlines()[-1][:200] if result.stderr else "no output"}
    return {"symbol": symbol, "rule": rule, "error": None}


def load(symbol: str, rule: str) -> tuple[pd.Series, pd.Series, dict]:
    """(strategy, hold) daily returns indexed by date, and per-stock facts: score-label correlation, threshold, exposure, trades."""
    folder = OUT / f"{symbol}_{rule}"
    equity = pd.read_csv(folder / "equity_curve.csv", parse_dates=["date"]).set_index("date")
    pred = pd.read_csv(folder / "walkforward_predictions.csv")
    folds = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["folds"]
    labelled = pred.dropna(subset=["target"])
    corr = float(np.corrcoef(labelled["trading_score"], labelled["target"])[0, 1]) if labelled["trading_score"].std() > 0 else float("nan")
    held = _held(pred["signal"].to_numpy())
    facts = {"corr": corr, "threshold": float(np.median([f["selected_threshold"] for f in folds])),
             "in_market": float(held.mean()), "trades_per_year": float(np.abs(np.diff(held, prepend=0)).sum() / (len(held) / 252))}
    return equity["portfolio_value"].pct_change().fillna(0.0), equity["close"].pct_change().fillna(0.0), facts


def _held(signal: np.ndarray) -> np.ndarray:
    """1 while the v1 rule holds the stock: enter on buy (-1), leave on sell (+1), otherwise keep."""
    out, pos = np.zeros(len(signal)), 0
    for i, s in enumerate(signal):
        pos = 1 if s == -1 else 0 if s == 1 else pos
        out[i] = pos
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-market", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--label", choices=["triple_barrier", "cpm_causal"], default="triple_barrier")
    args = parser.parse_args()
    global OUT
    if args.label != "triple_barrier":
        OUT = OUT.with_name(f"{OUT.name}_{args.label}")
    OUT.mkdir(parents=True, exist_ok=True)
    picked = sample(args.per_market)
    groups = {s: "core" for s in CORE} | {s: m for m, symbols in picked.items() for s in symbols if s not in CORE}
    runs = pd.DataFrame(Parallel(n_jobs=args.jobs)(delayed(run)(s, r, args.label, OUT) for s in groups for r in RULES))
    failed = set(runs.loc[runs["error"].notna(), "symbol"])
    symbols = [s for s in groups if s not in failed]               # a stock is compared only if both of its runs finished

    frames, rows = {}, []
    loaded = {(s, r): load(s, r) for s in symbols for r in RULES}
    for r in RULES:
        frames[r] = pd.DataFrame({s: loaded[(s, r)][0] for s in symbols})
    frames["hold"] = pd.DataFrame({s: loaded[(s, RULES[0])][1] for s in symbols})
    for s in symbols:
        for name in (*RULES, "hold"):
            r = frames[name][s].dropna()
            facts = loaded[(s, name)][2] if name != "hold" else {"in_market": 1.0}
            rows.append({"symbol": s, "group": groups[s], "rule": name, **facts,
                         "sharpe": float(r.mean() / r.std() * 252 ** 0.5) if r.std() > 0 else float("nan")})
    pairs = (("quantile", "accuracy"), ("zero", "accuracy"), ("accuracy", "hold"), ("quantile", "hold"), ("zero", "hold"))
    verdicts = [{"strategy": a, "baseline": b, **judge(frames[a], frames[b])} for a, b in pairs]

    table = pd.DataFrame(rows)
    table.to_csv(OUT.parent / f"{OUT.name}_compare.csv", index=False)
    verdict_table = pd.DataFrame(verdicts)
    verdict_table.to_csv(OUT.parent / f"{OUT.name}_verdicts.csv", index=False)

    pd.set_option("display.width", 220)
    print("== core: Sharpe, median threshold, share of days in the market, trades per year ==")
    print(table[table["group"] == "core"].pivot_table(index="symbol", columns="rule", values=["sharpe", "threshold", "in_market", "trades_per_year"]).round(3).to_string())
    print("\n== median by group ==")
    print(table.pivot_table(index="group", columns="rule", values=["sharpe", "in_market", "trades_per_year"], aggfunc="median").round(3).to_string())
    print("\n== verdicts (median Sharpe difference, 95% month-block interval) ==")
    print(verdict_table[["strategy", "baseline", "stocks", "median_sharpe_diff", "ci_lo", "ci_hi", "verdict", "share_stocks_ahead"]].round(3).to_string(index=False))
    print(f"\n{len(failed)} stocks dropped for a failed run" + (f": {sorted(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
