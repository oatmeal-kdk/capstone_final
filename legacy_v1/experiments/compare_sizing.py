"""Does the ESN score's size carry information, and do position-sizing rules beat all in / all out? Added 2026-11-09.

    python -m legacy_v1.experiments.compare_sizing [--runs results/legacy_v1/seed_check results/legacy_v1/seed_check_tb]

Reads finished run_walkforward.py runs (<ticker>_s<seed>/walkforward_predictions.csv and summary.json; one folder per label)
and re-trades the saved daily scores, so nothing is retrained.

1. Score strength (diagnostic, test years only): per stock and seed, the test days are split into quintiles of the score
   (most negative = strongest buy ... most positive = strongest sell) and the mean return of the next 10 days (from the next
   open) is taken per quintile; also the Spearman correlation of -score with that forward return. If the size of the score
   means something, the strongest-buy quintile should have the highest forward return and the strongest-sell the lowest.
2. Rules on the same scores and the fold's threshold thr (from summary.json), all filled at the next day's open, costs
   FEE_RATE and SLIPPAGE_RATE of production_config on the value traded (as simple_backtest):
     allin      the v1 rule: buy all below -thr, sell all above thr, otherwise keep (weight 0 / 1)
     split2/4   each buy signal adds 1/k of the portfolio, each sell signal removes 1/k (weight clipped to 0 .. 1)
     confirm2   allin, but a signal acts only on the second day in a row
     scaled     weight = clip(0.5 - score / (4 thr), 0, 1): 50% at score 0, 100% at score <= -2 thr, 0 at >= 2 thr;
                rebalanced only when the target moves more than 0.1 from the current weight
   and holding. allin is checked equal to simple_backtest.run_long_only_backtest (next_open, same costs).
Saved in results/legacy_v1/sizing/: strength.csv, rules.csv (per label, stock, seed and rule: Sharpe, CAGR, MDD, turnover).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from legacy_v1.backtest.simple_backtest import run_long_only_backtest
from legacy_v1.pipeline import production_config as cfg

OUT = Path("results/legacy_v1/sizing")
BUY_COST = 1.0 - 1.0 / ((1.0 + cfg.SLIPPAGE_RATE) * (1.0 + cfg.FEE_RATE))   # share of the traded value lost, as in simple_backtest
SELL_COST = 1.0 - (1.0 - cfg.SLIPPAGE_RATE) * (1.0 - cfg.FEE_RATE)
HORIZON = 10


def weights(rule: str, score: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Target weight decided at each day's close (traded at the next open)."""
    signal = np.where(score < -thr, -1, np.where(score > thr, 1, 0))
    out, w = np.zeros(len(score)), 0.0
    for i, s in enumerate(signal):
        if rule == "allin":
            w = 1.0 if s == -1 else 0.0 if s == 1 else w
        elif rule.startswith("split"):
            k = int(rule[5:])
            w = min(1.0, w + 1 / k) if s == -1 else max(0.0, w - 1 / k) if s == 1 else w
        elif rule == "confirm2":
            if i > 0 and s != 0 and signal[i - 1] == s:
                w = 1.0 if s == -1 else 0.0
        elif rule == "scaled":
            target = float(np.clip(0.5 - score[i] / (4 * thr[i]), 0.0, 1.0))
            if abs(target - w) > 0.1 or target in (0.0, 1.0) and target != w:
                w = target
        out[i] = w
    return out


def returns(close: np.ndarray, open_: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Daily returns when the weight decided on day i-1 is set at day i's open; cost per unit of weight traded."""
    r = np.zeros(len(close))
    held, turnover = 0.0, 0.0
    for i in range(1, len(close)):
        new = w[i - 1]
        gap = held * (open_[i] / close[i - 1] - 1.0)                 # overnight on yesterday's weight
        value_at_open = 1.0 + gap
        drift = held * open_[i] / close[i - 1] / value_at_open if value_at_open > 0 else 0.0   # weight after the gap
        trade = abs(new - drift)
        turnover += trade
        cost = (BUY_COST if new > drift else SELL_COST) * trade
        r[i] = (1.0 + gap) * (1.0 - cost) * (1.0 + new * (close[i] / open_[i] - 1.0)) - 1.0
        held = new * close[i] / open_[i] / (1.0 + new * (close[i] / open_[i] - 1.0)) if new > 0 else 0.0
    return r, turnover


def stats(r: np.ndarray) -> dict:
    equity = np.cumprod(1.0 + r)
    return {"sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
            "cagr": float(equity[-1] ** (252 / len(r)) - 1.0), "mdd": float((equity / np.maximum.accumulate(equity) - 1.0).min())}


def _check(pred: pd.DataFrame, thr: np.ndarray) -> None:
    signal = np.where(pred["trading_score"] < -thr, -1, np.where(pred["trading_score"] > thr, 1, 0))
    equity, _, _ = run_long_only_backtest(pred.assign(signal=signal), price_col="close", signal_col="signal", fee_rate=cfg.FEE_RATE,
                                          slippage_rate=cfg.SLIPPAGE_RATE, execution="next_open")
    reference = equity["portfolio_value"].pct_change().fillna(0.0).to_numpy()
    mine, _ = returns(pred["close"].to_numpy(), pred["open"].to_numpy(), weights("allin", pred["trading_score"].to_numpy(), thr))
    assert np.allclose(mine, reference, atol=1e-9), f"allin differs from run_long_only_backtest by {np.abs(mine - reference).max()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["results/legacy_v1/seed_check", "results/legacy_v1/seed_check_tb"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    strength, rows, checked = [], [], False
    for run_dir in map(Path, args.runs):
        for folder in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            ticker, seed = folder.name.rsplit("_s", 1)
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            pred = pd.read_csv(folder / "walkforward_predictions.csv")
            fold_thr = {f["fold"]: f["selected_threshold"] for f in summary["folds"]}
            thr = pred["fold"].map(fold_thr).to_numpy(dtype=float)
            score, close, open_ = pred["trading_score"].to_numpy(), pred["close"].to_numpy(), pred["open"].to_numpy()
            label = summary["target_method"]
            if not checked:
                _check(pred, thr)
                checked = True

            forward = np.full(len(close), np.nan)                    # next open .. close 10 days later
            forward[:-HORIZON - 1] = close[HORIZON + 1:] / open_[1:-HORIZON] - 1.0
            ok = ~np.isnan(forward)
            quintile = pd.qcut(score[ok], 5, labels=False, duplicates="drop")
            means = pd.Series(forward[ok]).groupby(quintile).mean()
            strength.append({"label": label, "ticker": ticker, "seed": int(seed), **{f"q{q + 1}": means.get(q, np.nan) for q in range(5)},
                             "spearman": float(spearmanr(-score[ok], forward[ok]).correlation)})

            for rule in ("allin", "split2", "split4", "confirm2", "scaled"):
                r, turnover = returns(close, open_, weights(rule, score, thr))
                rows.append({"label": label, "ticker": ticker, "seed": int(seed), "rule": rule, **stats(r),
                             "turnover_per_year": turnover / len(r) * 252})
            hold = np.concatenate([[0.0], close[1:] / close[:-1] - 1.0])
            rows.append({"label": label, "ticker": ticker, "seed": int(seed), "rule": "hold", **stats(hold), "turnover_per_year": 0.0})

    strength_table, table = pd.DataFrame(strength), pd.DataFrame(rows)
    strength_table.to_csv(OUT / "strength.csv", index=False)
    table.to_csv(OUT / "rules.csv", index=False)
    pd.set_option("display.width", 220)
    print("== mean 10-day forward return by score quintile (q1 = strongest buy ... q5 = strongest sell), mean over seeds ==")
    print(strength_table.groupby(["label", "ticker"])[["q1", "q2", "q3", "q4", "q5", "spearman"]].mean().round(4).to_string())
    print("\n== Sharpe by rule, mean over seeds ==")
    print(table.pivot_table(index=["label", "rule"], columns="ticker", values="sharpe", aggfunc="mean").round(3).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
