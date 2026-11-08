"""GA search of indicator rule parameters against CPM turning points, per stock and walk-forward fold.

    python -m legacy_v1.experiments.ga_indicator_params [--tickers SPY QQQ AAPL GLD] [--rules rsi cci ...] [--population 30]
                                                        [--generations 20] [--seed 0] [--jobs 4]

For each fold (the fold dates of run_walkforward.py: rolling 5y training, 1y validation, 1y test steps):
  1. P and T of CPM by the Pareto knee on the training closes only (as cpm_causal); turning points of a CPM run on the training
     closes, without the last reported point (it can still move).
  2. For every rule of legacy_v1/ga/rules.py (ALL_RULES: 14 indicator rules + 5 candle patterns): the GA (legacy_v1/ga/optimizer.py) minimises the normalized turning-point
     fitness (legacy_v1/ga/fitness.py) on the training turning points. The validation and test years are not seen.
  3. Out of sample: the GA parameters and the default parameters are both scored on the test year's turning points (a CPM run
     with the same P, T on the closes up to the test end, last reported point dropped; turning points inside the test year are
     evaluated, their neighbours only bound the zones). This uses test prices to build the answer key, so it is an
     evaluation only and is never fed back into a choice.
Saved in results/legacy_v1/ga_params/: folds.csv (one row per stock, fold and rule: P, T, turning points, normalized fitness of
the default and the GA parameters on training and test, hit and wrong rates, signal counts, the GA parameters as JSON).
Nothing here trades or changes the walk-forward pipeline; the parameters are an input for a later comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.ga.fitness import turning_point_fitness
from legacy_v1.ga.optimizer import GAConfig, optimize
from legacy_v1.ga.rules import ALL_RULES as RULES
from legacy_v1.labels.cpm import detect_turning_points
from legacy_v1.pipeline.run_walkforward import _pareto_knee_params

OUT = Path("results/legacy_v1/ga_params")


def folds(dates: pd.Series, train_years: int = 5, val_years: int = 1, step_years: int = 1) -> list[dict]:
    """The fold boundaries of run_walkforward.py with --train-window rolling (row positions)."""
    dates_np = dates.to_numpy()
    idx_for = lambda ts: int(np.searchsorted(dates_np, np.datetime64(ts)))  # noqa: E731
    out, cur = [], dates.iloc[0] + pd.DateOffset(years=train_years + val_years)
    while idx_for(cur) < len(dates):
        test_start, test_end = idx_for(cur), min(idx_for(cur + pd.DateOffset(years=step_years)), len(dates))
        val_start = idx_for(cur - pd.DateOffset(years=val_years))
        train_start = idx_for(cur - pd.DateOffset(years=val_years + train_years))
        if test_end <= test_start:
            break
        out.append({"fold": len(out) + 1, "train_start": train_start, "train_end": val_start, "test_start": test_start, "test_end": test_end,
                    "test_year": str(dates.iloc[test_start].date())})
        cur = cur + pd.DateOffset(years=step_years)
    return out


def _points(close: pd.Series, start: int, end: int, p_value: float, t_value: int) -> list[dict]:
    """Turning points of a CPM run on close[start:end] (last reported point dropped), indices into the full series."""
    pts = detect_turning_points(close.iloc[start:end].reset_index(drop=True), P=p_value, T=t_value)[:-1]
    return [{"index": int(p["index"]) + start, "type": p["type"]} for p in pts]


def _test_points(points: list[dict], test_start: int, test_end: int) -> list[dict]:
    """The turning points inside the test year plus one neighbour on each side (neighbours only bound the zones)."""
    inside = [i for i, p in enumerate(points) if test_start <= p["index"] < test_end]
    if not inside:
        return []
    return points[max(inside[0] - 1, 0):inside[-1] + 2]


def run_stock(ticker: str, rule_names: list[str], config: GAConfig) -> list[dict]:
    df = pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv")
    df = build_ta_features(df).dropna(subset=TA_FEATURE_COLUMNS).reset_index(drop=True)   # same rows and folds as run_walkforward
    close = df["Close"].astype(float)
    close_np = close.to_numpy()
    rows = []
    for f in folds(pd.to_datetime(df["Date"])):
        t0 = time.time()
        p_value, t_value = _pareto_knee_params(close.iloc[f["train_start"]:f["train_end"]])
        train_pts = _points(close, f["train_start"], f["train_end"], p_value, t_value)
        test_pts = _test_points(_points(close, f["train_start"], f["test_end"], p_value, t_value), f["test_start"], f["test_end"])
        frame = df.iloc[:f["test_end"]]                     # indicators are causal; rows after the test end are never used
        for name in rule_names:
            rule = RULES[name]
            cache: dict[tuple, np.ndarray] = {}

            def signal(params: dict) -> np.ndarray:
                key = tuple(params.values())
                if key not in cache:
                    cache[key] = np.asarray(rule.signal(frame, params), dtype=float)
                return cache[key]

            result = optimize(rule, lambda params: turning_point_fitness(close_np, signal(params), train_pts)["normalized"], config)
            default = rule.as_dict(rule.default())
            scores = {}
            for label, params in (("default", default), ("ga", result["params"])):
                for part, pts in (("train", train_pts), ("test", test_pts)):
                    s = turning_point_fitness(close_np, signal(params), pts)
                    scores.update({f"{label}_{part}": s["normalized"], f"{label}_{part}_hit": s["hit_rate"],
                                   f"{label}_{part}_wrong": s["wrong_rate"], f"{label}_{part}_signals": s["signals"]})
            rows.append({"ticker": ticker, **{k: f[k] for k in ("fold", "test_year")}, "P": p_value, "T": t_value,
                         "train_points": max(len(train_pts) - 2, 0), "test_points": max(len(test_pts) - 2, 0), "rule": name, **scores,
                         "evaluations": result["evaluations"], "params": json.dumps(result["params"]), "default_params": json.dumps(default)})
        print(f"{ticker} fold {f['fold']} ({f['test_year']}): P={p_value} T={t_value}, {len(train_pts)} training points, "
              f"{len(rule_names)} rules in {time.time() - t0:.0f}s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "GLD"])
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    config = GAConfig(population=args.population, generations=args.generations, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=args.jobs)(delayed(run_stock)(t, args.rules, config) for t in args.tickers)
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(out / "folds.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== median normalized fitness over stocks and folds (0 = every turning point hit, 1 = no better than no signal) ==")
    summary = table.groupby("rule")[["default_train", "ga_train", "default_test", "ga_test"]].median()
    summary["ga_better_test"] = table.assign(b=table["ga_test"] < table["default_test"]).groupby("rule")["b"].mean()
    print(summary.round(3).sort_values("ga_test").to_string())
    print(f"\n== GA parameters by fold, {args.tickers[0]} (values in the order of the rule's parameters; default first) ==")
    first = table[table["ticker"] == args.tickers[0]].copy()
    short = lambda js: "/".join(f"{v:g}" if isinstance(v, int) else f"{v:.3g}" for v in json.loads(js).values())  # noqa: E731
    first["short"] = first["params"].map(short)
    grid = first.pivot_table(index="rule", columns="fold", values="short", aggfunc="first")
    grid.insert(0, "default", first.groupby("rule")["default_params"].first().map(short))
    print(grid.to_string())
    print(f"\nSaved {out / 'folds.csv'}")


if __name__ == "__main__":
    main()
