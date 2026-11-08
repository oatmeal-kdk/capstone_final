"""Save the GA-tuned rule parameters of each stock to data/params/<ticker>.json, added 2026-11-08.

    python -m legacy_v1.pipeline.save_ga_params [--tickers SPY QQQ AAPL GLD] [--population 30] [--generations 20] [--jobs 4]
    python -m legacy_v1.pipeline.save_ga_params --from-run results/legacy_v1/esn_backtest_ga     # reuse a finished GA run

File layout:
  folds  one entry per walk-forward fold, keyed by its test start date: the training window, CPM P and T, and for each of the
         19 rules (legacy_v1/ga/rules.py ALL_RULES) the GA parameters with their training and default fitness. The parameters of a
         fold come from its training window only, so a backtest that uses fold k's entry for fold k's test year has no look-ahead.
         For backtests (select_esn_backtest.py --inputs ga --ga-params data/params).
  live   a copy of the last fold (the most recent training window), for trading from now on. Never use it to backtest earlier
         years: it was tuned on prices after them.
The folds, the CPM selection, the fitness and the GA are those of legacy_v1/ga/features.py and select_esn_backtest.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.experiments.ga_indicator_params import _points, folds
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.ga.features import tune_rule
from legacy_v1.ga.optimizer import GAConfig
from legacy_v1.ga.rules import ALL_RULES
from legacy_v1.pipeline.run_walkforward import _pareto_knee_params

OUT = Path("data/params")


def _frame(ticker: str) -> pd.DataFrame:
    """The rows every walk-forward script uses (ta warm-up dropped), so fold boundaries match."""
    df = build_ta_features(pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv"))
    return df.dropna(subset=TA_FEATURE_COLUMNS).reset_index(drop=True)


def _fold_entry(df: pd.DataFrame, f: dict, p_value: float, t_value: int, rules: dict) -> tuple[str, dict]:
    dates = pd.to_datetime(df["Date"])
    return str(dates.iloc[f["test_start"]].date()), {
        "train": [str(dates.iloc[f["train_start"]].date()), str(dates.iloc[f["train_end"] - 1].date())],
        "test_end": str(dates.iloc[f["test_end"] - 1].date()), "P": p_value, "T": t_value, "rules": rules}


def tune_stock(ticker: str, config: GAConfig) -> dict:
    df = _frame(ticker)
    close = df["Close"].astype(float)
    out = {}
    for f in folds(pd.to_datetime(df["Date"])):
        p_value, t_value = _pareto_knee_params(close.iloc[f["train_start"]:f["train_end"]])
        points = _points(close, f["train_start"], f["train_end"], p_value, t_value)
        frame, close_np = df.iloc[:f["test_end"]], close.to_numpy()[:f["test_end"]]
        rules = {}
        for name, rule in ALL_RULES.items():
            result, _ = tune_rule(rule, frame, close_np, points, config)
            rules[name] = {"params": result["params"], "train_fitness": result["objective"], "default_fitness": result["default_objective"]}
        key, entry = _fold_entry(df, f, p_value, t_value, rules)
        out[key] = entry
        print(f"{ticker} fold {f['fold']} ({key}) tuned", flush=True)
    return out


def from_run(ticker: str, run_dir: Path) -> dict:
    """Folds from a finished select_esn_backtest.py --inputs ga run (its ga_params.csv and folds.csv)."""
    df = _frame(ticker)
    ga = pd.read_csv(run_dir / "ga_params.csv")
    meta = pd.read_csv(run_dir / "folds.csv").set_index(["ticker", "fold"])
    ga = ga[ga["ticker"] == ticker]
    if ga.empty:
        raise ValueError(f"{ticker} is not in {run_dir / 'ga_params.csv'}")
    out = {}
    for f in folds(pd.to_datetime(df["Date"])):
        rows = ga[ga["fold"] == f["fold"]]
        rules = {r.rule: {"params": json.loads(r.params), "train_fitness": r.train_fitness, "default_fitness": r.default_fitness}
                 for r in rows.itertuples()}
        if set(rules) != set(ALL_RULES):
            raise ValueError(f"{ticker} fold {f['fold']}: rules {sorted(set(ALL_RULES) - set(rules))} missing in the run")
        m = meta.loc[(ticker, f["fold"])]
        if str(m["test_year"]) != str(pd.to_datetime(df["Date"]).iloc[f["test_start"]].date()):
            raise ValueError(f"{ticker} fold {f['fold']}: the run's test start {m['test_year']} does not match")
        key, entry = _fold_entry(df, f, float(m["P"]), int(m["T"]), rules)
        out[key] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "GLD"])
    parser.add_argument("--from-run", default=None, help="Folder of a finished select_esn_backtest.py --inputs ga run to copy from.")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    config = GAConfig(population=args.population, generations=args.generations, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_run:
        fold_sets = [from_run(t, Path(args.from_run)) for t in args.tickers]
        source = f"copied from {args.from_run}"
    else:
        fold_sets = Parallel(n_jobs=args.jobs)(delayed(tune_stock)(t, config) for t in args.tickers)
        source = "tuned by legacy_v1.pipeline.save_ga_params"
    for ticker, fold_set in zip(args.tickers, fold_sets):
        last = list(fold_set)[-1]
        payload = {"ticker": ticker, "saved": str(date.today()), "source": source,
                   "ga": {"population": config.population, "generations": config.generations, "seed": config.seed},
                   "fitness": "legacy_v1/ga/fitness.py normalized turning-point fitness on the training window's CPM turning points",
                   "folds": fold_set,
                   "live": {"from_fold": last, **fold_set[last],
                            "note": "most recent training window; for trading from now on, never for backtesting earlier years"}}
        path = out / f"{ticker}.json"
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"saved {path}: {len(fold_set)} folds, live from the fold starting {last} (training {fold_set[last]['train'][0]} ~ {fold_set[last]['train'][1]})")


if __name__ == "__main__":
    main()
