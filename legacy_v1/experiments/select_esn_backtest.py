"""Choose the ESN reservoir parameters per fold by backtest Sharpe on years before the test year, then trade the test year.

    python -m legacy_v1.experiments.select_esn_backtest [--tickers SPY QQQ AAPL GLD | --per-market 30] [--seeds 0 1] [--jobs 6]

Declared before running (2026-11-08); nothing below is to be changed after seeing results.
  Fixed       label cpm_causal (P, T by the Pareto knee on the training closes), the 20 ta indicators min-max scaled on the
              training window, ridge readout on [1, inputs, states] with washout 30, threshold = 80th percentile of |score| on the
              rows the readout was fitted on (signal on for about 20% of those days; no labels), the v1 rule (buy below -thr,
              sell above thr, otherwise keep the position), fills at the next day's open, costs 0.05% + 0.05%, no overlay, rolling
              5y training / 1y validation / 1y test folds of run_walkforward.py. Every score is the mean of an ensemble of
              reservoirs with seeds --seeds (0 and 1); the reservoir runs without resets from the training start to the test end.
  Candidates  grid (243): reservoir 50/100/200, spectral radius 0.5/0.9/1.1, input scale 0.1/0.3/0.7, leak 0.1/0.3/0.7,
              ridge alpha 20/200/2000; sparsity 0.15. The default ESN_CONFIG (100, 0.9, 0.3, 0.3, 200) is one of them.
  Selection   per fold, the candidate with the highest Sharpe over four selection years, each traded starting flat with the
              readout and threshold fitted on every training-window row before it: training years 3, 4 and 5, and the
              validation year (readout on the whole training window). The test year is not used.
  Test        the selected and the default candidate are refitted on the whole training window and trade the test year; the
              test years are stitched and backtested once (as run_walkforward.py does).
  Verdict     legacy_v1/experiments/judge.py over stocks: median Sharpe(selected) - Sharpe(default), 95% month-block interval.
              Also: each against holding; and per fold, the Spearman correlation over the 243 candidates between the selection
              Sharpe and the test-year Sharpe (does a good selection score carry into the next year? diagnostic only, never used
              to choose).
Inputs (--inputs, added 2026-11-08 before any ga run; the rest of the design is unchanged):
  ta  the 20 ta indicators min-max scaled on the training window (the design above).
  ga  the paper's inputs: per fold, the 19 rules of legacy_v1/ga/rules.py (14 indicator rules + 5 candle patterns) tuned by the
      GA on the training window's CPM turning points (legacy_v1/ga/features.py, population/generations from --population and
      --generations), and their -1/0/+1 signals from the training start to the test end fed to the ESN unscaled.
      Saved in results/legacy_v1/esn_backtest_ga/, with the GA parameters of every fold in ga_params.csv.
      --ga-params data/params (added 2026-11-08): read each fold's parameters from legacy_v1/pipeline/save_ga_params.py files
      instead of running the GA (the same parameters, without the hour of tuning).
  ga_ta_candle (added 2026-11-09, the default input of run_walkforward.py since 2026-11-08): per fold, the 20 ta indicators
      computed with the fold's stored GA indicator parameters (min-max scaled on the training window) + the 5 candle patterns
      with the fold's GA parameters (-1/0/+1, unscaled), read from --ga-params (default data/params). Rows where a long GA
      window leaves an indicator undefined (the first rows of the data, first fold only) are fed as 0 and never fitted on.
      Saved in results/legacy_v1/esn_backtest_ga_ta_candle/.
Saved in results/legacy_v1/esn_backtest/: folds.csv (per stock and fold: P, T, selected parameters, selection Sharpe of the
selected and the default, the Spearman correlation), compare.csv (per stock Sharpe), verdicts.csv, daily returns per stock.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from legacy_v1.backtest.simple_backtest import run_long_only_backtest
from legacy_v1.experiments.ga_indicator_params import folds
from legacy_v1.experiments.judge import judge
from legacy_v1.experiments.ga_indicator_params import _points
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.ga.features import ga_signal_inputs, signals_from_params, ta_values_from_params
from legacy_v1.ga.rules import CANDLE_RULES
from legacy_v1.ga.optimizer import GAConfig
from legacy_v1.model.esn_readout import WASHOUT, esn_states, fit_readout, inner_folds, scores, to_signal
from legacy_v1.model.scaling import MinMaxUnitScaler
from legacy_v1.pipeline import production_config as cfg
from legacy_v1.pipeline.run_walkforward import _known_triangle, _pareto_knee_params

OUT = Path("results/legacy_v1/esn_backtest")
SIGNAL_SHARE = 0.2
GRID = [{"reservoir_size": r, "spectral_radius": sr, "sparsity": 0.15, "input_scale": s, "leaking_rate": lk, "log10_alpha": float(np.log10(a))}
        for r, sr, s, lk, a in itertools.product([50, 100, 200], [0.5, 0.9, 1.1], [0.1, 0.3, 0.7], [0.1, 0.3, 0.7], [20, 200, 2000])]
DEFAULT = {"reservoir_size": 100, "spectral_radius": 0.9, "sparsity": 0.15, "input_scale": 0.3, "leaking_rate": 0.3, "log10_alpha": float(np.log10(200))}


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else float("nan")


def year_returns(close: np.ndarray, open_: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Daily returns of the v1 rule over one stretch, starting flat: the signal of day t is traded at the open of day t+1, buy
    (-1) when flat, sell (+1) when holding, all in / all out, costs of production_config. The same arithmetic as
    simple_backtest.run_long_only_backtest(execution="next_open") without its per-row bookkeeping (checked equal in `_check`)."""
    fee, slip = cfg.FEE_RATE, cfg.SLIPPAGE_RATE
    cash, shares = 1.0, 0.0
    values = np.empty(len(close))
    for i in range(len(close)):
        acted = signal[i - 1] if i > 0 else 0
        if acted == -1 and shares <= 0.0 and cash > 0.0:
            price = open_[i] * (1.0 + slip)
            shares = cash / (price * (1.0 + fee))
            cash -= shares * price * (1.0 + fee)
        elif acted == 1 and shares > 0.0:
            cash += shares * open_[i] * (1.0 - slip) * (1.0 - fee)
            shares = 0.0
        values[i] = cash + shares * close[i]
    out = np.zeros(len(close))
    out[1:] = values[1:] / values[:-1] - 1.0
    return out


def _check() -> None:
    """year_returns against simple_backtest.run_long_only_backtest on random signals."""
    rng = np.random.default_rng(0)
    close = np.cumprod(1 + rng.normal(0, 0.01, 300)) * 100
    open_ = close * (1 + rng.normal(0, 0.003, 300))
    signal = rng.choice([-1, 0, 1], 300, p=[0.1, 0.8, 0.1]).astype(float)
    equity, _, _ = run_long_only_backtest(pd.DataFrame({"close": close, "open": open_, "signal": signal}), price_col="close",
                                          signal_col="signal", fee_rate=cfg.FEE_RATE, slippage_rate=cfg.SLIPPAGE_RATE, execution="next_open")
    reference = equity["portfolio_value"].pct_change().fillna(0.0).to_numpy()
    assert np.allclose(year_returns(close, open_, signal), reference, atol=1e-12), "year_returns differs from run_long_only_backtest"


def ensemble_signal(models_states: list, X: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Signals on `rows` of the seed ensemble: readout per seed on fit_rows, mean score, threshold from the fit rows' scores."""
    fit_scores, row_scores = [], []
    for model, states in models_states:
        w = fit_readout(model, X, states, y, fit_rows)
        fit_scores.append(scores(model, X, states, w, fit_rows))
        row_scores.append(scores(model, X, states, w, rows))
    threshold = float(np.quantile(np.abs(np.mean(fit_scores, axis=0)), 1 - SIGNAL_SHARE))
    return to_signal(np.mean(row_scores, axis=0), threshold)


def run_stock(ticker: str, seeds: list[int], inputs: str = "ta", ga_config: GAConfig = GAConfig(),
              ga_params_dir: str | None = None) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    df = build_ta_features(pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv")).dropna(subset=TA_FEATURE_COLUMNS).reset_index(drop=True)
    dates = pd.to_datetime(df["Date"])
    close, open_ = df["Close"].astype(float).to_numpy(), df["Open"].astype(float).to_numpy()
    X_full = df[TA_FEATURE_COLUMNS].astype(float).to_numpy()
    stored = json.loads((Path(ga_params_dir) / f"{ticker}.json").read_text(encoding="utf-8")) if ga_params_dir else None
    fold_rows, ga_rows, test_signal = [], [], {"selected": np.zeros(len(df)), "default": np.zeros(len(df))}
    test_span = []
    for f in folds(dates):
        t0 = time.time()
        a, b, ts, te = f["train_start"], f["train_end"], f["test_start"], f["test_end"]
        p_value, t_value = _pareto_knee_params(pd.Series(close[a:b]))
        if inputs == "ga" and stored is not None:          # parameters saved by save_ga_params.py for this very fold
            entry = stored["folds"].get(f["test_year"])
            if entry is None:
                raise KeyError(f"{ticker}: no stored parameters for the fold starting {f['test_year']}")
            X = signals_from_params(df, a, te, {k: v["params"] for k, v in entry["rules"].items()})
            ga_info = entry["rules"]
        elif inputs == "ga":                                # GA-tuned rule signals, parameters from the training window only
            X, ga_info = ga_signal_inputs(df, a, te, _points(pd.Series(close), a, b, p_value, t_value), ga_config)
            ga_rows += [{"ticker": ticker, "fold": f["fold"], "rule": k, **{n: v for n, v in g.items() if n != "params"},
                         "params": json.dumps(g["params"])} for k, g in ga_info.items()]
        elif inputs == "ga_ta_candle":
            entry = stored["folds"].get(f["test_year"])
            if entry is None:
                raise KeyError(f"{ticker}: no stored parameters for the fold starting {f['test_year']}")
            rule_params = {k: v["params"] for k, v in entry["rules"].items()}
            ta_raw = ta_values_from_params(df, a, te, rule_params)
            defined = ~np.isnan(ta_raw).any(axis=1)
            if not defined[b - a:].all():
                raise ValueError(f"{ticker} fold {f['fold']}: GA indicators undefined after the training window")
            scaler = MinMaxUnitScaler()
            scaler.fit_transform(ta_raw[:b - a][defined[:b - a]])
            ta_scaled = np.nan_to_num(scaler.transform(ta_raw), nan=0.0)                        # warm-up rows as 0, never fitted
            X = np.hstack([ta_scaled, signals_from_params(df, a, te, rule_params, rules=list(CANDLE_RULES))])
        else:
            scaler = MinMaxUnitScaler()
            X = np.vstack([scaler.fit_transform(X_full[a:b]), scaler.transform(X_full[b:te])])   # training start .. test end
        y_train = _known_triangle(pd.Series(close[a:b]), p_value, t_value)
        y = np.concatenate([y_train, np.full(te - b, np.nan)])                                  # targets only inside training
        known = ~np.isnan(y) & (np.arange(len(y)) >= WASHOUT)
        if inputs == "ga_ta_candle":
            known &= defined
        n_train = b - a
        selection_years = inner_folds(dates.iloc[a:b].reset_index(drop=True)) + [(n_train, ts - a)]   # years 3-5 + validation
        test_rows = np.arange(ts - a, te - a)

        sel_scores, test_scores = [], []
        for params in GRID:
            ms = [esn_states(X, params, s) for s in seeds]
            r = []
            for start, end in selection_years:
                fit_rows = np.flatnonzero(known[:min(start, n_train)])
                sig = ensemble_signal(ms, X, y, fit_rows, np.arange(start, end))
                r.append(year_returns(close[a + start:a + end], open_[a + start:a + end], sig))
            sel_scores.append(sharpe(np.concatenate(r)))
            sig = ensemble_signal(ms, X, y, np.flatnonzero(known[:n_train]), test_rows)
            test_scores.append(sharpe(year_returns(close[ts:te], open_[ts:te], sig)))        # diagnostic only
        sel = np.nan_to_num(np.array(sel_scores), nan=-np.inf)
        best = int(np.argmax(sel))
        d_idx = GRID.index(DEFAULT)
        for label, idx in (("selected", best), ("default", d_idx)):
            ms = [esn_states(X, GRID[idx], s) for s in seeds]
            test_signal[label][ts:te] = ensemble_signal(ms, X, y, np.flatnonzero(known[:n_train]), test_rows)
        test_span.append((ts, te))
        ok = np.isfinite(sel_scores) & np.isfinite(test_scores)
        rho = float(spearmanr(np.array(sel_scores)[ok], np.array(test_scores)[ok]).correlation) if ok.sum() > 10 else float("nan")
        fold_rows.append({"ticker": ticker, "fold": f["fold"], "test_year": f["test_year"], "P": p_value, "T": t_value,
                          "selection_sharpe_selected": sel_scores[best], "selection_sharpe_default": sel_scores[d_idx],
                          "test_year_sharpe_selected": test_scores[best], "test_year_sharpe_default": test_scores[d_idx],
                          "spearman_selection_test": rho, **{f"sel_{k}": v for k, v in GRID[best].items()}})
        print(f"{ticker} fold {f['fold']} ({f['test_year']}): selected {GRID[best]} selection Sharpe {sel_scores[best]:.2f} "
              f"(default {sel_scores[d_idx]:.2f}), spearman {rho:+.2f}, {time.time() - t0:.0f}s", flush=True)

    start, end = test_span[0][0], test_span[-1][1]
    daily = pd.DataFrame(index=pd.DatetimeIndex(dates.iloc[start:end].to_numpy(), name="date"))
    for label in ("selected", "default"):
        daily[label] = year_returns(close[start:end], open_[start:end], test_signal[label][start:end])
    daily["hold"] = pd.Series(close[start:end]).pct_change().fillna(0.0).to_numpy()
    return fold_rows, daily, ga_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--per-market", type=int, default=0, help="If > 0: the 64-stock set of compare_labels.py (core + random US/KR).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--inputs", choices=["ta", "ga", "ga_ta_candle"], default="ta")
    parser.add_argument("--population", type=int, default=30, help="GA population for --inputs ga.")
    parser.add_argument("--generations", type=int, default=20, help="GA generations for --inputs ga.")
    parser.add_argument("--ga-params", default=None, help="For --inputs ga: folder of save_ga_params.py files (data/params); each fold "
                        "uses its stored parameters instead of running the GA.")
    args = parser.parse_args()
    global OUT
    if args.inputs in ("ga", "ga_ta_candle"):
        OUT = OUT.with_name(f"{OUT.name}_{args.inputs}")
    if args.inputs == "ga_ta_candle" and args.ga_params is None:
        args.ga_params = "data/params"
    ga_config = GAConfig(population=args.population, generations=args.generations)
    _check()
    if args.per_market:
        from legacy_v1.experiments.compare_execution import CORE, sample
        picked = sample(args.per_market)
        tickers = list(dict.fromkeys([*CORE, *[s for symbols in picked.values() for s in symbols]]))
    else:
        tickers = args.tickers or ["SPY", "QQQ", "AAPL", "GLD"]
    OUT.mkdir(parents=True, exist_ok=True)

    def safe(t):
        try:
            return run_stock(t, args.seeds, args.inputs, ga_config, args.ga_params)
        except Exception as exc:                       # a stock without enough history is dropped, not fatal
            print(f"{t}: failed ({exc})", flush=True)
            return [], None, []

    results = dict(zip(tickers, Parallel(n_jobs=args.jobs)(delayed(safe)(t) for t in tickers)))
    done = [t for t, (_, d, _) in results.items() if d is not None]
    if args.inputs == "ga":
        pd.DataFrame([row for t in done for row in results[t][2]]).to_csv(OUT / "ga_params.csv", index=False)
    folds_table = pd.DataFrame([row for t in done for row in results[t][0]])
    folds_table.to_csv(OUT / "folds.csv", index=False)
    frames = {label: pd.DataFrame({t: results[t][1][label] for t in done}) for label in ("selected", "default", "hold")}
    for label, frame in frames.items():
        frame.to_csv(OUT / f"daily_{label}.csv")
    compare = pd.DataFrame({label: frame.apply(lambda r: sharpe(r.dropna().to_numpy())) for label, frame in frames.items()})
    compare.to_csv(OUT / "compare.csv")
    verdicts = pd.DataFrame([{"strategy": s, "baseline": b, **judge(frames[s], frames[b])}
                             for s, b in (("selected", "default"), ("selected", "hold"), ("default", "hold"))])
    verdicts.to_csv(OUT / "verdicts.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n== test-period Sharpe by stock ==")
    print(compare.round(3).to_string())
    print("\n== verdicts (median Sharpe difference, 95% month-block interval) ==")
    print(verdicts[["strategy", "baseline", "stocks", "median_sharpe_diff", "ci_lo", "ci_hi", "verdict", "share_stocks_ahead"]].round(3).to_string(index=False))
    print(f"\nSpearman(selection Sharpe, next test-year Sharpe) over candidates: median {folds_table['spearman_selection_test'].median():+.2f}, "
          f"above 0 in {(folds_table['spearman_selection_test'] > 0).mean():.0%} of {len(folds_table)} folds")
    print(f"default chosen in {((folds_table[[f'sel_{k}' for k in DEFAULT]] == pd.Series(DEFAULT).add_prefix('sel_')).all(axis=1)).mean():.0%} of folds")
    print(f"{len(tickers) - len(done)} stocks dropped")


if __name__ == "__main__":
    main()
