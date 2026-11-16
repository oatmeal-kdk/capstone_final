"""Fixed ESN settings on the 25 inputs (ga_ta_candle): reservoir 500 / 1000 / 3000 x spectral radius 0.1 / 0.3 / 0.5 / 0.7 / 0.9.
Added 2026-11-12 at the professor's request.

    python -m legacy_v1.experiments.compare_reservoir [--tickers SPY QQQ AAPL GLD] [--seeds 0] [--jobs 4]
                                                      [--reservoirs 500 1000 3000] [--radii 0.1 0.3 0.5 0.7 0.9]

Everything but the two ESN settings is select_esn_backtest.py's design with --inputs ga_ta_candle: the 20 ta indicators with
the fold's GA parameters (min-max scaled on the training window) + the 5 GA candle patterns, label cpm_causal, ridge readout
on [1, inputs, states] (alpha 200, washout 30), threshold = 80th percentile of |score| on the training rows, the v1 rule,
next-open fills with costs, rolling 5y / 1y / 1y folds, seed ensemble over --seeds. Sparsity 0.15, input scale 0.3 and leak 0.3
stay at ESN_CONFIG. Nothing is selected: every setting trades every test year, and the test years are stitched once.
The default ESN (reservoir 100, radius 0.9) and holding are run alongside as references.
Saved in results/legacy_v1/reservoir/ (+ _<tag>): daily_<setting>.csv (daily returns, one column per stock), summary.csv
(per setting and stock: Sharpe, CAGR, MDD, trades), verdicts.csv (each setting against the default ESN and against holding,
legacy_v1/experiments/judge.py), heatmap.png (median Sharpe and MDD over the stocks).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.experiments.ga_indicator_params import folds
from legacy_v1.experiments.judge import judge
from legacy_v1.experiments.select_esn_backtest import DEFAULT, _check, ensemble_signal, fold_inputs, sharpe, year_returns
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.model.esn_readout import WASHOUT, esn_states
from legacy_v1.pipeline.run_walkforward import _known_triangle, _pareto_knee_params

OUT = Path("results/legacy_v1/reservoir")


def label(params: dict) -> str:
    """"default" for ESN_CONFIG, else r<reservoir>_sr<radius>."""
    if all(params[k] == v for k, v in DEFAULT.items()):
        return "default"
    return f"r{params['reservoir_size']}_sr{params['spectral_radius']}"


def run_stock(ticker: str, settings: list[dict], seeds: list[int], ga_params: str) -> dict[str, pd.Series]:
    """{setting label: daily returns over the stitched test years}, plus "hold"."""
    df = build_ta_features(pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv")).dropna(subset=TA_FEATURE_COLUMNS).reset_index(drop=True)
    dates = pd.to_datetime(df["Date"])
    close, open_ = df["Close"].astype(float).to_numpy(), df["Open"].astype(float).to_numpy()
    X_full = df[TA_FEATURE_COLUMNS].astype(float).to_numpy()
    stored = json.loads((Path(ga_params) / f"{ticker}.json").read_text(encoding="utf-8"))
    signal = {label(p): np.zeros(len(df)) for p in settings}
    spans = []
    for f in folds(dates):
        t0 = time.time()
        a, b, ts, te = f["train_start"], f["train_end"], f["test_start"], f["test_end"]
        p_value, t_value = _pareto_knee_params(pd.Series(close[a:b]))
        X, defined = fold_inputs(df, X_full, "ga_ta_candle", stored, f)
        y = np.concatenate([_known_triangle(pd.Series(close[a:b]), p_value, t_value), np.full(te - b, np.nan)])
        fit_rows = np.flatnonzero((~np.isnan(y) & (np.arange(len(y)) >= WASHOUT) & defined)[:b - a])
        test_rows = np.arange(ts - a, te - a)
        for p in settings:
            ms = [esn_states(X, p, s) for s in seeds]
            signal[label(p)][ts:te] = ensemble_signal(ms, X, y, fit_rows, test_rows)
        spans.append((ts, te))
        print(f"{ticker} fold {f['fold']} ({f['test_year']}): {len(settings)} settings, {time.time() - t0:.0f}s", flush=True)
    start, end = spans[0][0], spans[-1][1]
    index = pd.DatetimeIndex(dates.iloc[start:end].to_numpy(), name="date")
    out = {k: pd.Series(year_returns(close[start:end], open_[start:end], s[start:end]), index=index) for k, s in signal.items()}
    out["hold"] = pd.Series(close[start:end], index=index).pct_change().fillna(0.0)
    out["_trades"] = pd.Series({k: int(np.count_nonzero(np.diff(_position(s[start:end])))) for k, s in signal.items()})
    return out


def _position(signal: np.ndarray) -> np.ndarray:
    """0 / 1 holding of the v1 rule (buy when flat, sell when holding), for counting trades."""
    pos, held = np.zeros(len(signal)), 0.0
    for i in range(1, len(signal)):
        if signal[i - 1] == -1:
            held = 1.0
        elif signal[i - 1] == 1:
            held = 0.0
        pos[i] = held
    return pos


def stats(r: pd.Series) -> dict:
    r = r.dropna()                                      # stocks trade on different days, so a joined frame has gaps
    equity = (1 + r).cumprod()
    return {"sharpe": sharpe(r.to_numpy()), "cagr": float(equity.iloc[-1] ** (252 / len(r)) - 1),
            "mdd": float((equity / equity.cummax() - 1).min())}


def heatmap(summary: pd.DataFrame, reservoirs: list[int], radii: list[float], path: Path) -> None:
    med = summary.groupby("setting")[["sharpe", "mdd"]].median()
    ref = {k: med.loc[k] for k in ("default", "hold")}
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title, fmt in ((axes[0], "sharpe", "Sharpe", "{:.2f}"), (axes[1], "mdd", "MDD", "{:.1%}")):
        grid = np.array([[med.loc[label({"reservoir_size": r, "spectral_radius": sr}), col] for sr in radii] for r in reservoirs])
        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto")
        for i in range(len(reservoirs)):
            for j in range(len(radii)):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(radii)), radii)
        ax.set_yticks(range(len(reservoirs)), reservoirs)
        ax.set_xlabel("spectral radius")
        ax.set_ylabel("reservoir size")
        ax.set_title(f"{title} (종목 중앙값)  기본 ESN {fmt.format(ref['default'][col])} / 보유 {fmt.format(ref['hold'][col])}")
        fig.colorbar(im, ax=ax, fraction=0.05)
    stocks = ", ".join(summary["ticker"].unique())
    fig.suptitle(f"입력 25개 (ga_ta_candle), 테스트 기간 이어붙임 — {stocks}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "GLD"])
    parser.add_argument("--reservoirs", nargs="+", type=int, default=[500, 1000, 3000])
    parser.add_argument("--radii", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--ga-params", default="data/params")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    out = OUT.with_name(f"{OUT.name}_{args.tag}") if args.tag else OUT
    out.mkdir(parents=True, exist_ok=True)
    _check()
    settings = [{**DEFAULT, "reservoir_size": r, "spectral_radius": sr} for r, sr in itertools.product(args.reservoirs, args.radii)]
    if DEFAULT not in settings:
        settings.append(DEFAULT)
    results = dict(zip(args.tickers, Parallel(n_jobs=args.jobs)(delayed(run_stock)(t, settings, args.seeds, args.ga_params)
                                                                for t in args.tickers)))
    frames = {k: pd.DataFrame({t: res[k] for t, res in results.items()}) for k in [*map(label, settings), "hold"]}
    for k, frame in frames.items():
        frame.to_csv(out / f"daily_{k}.csv")
    rows = []
    for k, frame in frames.items():
        for t in args.tickers:
            trades = None if k == "hold" else int(results[t]["_trades"][k])
            rows.append({"setting": k, "ticker": t, **stats(frame[t]), "trades": trades})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    verdicts = pd.DataFrame([{"setting": k, "baseline": base, **judge(frames[k], frames[base])}
                             for k in frames if k not in ("default", "hold") for base in ("default", "hold")])
    verdicts.to_csv(out / "verdicts.csv", index=False)
    heatmap(summary, args.reservoirs, args.radii, out / "heatmap.png")

    pd.set_option("display.width", 220)
    print("\n== Sharpe by setting and stock ==")
    print(summary.pivot(index="setting", columns="ticker", values="sharpe")[args.tickers].assign(median=lambda d: d.median(axis=1)).round(3).to_string())
    print("\n== MDD by setting and stock ==")
    print(summary.pivot(index="setting", columns="ticker", values="mdd")[args.tickers].assign(median=lambda d: d.median(axis=1)).round(3).to_string())
    print("\n== against the default ESN (median Sharpe difference, 95% month-block interval) ==")
    print(verdicts[verdicts["baseline"] == "default"][["setting", "median_sharpe_diff", "ci_lo", "ci_hi", "verdict", "median_mdd_diff"]].round(3).to_string(index=False))
    print(f"\nsaved in {out}")


if __name__ == "__main__":
    main()
