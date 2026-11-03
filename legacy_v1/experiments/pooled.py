"""The v1 recipe trained on many stocks at once: CPM triangle vs triple barrier, raw vs rolling-z inputs, per market.

    python -m legacy_v1.experiments.pooled [--markets KOSPI KOSDAQ US] [--limit 40] [--jobs 6]

Declared before running; nothing below was changed after seeing results.
  Stocks      the cleaned universe of each market (pipeline_v2, survivor-only: stocks listed today).
  Inputs      the 20 ta indicators of the v1 recipe, either as they are ("raw") or as a rolling z-score against the stock's own
              past 252 days, clipped at +-5 ("z"). Both are then min-max scaled to [-1, 1] on the pooled training rows of each fold.
              Both variants use exactly the same rows (a row needs a full 252-day history for the z-score).
  Labels      triple barrier: ATR(14) x 2, 10 days (the locked label).
              CPM triangle: per stock and per fold, P and T chosen by the Pareto knee on that stock's training prices only;
              training targets from a CPM run ending at the training end, validation targets from one ending at the validation
              end; the last reported turning point and everything after it is left out.
  Folds       test year Y = 2016 ... 2026 (2026 up to the last price); validation = Y-1; training = everything before. One model
              per market, fold, label and input variant, trained on every 5th day of every stock.
  Model       the v1 ESN reservoir (100 units, spectral radius 0.9, sparsity 0.15, input scale 0.3, leak 0.3, seed 42) read over the
              last 20 days of each row from a zero state (stocks cannot be chained into one sequence), ridge readout alpha 200 on
              [1, today's inputs, state]. Threshold chosen on validation accuracy as in v1.
  Trading     per stock, the v1 all-in/all-out rule and the v1 volatility-target overlay (10%), fills at the next day's open,
              costs 0.05% + 0.05%. Baselines: hold the stock, and hold it under the same overlay.
  Verdict     legacy_v1/experiments/judge.py (declared before the results): median over stocks of the Sharpe difference to the
              baseline, 95% calendar-month block-bootstrap interval; above 0 = wins. Primary comparison: each model with the
              overlay against hold + overlay. Also shown: each model without the overlay against plain hold.
Note: the test years run to 2026 like the earlier v1 evaluation, so they include the years after 2018 that the newer
experiments kept unread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.backtest.simple_backtest import run_long_only_backtest, run_long_only_backtest_vol_target
from legacy_v1.experiments.judge import judge
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.labels.triple_barrier import label_triple_barrier
from legacy_v1.model.esn import EchoStateNetwork
from legacy_v1.model.metrics import score_to_signal
from legacy_v1.model.scaling import MinMaxUnitScaler
from legacy_v1.model.thresholds import _select_thresholds
from legacy_v1.pipeline import production_config as cfg
from legacy_v1.pipeline.run_walkforward import ESN_CONFIG, _known_triangle, _pareto_knee_params

OUT = Path("results/legacy_v1/pooled")
TEST_YEARS = list(range(2016, 2027))
WINDOW, NORMAL, STRIDE, HOLD = 20, 252, 5, 10
LABELS, INPUTS = ("triple_barrier", "cpm"), ("raw", "z")
CORE = ["SPY", "QQQ", "AAPL", "GLD"]
UNIVERSE = Path(__file__).resolve().parents[1] / "universe"


def market_tickers(market_name: str) -> list[str]:
    """The cleaned universe of one market, in the column order of pipeline_v2's cached panel (exported to universe/pooled_<market>.txt)."""
    return (UNIVERSE / f"pooled_{market_name}.txt").read_text(encoding="utf-8").split()


def prepare(ticker: str, with_cpm: bool = True) -> dict | None:
    """One stock: inputs, labels for every fold, prices for trading. Runs in a worker process. with_cpm=False skips the CPM
    targets (the per-fold Pareto-knee search is the slow part)."""
    try:
        df = pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv", usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    except Exception:
        return None
    df = df.dropna().drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)].reset_index(drop=True)
    if len(df) < 2000:
        return None
    df = build_ta_features(df)
    raw = df[TA_FEATURE_COLUMNS].astype(float)
    z = ((raw - raw.rolling(NORMAL, min_periods=NORMAL).mean()) / raw.rolling(NORMAL, min_periods=NORMAL).std().replace(0, np.nan)).clip(-5, 5)
    ok = np.isfinite(raw.to_numpy()).all(axis=1) & np.isfinite(z.to_numpy()).all(axis=1)
    complete = pd.Series(ok.astype(float)).rolling(WINDOW, min_periods=WINDOW).sum().to_numpy() == WINDOW
    dates = pd.to_datetime(df["Date"]).to_numpy()
    close = df["Close"].astype(float)
    tb = label_triple_barrier(df, k=cfg.TRIPLE_BARRIER_K, max_holding=cfg.TRIPLE_BARRIER_MAX_HOLDING).to_numpy(dtype=float)
    cpm = {}
    for year in (TEST_YEARS if with_cpm else []):
        train_cut = int(np.searchsorted(dates, np.datetime64(f"{year - 1}-01-01")))    # rows before the validation year
        val_cut = int(np.searchsorted(dates, np.datetime64(f"{year}-01-01")))
        if train_cut < 750:
            continue
        p_value, t_value = _pareto_knee_params(close.iloc[:train_cut])
        cpm[year] = {"P": p_value, "T": t_value, "train": _known_triangle(close.iloc[:train_cut], p_value, t_value).astype(np.float32),
                     "val": _known_triangle(close.iloc[:val_cut], p_value, t_value).astype(np.float32)}
    log_ret = np.log(close / close.shift(1))
    return {"ticker": ticker, "dates": dates, "rows": np.flatnonzero(complete), "raw": raw.to_numpy(np.float32), "z": z.fillna(0).to_numpy(np.float32),
            "tb": tb.astype(np.float32), "cpm": cpm, "close": close.to_numpy(), "open": df["Open"].astype(float).to_numpy(),
            "vol": (log_ret.rolling(20).std() * np.sqrt(252.0)).to_numpy()}


class Market:
    """All stocks of one market stacked: features are kept per stock (one big array with offsets) so 20-day windows never
    cross from one stock into the next."""

    def __init__(self, parts: list[dict]):
        self.parts = parts
        self.offset = np.cumsum([0] + [len(p["raw"]) for p in parts])[:-1]
        self.features = {kind: np.concatenate([p[kind] for p in parts]) for kind in INPUTS}
        self.stock = np.concatenate([np.full(len(p["rows"]), i, np.int32) for i, p in enumerate(parts)])
        self.pos = np.concatenate([p["rows"] for p in parts]).astype(np.int64)
        self.date = np.concatenate([p["dates"][p["rows"]] for p in parts])

    def window(self, rows: np.ndarray, kind: str) -> np.ndarray:
        g = self.offset[self.stock[rows]] + self.pos[rows]
        return self.features[kind][g[:, None] + np.arange(-WINDOW + 1, 1)]

    def labels(self, rows: np.ndarray, label: str, year: int, part: str) -> np.ndarray:
        """Targets of `rows` known at the end of the training ('train') or validation ('val') part of fold `year`; NaN if unknown."""
        cut = np.datetime64(f"{year - 1 if part == 'train' else year}-01-01")
        out = np.full(len(rows), np.nan, np.float32)
        for i in np.unique(self.stock[rows]):
            sel = np.flatnonzero(self.stock[rows] == i)
            p, pos = self.parts[i], self.pos[rows[sel]]
            if label == "triple_barrier":
                n_known = int(np.searchsorted(p["dates"], cut))
                out[sel] = np.where(pos + HOLD < n_known, p["tb"][pos], np.nan)
            elif year in p["cpm"]:
                target = p["cpm"][year][part]
                out[sel] = np.where(pos < len(target), target[np.minimum(pos, len(target) - 1)], np.nan)
        return out


def design(market: Market, rows: np.ndarray, kind: str, scaler: MinMaxUnitScaler, w_in: np.ndarray, w: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    leak = float(ESN_CONFIG["leaking_rate"])
    out = np.empty((len(rows), 1 + w_in.shape[1] + w.shape[0]), np.float32)
    for start in range(0, len(rows), chunk):
        seq = scaler.transform(market.window(rows[start:start + chunk], kind).reshape(-1, w_in.shape[1]).astype(np.float64)).reshape(-1, WINDOW, w_in.shape[1])
        state = np.zeros((len(seq), w.shape[0]))
        for step in range(WINDOW):
            state = (1.0 - leak) * state + leak * np.tanh(seq[:, step] @ w_in.T + state @ w.T)
        out[start:start + chunk] = np.hstack([np.ones((len(seq), 1)), seq[:, -1], state])
    return out


def ridge(a: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram, moment = np.zeros((a.shape[1], a.shape[1])), np.zeros(a.shape[1])
    for s in range(0, len(a), 200_000):
        block = a[s:s + 200_000].astype(np.float64)
        gram += block.T @ block
        moment += block.T @ y[s:s + 200_000]
    penalty = alpha * np.eye(a.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, moment)


def predict(a: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.clip(np.concatenate([a[s:s + 200_000].astype(np.float64) @ beta for s in range(0, len(a), 200_000)]), -1.0, 1.0)


def stats(returns: pd.Series) -> dict:
    years = len(returns) / 252
    equity = (1 + returns).cumprod()
    return {"sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan,
            "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else np.nan, "mdd": float((equity / equity.cummax() - 1).min())}


def trade(ticker: str, frame: pd.DataFrame, signals: dict) -> tuple[list[dict], dict]:
    """Backtests of one stock: every model signal with and without the overlay, plus the two baselines.
    Returns (summary rows, {(strategy, overlay): daily return series})."""
    rows, daily = [], {}
    dates = pd.to_datetime(frame["date"])
    kw = {"price_col": "close", "fee_rate": cfg.FEE_RATE, "slippage_rate": cfg.SLIPPAGE_RATE, "execution": "next_open"}
    overlay = {"vol_col": "vol", "target_vol": cfg.RISK_OVERLAY_TARGET_VOL, "rebalance_band": cfg.RISK_OVERLAY_REBALANCE_BAND}
    for name, signal in signals.items():
        f = frame.assign(signal=signal)
        plain = run_long_only_backtest(f, initial_cash=1.0, **kw)[0]
        sized = run_long_only_backtest_vol_target(f, initial_cash=1.0, **kw, **overlay)[0]
        for overlaid, curve in ((False, plain), (True, sized)):
            returns = pd.Series(curve["portfolio_value"].pct_change().fillna(0).to_numpy(), index=dates)
            daily[(name, overlaid)] = returns
            rows.append({"ticker": ticker, "strategy": name, "overlay": overlaid, "exposure": float((curve["position"] > 0).mean()), **stats(returns)})
    hold_overlay = run_long_only_backtest_vol_target(frame.assign(signal=-1), initial_cash=1.0, **kw, **overlay)[0]
    daily[("hold", False)] = pd.Series(frame["close"].pct_change().fillna(0).to_numpy(), index=dates)
    daily[("hold", True)] = pd.Series(hold_overlay["portfolio_value"].pct_change().fillna(0).to_numpy(), index=dates)
    rows.append({"ticker": ticker, "strategy": "hold", "overlay": False, "exposure": 1.0, **stats(daily[("hold", False)])})
    rows.append({"ticker": ticker, "strategy": "hold", "overlay": True, "exposure": 1.0, **stats(daily[("hold", True)])})
    return rows, daily


def run_market(market_name: str, jobs: int, limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = market_tickers(market_name)[:limit]
    parts = [p for p in Parallel(n_jobs=jobs)(delayed(prepare)(t) for t in tickers) if p is not None and len(p["rows"])]
    market = Market(parts)
    reservoir = EchoStateNetwork(input_dim=len(TA_FEATURE_COLUMNS), reservoir_size=int(ESN_CONFIG["reservoir_size"]), spectral_radius=float(ESN_CONFIG["spectral_radius"]),
                                 sparsity=float(ESN_CONFIG["sparsity"]), input_scale=float(ESN_CONFIG["input_scale"]), leaking_rate=float(ESN_CONFIG["leaking_rate"]),
                                 ridge_alpha=float(ESN_CONFIG["ridge_alpha"]), washout=0, random_state=int(ESN_CONFIG["random_state"]))
    w_in, w, alpha = reservoir.Win, reservoir.W, float(ESN_CONFIG["ridge_alpha"])
    years = market.date.astype("datetime64[Y]").astype(int) + 1970
    sampled = (market.pos + market.stock) % STRIDE == 0
    scores = {(label, kind): np.full(len(market.date), np.nan, np.float32) for label in LABELS for kind in INPUTS}
    folds = []
    for year in TEST_YEARS:
        train = np.flatnonzero(sampled & (years < year - 1))
        val, test = np.flatnonzero(years == year - 1), np.flatnonzero(years == year)
        if len(train) < 5000 or not len(test):
            continue
        for kind in INPUTS:
            scaler = MinMaxUnitScaler().fit(market.window(train, kind)[:, -1].astype(np.float64))      # fitted on training rows only
            a_train, a_val, a_test = (design(market, r, kind, scaler, w_in, w) for r in (train, val, test))
            for label in LABELS:
                y_train, y_val = market.labels(train, label, year, "train"), market.labels(val, label, year, "val")
                fit, check = np.isfinite(y_train), np.isfinite(y_val)
                beta = ridge(a_train[fit], y_train[fit].astype(np.float64), alpha)
                chosen = _select_thresholds(y_val[check], predict(a_val[check], beta), list(ESN_CONFIG["threshold_values"])) if check.any() else {"threshold": 0.1, "validation_accuracy": None}
                scores[(label, kind)][test] = predict(a_test, beta)
                folds.append({"market": market_name, "year": year, "label": label, "inputs": kind, "train_rows": int(fit.sum()), "val_rows": int(check.sum()),
                              "threshold": chosen["threshold"], "val_accuracy": chosen["validation_accuracy"]})
                market_thresholds = folds[-1]["threshold"]
                scores[(label, kind, "thr", year)] = market_thresholds
        print(f"  [{market_name}] fold {year}: {len(train):,} training rows, {len(test):,} test rows", flush=True)

    trades = []
    jobs_list = []
    for i, p in enumerate(parts):
        mine = np.flatnonzero((market.stock == i) & (years >= TEST_YEARS[0]))
        if len(mine) < 252:
            continue
        pos = market.pos[mine]
        frame = pd.DataFrame({"date": p["dates"][pos], "close": p["close"][pos], "open": p["open"][pos], "vol": p["vol"][pos]})
        signals = {}
        for label in LABELS:
            for kind in INPUTS:
                score = scores[(label, kind)][mine]
                thr = np.array([scores.get((label, kind, "thr", int(y)), 0.1) for y in years[mine]])
                signals[f"{label}/{kind}"] = np.where(np.isfinite(score), np.where(score < -thr, -1, np.where(score > thr, 1, 0)), 0)
        jobs_list.append((p["ticker"], frame, signals))
    daily: dict = {}
    for (rows, series), (ticker, _, _) in zip(Parallel(n_jobs=jobs)(delayed(trade)(t, f, s) for t, f, s in jobs_list), jobs_list):
        trades.extend(rows)
        for key, values in series.items():
            daily.setdefault(key, {})[ticker] = values
    daily = {key: pd.DataFrame(columns).sort_index() for key, columns in daily.items()}
    np.savez_compressed(OUT / f"pooled_daily_{market_name}{'_first' + str(limit) if limit else ''}.npz",
                        **{f"{name}|{overlay}": frame.to_numpy(np.float32) for (name, overlay), frame in daily.items()},
                        dates=next(iter(daily.values())).index.strftime("%Y-%m-%d").to_numpy(), tickers=np.array(next(iter(daily.values())).columns))
    verdicts = []
    for (name, overlay), frame in daily.items():
        if name == "hold":
            continue
        verdicts.append({"market": market_name, "strategy": name, "overlay": overlay, "baseline": "hold + overlay" if overlay else "hold",
                         **judge(frame, daily[("hold", overlay)])})
    table = pd.DataFrame(trades).assign(market=market_name)
    table.attrs["verdicts"] = verdicts
    cpm_settings = pd.DataFrame([{"market": market_name, "ticker": p["ticker"], "year": y, "P": v["P"], "T": v["T"]} for p in parts for y, v in p["cpm"].items()])
    return table, pd.DataFrame(folds).merge(cpm_settings.groupby(["market", "year"])[["P", "T"]].median().reset_index(), on=["market", "year"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["US", "KOSPI", "KOSDAQ"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_first{args.limit}" if args.limit else ""
    tables, folds, verdicts = [], [], []
    for name in args.markets:
        table, fold = run_market(name, args.jobs, args.limit)
        verdicts.extend(table.attrs["verdicts"])
        tables.append(table)
        folds.append(fold)
        pd.concat(tables).to_csv(OUT / f"pooled_results{tag}.csv", index=False)
        pd.concat(folds).to_csv(OUT / f"pooled_folds{tag}.csv", index=False)
        pd.DataFrame(verdicts).to_csv(OUT / f"pooled_verdicts{tag}.csv", index=False)
    table = pd.concat(tables, ignore_index=True)
    pd.set_option("display.width", 220)
    for name in args.markets:
        part = table[table["market"] == name]
        wide = part.pivot_table(index="ticker", columns=["strategy", "overlay"], values="sharpe")
        med = wide.median().unstack("overlay").rename(columns={False: "plain", True: "with overlay"})
        print(f"\n== {name}: {len(wide)} stocks, median Sharpe (test years {TEST_YEARS[0]}-{TEST_YEARS[-1]}) ==")
        print(med.round(3).to_string())
        mine = pd.DataFrame([v for v in verdicts if v["market"] == name])
        print("verdicts (median Sharpe difference to the baseline, 95% month-block interval; share of stocks ahead is shown, not used):")
        print(mine.drop(columns="market").round(3).to_string(index=False))
        core = part[part["ticker"].isin(CORE)]
        if len(core):
            print(core.pivot_table(index=["ticker"], columns=["strategy", "overlay"], values="sharpe").round(3).to_string())
    (OUT / f"pooled_meta{tag}.json").write_text(json.dumps({"test_years": TEST_YEARS, "window": WINDOW, "normalization": NORMAL, "stride": STRIDE,
                                                            "esn": {k: v for k, v in ESN_CONFIG.items()}, "markets": args.markets, "limit": args.limit}, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT}/pooled_results{tag}.csv, pooled_folds{tag}.csv")


if __name__ == "__main__":
    main()
