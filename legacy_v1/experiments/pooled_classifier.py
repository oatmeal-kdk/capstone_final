"""Triple barrier without the 'mostly hold' problem: classify instead of regress, fix the signal rate, hold only while it is on.

    python -m legacy_v1.experiments.pooled_classifier [--markets US KOSPI KOSDAQ] [--limit 40] [--jobs 6]

Declared before running; nothing below was changed after seeing results. Everything not listed is as in pooled.py (stocks,
raw ta inputs, folds 2016-2026, v1 reservoir over 20-day windows, fills at the next open, costs, overlay, baselines, verdict).
  Label       the locked triple barrier (ATR(14) x 2, 10 days); -1 = upper barrier first. Not changed.
  Readouts    tb_cls   multinomial logistic regression on [today's inputs, reservoir state] -> P(upper barrier first)
              tb_reg   the v1 ridge regression on the same design (score; lower = more bullish), to separate the effect of the
                       readout from the effect of the new threshold and rule
  Threshold   the 80th percentile of the TRAINING scores (P(up) for tb_cls, -score for tb_reg): the signal is on for about
              20% of training days. Not chosen on validation accuracy any more.
  Rule        hold while the signal is on, otherwise be in cash (the v1 rule kept a position until a rare sell signal).
Inputs: raw only (they were slightly better than rolling z in pooled.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from legacy_v1.experiments.judge import judge
from legacy_v1.experiments.pooled import CORE, STRIDE, TEST_YEARS, Market, design, market_tickers, predict, prepare, ridge, trade
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS
from legacy_v1.model.esn import EchoStateNetwork
from legacy_v1.model.scaling import MinMaxUnitScaler
from legacy_v1.pipeline.run_walkforward import ESN_CONFIG

OUT = Path("results/legacy_v1/pooled_classifier")
ON_SHARE = 0.20
KIND = "raw"
MODELS = ("tb_cls", "tb_reg")


def run_market(name: str, jobs: int, limit: int | None) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    tickers = market_tickers(name)[:limit]
    parts = [p for p in Parallel(n_jobs=jobs)(delayed(prepare)(t, False) for t in tickers) if p is not None and len(p["rows"])]
    market = Market(parts)
    esn = EchoStateNetwork(input_dim=len(TA_FEATURE_COLUMNS), reservoir_size=int(ESN_CONFIG["reservoir_size"]), spectral_radius=float(ESN_CONFIG["spectral_radius"]),
                           sparsity=float(ESN_CONFIG["sparsity"]), input_scale=float(ESN_CONFIG["input_scale"]), leaking_rate=float(ESN_CONFIG["leaking_rate"]),
                           ridge_alpha=float(ESN_CONFIG["ridge_alpha"]), washout=0, random_state=int(ESN_CONFIG["random_state"]))
    years = market.date.astype("datetime64[Y]").astype(int) + 1970
    sampled = (market.pos + market.stock) % STRIDE == 0
    bullish = {m: np.full(len(market.date), np.nan, np.float32) for m in MODELS}        # higher = more bullish, for both models
    limits, folds = {m: {} for m in MODELS}, []
    for year in TEST_YEARS:
        train, test = np.flatnonzero(sampled & (years < year - 1)), np.flatnonzero(years == year)
        if len(train) < 5000 or not len(test):
            continue
        scaler = MinMaxUnitScaler().fit(market.window(train, KIND)[:, -1].astype(np.float64))
        a_train, a_test = design(market, train, KIND, scaler, esn.Win, esn.W), design(market, test, KIND, scaler, esn.Win, esn.W)
        y = market.labels(train, "triple_barrier", year, "train")
        fit = np.isfinite(y)
        classifier = LogisticRegression(C=1.0, max_iter=300, tol=1e-3).fit(a_train[fit, 1:], y[fit].astype(int))       # column 0 is the constant
        up = list(classifier.classes_).index(-1)
        beta = ridge(a_train[fit], y[fit].astype(np.float64), float(ESN_CONFIG["ridge_alpha"]))
        train_scores = {"tb_cls": classifier.predict_proba(a_train[fit, 1:])[:, up], "tb_reg": -predict(a_train[fit], beta)}
        test_scores = {"tb_cls": classifier.predict_proba(a_test[:, 1:])[:, up], "tb_reg": -predict(a_test, beta)}
        for model in MODELS:
            limits[model][year] = float(np.quantile(train_scores[model], 1 - ON_SHARE))           # from training scores only
            bullish[model][test] = test_scores[model]
            folds.append({"market": name, "year": year, "model": model, "train_rows": int(fit.sum()), "threshold": limits[model][year],
                          "test_on_share": float((test_scores[model] >= limits[model][year]).mean()),
                          "train_label_shares": {int(k): float(v) for k, v in zip(*np.unique(y[fit], return_counts=True))}})
        print(f"  [{name}] fold {year}: {int(fit.sum()):,} training rows, {len(test):,} test rows; signal on for "
              + ", ".join(f"{m} {folds[-len(MODELS) + i]['test_on_share']:.0%}" for i, m in enumerate(MODELS)) + " of test days", flush=True)

    jobs_list = []
    for i, p in enumerate(parts):
        mine = np.flatnonzero((market.stock == i) & (years >= TEST_YEARS[0]))
        if len(mine) < 252:
            continue
        pos = market.pos[mine]
        frame = pd.DataFrame({"date": p["dates"][pos], "close": p["close"][pos], "open": p["open"][pos], "vol": p["vol"][pos]})
        signals = {}
        for model in MODELS:
            threshold = np.array([limits[model].get(int(y), np.inf) for y in years[mine]])
            on = np.isfinite(bullish[model][mine]) & (bullish[model][mine] >= threshold)
            signals[model] = np.where(on, -1, 1)        # -1 = be long; +1 = be in cash (exit if holding). The position follows the signal.
        jobs_list.append((p["ticker"], frame, signals))
    rows, daily = [], {}
    for (summary, series), (ticker, _, _) in zip(Parallel(n_jobs=jobs)(delayed(trade)(t, f, s) for t, f, s in jobs_list), jobs_list):
        rows.extend(summary)
        for key, values in series.items():
            daily.setdefault(key, {})[ticker] = values
    daily = {key: pd.DataFrame(columns).sort_index() for key, columns in daily.items()}
    verdicts = [{"market": name, "model": model, "overlay": overlay, "baseline": "hold + overlay" if overlay else "hold", **judge(frame, daily[("hold", overlay)])}
                for (model, overlay), frame in daily.items() if model != "hold"]
    return pd.DataFrame(rows).assign(market=name), verdicts, pd.DataFrame(folds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["US", "KOSPI", "KOSDAQ"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_first{args.limit}" if args.limit else ""
    tables, verdicts, folds = [], [], []
    pd.set_option("display.width", 220)
    for name in args.markets:
        table, verdict, fold = run_market(name, args.jobs, args.limit)
        tables.append(table)
        verdicts.extend(verdict)
        folds.append(fold)
        pd.concat(tables).to_csv(OUT / f"results{tag}.csv", index=False)
        pd.DataFrame(verdicts).to_csv(OUT / f"verdicts{tag}.csv", index=False)
        pd.concat(folds).to_csv(OUT / f"folds{tag}.csv", index=False)
        wide = table.pivot_table(index="ticker", columns=["strategy", "overlay"], values=["sharpe", "cagr", "mdd", "exposure"])
        print(f"\n== {name}: {len(wide.index)} stocks, medians ==")
        print(wide.median().unstack("overlay").rename(columns={False: "plain", True: "with overlay"}).round(3).to_string())
        print("verdicts (median Sharpe difference to the baseline, 95% month-block interval):")
        print(pd.DataFrame(verdict).drop(columns="market").round(3).to_string(index=False))
        core = table[table["ticker"].isin(CORE)]
        if len(core):
            print(core.pivot_table(index="ticker", columns=["strategy", "overlay"], values="sharpe").round(2).to_string())
    print(f"\nSaved in {OUT}: results{tag}.csv, verdicts{tag}.csv, folds{tag}.csv")


if __name__ == "__main__":
    main()
