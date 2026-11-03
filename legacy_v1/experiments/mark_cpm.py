"""Mark CPM turning points for every stock with a price file, with P and T chosen without future data.

    python -m legacy_v1.experiments.mark_cpm [--jobs 6] [--limit 20]

Rule (one for every stock): P and T are chosen by the Pareto knee of the usual 8 x 9 grid using ONLY prices up to
2018-12-31 (the development period). With those P and T, CPM is run on the stock's whole price history and its points are
stored; the last reported point can still move and is flagged final = False. A stock needs at least three years of prices
before 2019 to have its P and T chosen; otherwise it is listed as skipped. The price files themselves are not touched.

Output: data/labels/cpm/<ticker>.csv (Date, Close, cpm_point_type 'bottom'/'top'/'', turning_label -1/+1/0, final) and
data/labels/cpm/_summary.csv (one row per stock: P, T, number of points, spacing, or why it was skipped).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.labels.cpm import detect_turning_points
from legacy_v1.pipeline.run_walkforward import _pareto_knee_params

PROCESSED = Path("data/processed")
OUT = Path("data/labels/cpm")
CHOOSE_UNTIL = pd.Timestamp("2018-12-31")
MIN_CHOOSE_ROWS = 756


def market_of(ticker: str) -> str:
    return "CRYPTO" if ticker.endswith("USDT") else "KOSDAQ" if ticker.endswith(".KQ") else "KOSPI" if ticker.endswith(".KS") else "US"


def mark(ticker: str) -> dict:
    row = {"ticker": ticker, "market": market_of(ticker)}
    try:
        data = pd.read_csv(PROCESSED / ticker / "1d" / f"{ticker}_with_tp.csv", usecols=["Date", "Close"], parse_dates=["Date"])
    except Exception as exc:
        return {**row, "status": "skipped", "reason": f"unreadable price file ({type(exc).__name__})"}
    data = data.dropna().drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    data = data[data["Close"] > 0].reset_index(drop=True)
    choose = data["Close"][data["Date"] <= CHOOSE_UNTIL]
    row.update({"rows": len(data), "first_date": str(data["Date"].iloc[0].date()) if len(data) else None,
                "last_date": str(data["Date"].iloc[-1].date()) if len(data) else None, "rows_before_2019": len(choose)})
    if len(choose) < MIN_CHOOSE_ROWS:
        return {**row, "status": "skipped", "reason": f"only {len(choose)} rows before 2019 (need {MIN_CHOOSE_ROWS})"}
    p_value, t_value = _pareto_knee_params(choose)
    points = detect_turning_points(data["Close"], P=p_value, T=t_value)
    kind, label, final = np.full(len(data), "", object), np.zeros(len(data), np.int8), np.zeros(len(data), bool)
    for k, point in enumerate(points):
        i = int(point["index"])
        kind[i], label[i], final[i] = point["type"], -1 if point["type"] == "bottom" else 1, k < len(points) - 1
    OUT.mkdir(parents=True, exist_ok=True)
    data.assign(cpm_point_type=kind, turning_label=label, final=final).to_csv(OUT / f"{ticker}.csv", index=False)
    gaps = np.diff([int(p["index"]) for p in points])
    return {**row, "status": "marked", "P": p_value, "T": t_value, "points": len(points), "bottoms": int((label == -1).sum()), "tops": int((label == 1).sum()),
            "median_bars_between_points": float(np.median(gaps)) if len(gaps) else np.nan, "points_per_year": len(points) / (len(data) / 252)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tickers = sorted(p.parent.parent.name for p in PROCESSED.glob("*/1d/*_with_tp.csv") if "_rel" not in p.parent.parent.name)[: args.limit]
    rows = Parallel(n_jobs=args.jobs)(delayed(mark)(t) for t in tickers)
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(summary)} price files: " + ", ".join(f"{k} {v}" for k, v in summary["status"].value_counts().items()))
    marked = summary[summary["status"] == "marked"]
    print(marked.groupby("market").agg(stocks=("ticker", "size"), median_P=("P", "median"), median_T=("T", "median"),
                                      median_points_per_year=("points_per_year", "median"), median_spacing=("median_bars_between_points", "median")).round(2).to_string())
    print("\nmost common (P, T):")
    print(marked.groupby(["P", "T"]).size().sort_values(ascending=False).head(10).rename("stocks").to_string())
    skipped = summary[summary["status"] == "skipped"]
    if len(skipped):
        print("\nskipped by market:", skipped["market"].value_counts().to_dict())
    print(f"\nSaved: {OUT}/<ticker>.csv and {OUT}/_summary.csv")


if __name__ == "__main__":
    main()
