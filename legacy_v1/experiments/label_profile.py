"""What the two labels look like: CPM triangle (P, T chosen on prices up to 2018) vs the locked triple barrier, per market.

    python -m legacy_v1.experiments.label_profile [--per-market 150]

Descriptive only (no model, no trading). For a random sample of stocks (fixed seed) with CPM marks in data/labels/cpm/:
how often each label takes each value, how often it changes, how far ahead it looks, and how often the two agree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.labels.cpm_target import build_triangle_target
from legacy_v1.labels.triple_barrier import label_triple_barrier

MARKS = Path("data/labels/cpm")
OUT = Path("results/legacy_v1/label_profile.csv")


def profile(ticker: str) -> dict | None:
    marks = pd.read_csv(MARKS / f"{ticker}.csv", parse_dates=["Date"])
    prices = pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv", usecols=["Date", "Open", "High", "Low", "Close", "Volume"], parse_dates=["Date"])
    prices = prices.dropna().drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    prices = prices[prices["Date"].isin(marks["Date"])].reset_index(drop=True)
    if len(prices) != len(marks):
        return None
    final = marks.index[(marks["cpm_point_type"].fillna("") != "") & marks["final"]].to_numpy()
    if len(final) < 3:
        return None
    points = [{"index": int(i), "type": marks.loc[i, "cpm_point_type"]} for i in final]
    triangle = build_triangle_target(pd.DataFrame(index=marks.index), points).to_numpy(dtype=float).copy()
    triangle[: final[0]], triangle[final[-1] + 1:] = np.nan, np.nan
    tb = label_triple_barrier(prices, k=2.0, max_holding=10).to_numpy(dtype=float)
    ahead = np.searchsorted(final, np.arange(len(marks)), side="right")                  # the next final point decides today's CPM value
    reach = np.where(ahead < len(final), final[np.minimum(ahead, len(final) - 1)] - np.arange(len(marks)), np.nan)
    both = np.isfinite(triangle) & np.isfinite(tb) & (tb != 0)
    valid_tb = np.isfinite(tb)
    years = len(marks) / 252
    return {"ticker": ticker, "years": years,
            "tb_up": float((tb[valid_tb] == -1).mean()), "tb_zero": float((tb[valid_tb] == 0).mean()), "tb_down": float((tb[valid_tb] == 1).mean()),
            "tb_changes_per_year": float((np.diff(tb[valid_tb]) != 0).sum() / years),
            "cpm_points_per_year": len(final) / years, "cpm_median_spacing": float(np.median(np.diff(final))),
            "cpm_label_look_ahead_median": float(np.nanmedian(reach)), "cpm_label_look_ahead_p90": float(np.nanpercentile(reach, 90)),
            "sign_agreement": float((np.sign(triangle[both]) == np.sign(tb[both])).mean()),
            "corr": float(np.corrcoef(triangle[both], tb[both])[0, 1]) if both.sum() > 30 else np.nan}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-market", type=int, default=150)
    args = parser.parse_args()
    summary = pd.read_csv(MARKS / "_summary.csv")
    marked = summary[summary["status"] == "marked"]
    picked = pd.concat([part.sample(min(args.per_market, len(part)), random_state=0) for market, part in marked.groupby("market") if market != "CRYPTO"])
    rows = [r for r in Parallel(n_jobs=6)(delayed(profile)(t) for t in picked["ticker"]) if r is not None]
    table = pd.DataFrame(rows).merge(picked[["ticker", "market", "P", "T"]], on="ticker")
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(table.drop(columns=["ticker"]).groupby("market").median().round(3).T.to_string())
    print(f"\n{len(table)} stocks. Saved: {OUT}")


if __name__ == "__main__":
    main()
