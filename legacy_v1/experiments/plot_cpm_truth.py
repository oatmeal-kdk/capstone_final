"""The CPM triangle on its own: the hindsight 'answer' the CPM-label models were trained to reproduce.

    python -m legacy_v1.experiments.plot_cpm_truth [--tickers SPY QQQ AAPL GLD] [--start 2018-01-01] [--end 2020-12-31]

Turning points come from data/labels/cpm (P and T chosen by the Pareto knee on prices up to 2018, CPM run on the whole history;
the last reported point is not final and is not used). The triangle puts -1 on troughs and +1 on peaks and joins them with
straight lines. Outputs: results/legacy_v1/figures/<ticker>_cpm_truth_<start>_<end>.png and results/legacy_v1/cpm_truth_summary.csv
(swings over the whole history: how far and how long the rising and falling legs go).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from legacy_v1.labels.cpm_target import build_triangle_target

MARKS = Path("data/labels/cpm")
FIGURES = Path("results/legacy_v1/figures")
BUY, SELL, INK, LINE = "#1a9e5a", "#d1453b", "#222222", "#2b6ea8"
plt.rcParams.update({"font.family": ["Malgun Gothic", "sans-serif"], "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e7eb", "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 130})


def load(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    marks = pd.read_csv(MARKS / f"{ticker}.csv", parse_dates=["Date"])
    settings = pd.read_csv(MARKS / "_summary.csv").set_index("ticker").loc[ticker]
    final = marks[(marks["cpm_point_type"].fillna("") != "") & marks["final"]]
    points = [{"index": int(i), "type": t} for i, t in zip(final.index, final["cpm_point_type"])]
    triangle = build_triangle_target(pd.DataFrame(index=marks.index), points).to_numpy(dtype=float).copy()
    triangle[: final.index[0]], triangle[final.index[-1] + 1:] = np.nan, np.nan
    return marks, final, triangle, {"P": float(settings["P"]), "T": int(settings["T"])}


def swings(ticker: str, final: pd.DataFrame, settings: dict, total_days: int) -> dict:
    price, kind, pos = final["Close"].to_numpy(), final["cpm_point_type"].to_numpy(), final.index.to_numpy()
    moves = price[1:] / price[:-1] - 1
    days = np.diff(pos)
    rising = kind[:-1] == "bottom"                                   # a leg that starts at a trough rises to the next peak
    return {"ticker": ticker, "P": settings["P"], "T": settings["T"], "points": len(final), "points_per_year": len(final) / (total_days / 252),
            "rising_leg_median_move": float(np.median(moves[rising])), "falling_leg_median_move": float(np.median(moves[~rising])),
            "rising_leg_median_days": float(np.median(days[rising])), "falling_leg_median_days": float(np.median(days[~rising])),
            "time_in_rising_legs": float(days[rising].sum() / days.sum()), "largest_rise": float(moves[rising].max()), "largest_fall": float(moves[~rising].min())}


def plot(ticker: str, start: str, end: str) -> tuple[Path, dict]:
    marks, final, triangle, settings = load(ticker)
    window = (marks["Date"] >= start) & (marks["Date"] <= end)
    dates, close = marks.loc[window, "Date"], marks.loc[window, "Close"]
    inside = final[(final["Date"] >= start) & (final["Date"] <= end)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(dates, close, color=INK, linewidth=1.1)
    for kind, marker, colour, name in (("bottom", "^", BUY, "저점"), ("top", "v", SELL, "고점")):
        rows = inside[inside["cpm_point_type"] == kind]
        ax.scatter(rows["Date"], rows["Close"], marker=marker, s=70, color=colour, edgecolor="white", linewidth=0.8, zorder=3, label=f"{name} {len(rows)}개")
    previous = None
    for _, row in inside.iterrows():                                  # label each leg with its size
        if previous is not None:
            move = row["Close"] / previous["Close"] - 1
            middle = previous["Date"] + (row["Date"] - previous["Date"]) / 2
            ax.annotate(f"{move:+.0%}", (middle, (row["Close"] + previous["Close"]) / 2), fontsize=7.5, color=BUY if move > 0 else SELL,
                        ha="center", va="bottom" if move > 0 else "top")
        previous = row
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title(f"CPM 전환점 (P={settings['P']:.0%}, T={settings['T']}, 2018년까지의 가격으로 파레토 무릎 선택) — 숫자는 각 흐름의 등락률", loc="left", fontsize=10.5)
    ax = axes[1]
    ax.plot(dates, triangle[window.to_numpy()], color=LINE, linewidth=1.8)
    ax.fill_between(dates, triangle[window.to_numpy()], 0, where=triangle[window.to_numpy()] < 0, color=BUY, alpha=0.12, interpolate=True)
    ax.fill_between(dates, triangle[window.to_numpy()], 0, where=triangle[window.to_numpy()] > 0, color=SELL, alpha=0.12, interpolate=True)
    ax.axhline(0, color="#8a94a3", linewidth=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1], ["−1 (저점)", "0", "+1 (고점)"])
    ax.set_title("정답 삼각파: 저점 −1, 고점 +1, 사이는 직선. 초록은 저점 쪽 절반(매수 구간), 빨강은 고점 쪽 절반(매도 구간)", loc="left", fontsize=10.5)
    fig.suptitle(f"{ticker} {start[:4]}~{end[:4]}: CPM 정답 삼각파", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{ticker}_cpm_truth_{start[:4]}_{end[:4]}.png"
    fig.savefig(path)
    plt.close(fig)
    return path, swings(ticker, final, settings, len(marks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "GLD"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2020-12-31")
    args = parser.parse_args()
    rows = []
    for ticker in args.tickers:
        path, row = plot(ticker, args.start, args.end)
        rows.append(row)
        print(path)
    table = pd.DataFrame(rows)
    table.to_csv(Path("results/legacy_v1/cpm_truth_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
