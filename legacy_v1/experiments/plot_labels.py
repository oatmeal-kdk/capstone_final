"""Where do the two labels put their marks? CPM turning points and triple-barrier labels on the same price chart.

    python -m legacy_v1.experiments.plot_labels [--ticker SPY] [--years 2018 2020]

For each year: CPM's P and T are chosen by the Pareto knee on every price before the year that precedes it (cpm_causal trains
on all earlier rows and keeps the last year for validation), and
CPM is run on prices up to the end of the year; its last reported point is not final and is not drawn as a label. The triple
barrier (ATR x 2, 10 days) is computed on the same prices. Bottom panel: the two training targets the ESN would regress on
(sign convention of the code: -1 = buy, +1 = sell). Output: results/legacy_v1/figures/<ticker>_cpm_vs_tb_<year>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from legacy_v1.labels.cpm import detect_turning_points
from legacy_v1.labels.triple_barrier import label_triple_barrier
from legacy_v1.pipeline.run_walkforward import _known_triangle, _pareto_knee_params

OUT = Path("results/legacy_v1/figures")
BUY, SELL, NEUTRAL, INK = "#1a9e5a", "#d1453b", "#b8bec7", "#222222"
plt.rcParams.update({"font.family": ["Malgun Gothic", "sans-serif"], "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e7eb", "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 130})


def plot_year(ticker: str, year: int) -> Path:
    data = pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    end = data.index[data["Date"] <= f"{year}-12-31"][-1]
    start = data.index[data["Date"] >= f"{year}-01-01"][0]
    train_end = data.index[data["Date"] >= f"{year - 1}-01-01"][0]                  # cpm_causal: all rows before the validation year
    upto = data.iloc[: end + 1].reset_index(drop=True)
    close = upto["Close"].astype(float)

    p_value, t_value = _pareto_knee_params(close.iloc[:train_end])
    points = detect_turning_points(close, P=p_value, T=t_value)
    settled, pending = points[:-1], points[-1:]
    triangle = _known_triangle(close, p_value, t_value)
    barrier = label_triple_barrier(upto, k=2.0, max_holding=10).to_numpy(dtype=float)

    window = slice(start, end + 1)
    dates, price = upto["Date"].iloc[window], close.iloc[window]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2]})

    ax = axes[0]
    ax.plot(dates, price, color=INK, linewidth=1.2)
    for point in settled:
        i = int(point["index"])
        if start <= i <= end:
            bottom = point["type"] == "bottom"
            ax.scatter(upto["Date"].iloc[i], close.iloc[i], marker="^" if bottom else "v", s=70, color=BUY if bottom else SELL, zorder=3,
                       edgecolor="white", linewidth=0.8)
    for point in pending:
        i = int(point["index"])
        if start <= i <= end:
            ax.scatter(upto["Date"].iloc[i], close.iloc[i], marker="o", s=60, facecolor="none", edgecolor="#7a8594", linewidth=1.4, zorder=3)
    n_bottom = sum(1 for p in settled if start <= p["index"] <= end and p["type"] == "bottom")
    n_top = sum(1 for p in settled if start <= p["index"] <= end and p["type"] == "top")
    ax.scatter([], [], marker="^", color=BUY, label=f"CPM 저점 ({n_bottom}개)")
    ax.scatter([], [], marker="v", color=SELL, label=f"CPM 고점 ({n_top}개)")
    ax.scatter([], [], marker="o", facecolor="none", edgecolor="#7a8594", label="아직 확정 안 된 마지막 점")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.set_title(f"CPM 전환점 — P={p_value:.0%}, T={t_value} ({year - 1}년 이전 전체로 파레토 무릎 선택, cpm_causal과 같은 방식)", loc="left", fontsize=10.5)

    ax = axes[1]
    ax.plot(dates, price, color=INK, linewidth=1.0, alpha=0.6)
    labels = barrier[window]
    for value, colour, name in ((-1.0, BUY, "위 배리어 먼저 (라벨 −1, 매수)"), (1.0, SELL, "아래 배리어 먼저 (라벨 +1, 매도)"), (0.0, NEUTRAL, "10일 안에 안 닿음 (0)")):
        mask = labels == value
        ax.scatter(dates[mask], price[mask], s=14, color=colour, zorder=3, label=f"{name}: {int(mask.sum())}일")
    ax.legend(loc="upper left", ncol=3, fontsize=8.5)
    ax.set_title("트리플배리어 라벨 — 매일 하나씩 (ATR×2, 10일)", loc="left", fontsize=10.5)

    ax = axes[2]
    ax.plot(dates, triangle[window], color="#2b6ea8", linewidth=1.6, label="CPM 삼각파 목표")
    ax.step(dates, labels, where="mid", color="#c27c00", linewidth=1.0, alpha=0.9, label="트리플배리어 라벨")
    ax.axhline(0, color="#8a94a3", linewidth=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1], ["−1 (매수)", "0", "+1 (매도)"])
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("ESN이 맞히려는 목표값 (확정되지 않은 끝부분은 비워 둠)", loc="left", fontsize=10.5)
    agree = np.sign(triangle[window]) == np.sign(labels)
    usable = ~np.isnan(triangle[window]) & (labels != 0) & ~np.isnan(labels)
    fig.suptitle(f"{ticker} {year}년: CPM과 트리플배리어가 찍는 위치   (둘 다 0이 아닌 날 중 부호 일치 {agree[usable].mean():.0%})", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{ticker}_cpm_vs_tb_{year}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2020])
    args = parser.parse_args()
    for year in args.years:
        print(plot_year(args.ticker, year))


if __name__ == "__main__":
    main()
