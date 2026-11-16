"""Where do the two labels put their marks? CPM turning points and triple-barrier labels on the same price chart.

    python -m legacy_v1.experiments.plot_labels [--ticker SPY] [--years 2018 2020]

For each year: CPM's P and T are chosen by the Pareto knee on every price before the year that precedes it (cpm_causal trains
on all earlier rows and keeps the last year for validation), and
CPM is run on prices up to the end of the year; its last reported point is not final and is not drawn as a label. The triple
barrier (ATR x 2, 10 days) is computed on the same prices. Bottom panel: the two training targets the ESN would regress on
(sign convention of the code: -1 = buy, +1 = sell); --no-targets leaves it out. Output: results/legacy_v1/figures/
<ticker>_cpm_vs_tb_<year>[_marks].png (--out-dir to change the folder)
--triangle-only (added 2026-11-09): one panel with the CPM triangle target alone, <ticker>_cpm_triangle_<year>.png
--price-to-triangle (added 2026-11-09): price with CPM points -> arrow -> triangle target, <ticker>_cpm_price_to_triangle_<year>.png
Both use a trading-day x axis, so the triangle's lines are straight (a date axis bends them at weekends and holidays).
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


def plot_year(ticker: str, year: int, targets: bool = True, out: Path = OUT) -> Path:
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
    if targets:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2]})
    else:                                                          # --no-targets: only the two price panels
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)

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

    if targets:
        _plot_targets(axes[2], dates, triangle[window], labels)
    agree = np.sign(triangle[window]) == np.sign(labels)
    usable = ~np.isnan(triangle[window]) & (labels != 0) & ~np.isnan(labels)
    fig.suptitle(f"{ticker} {year}년: CPM과 트리플배리어가 찍는 위치   (둘 다 0이 아닌 날 중 부호 일치 {agree[usable].mean():.0%})", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ticker}_cpm_vs_tb_{year}{'' if targets else '_marks'}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _triangle_year(ticker: str, year: int) -> dict:
    """Prices, settled CPM points and the triangle target of one year (cpm_causal P, T as in plot_year), on a trading-day axis."""
    data = pd.read_csv(f"data/processed/{ticker}/1d/{ticker}_with_tp.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    end = data.index[data["Date"] <= f"{year}-12-31"][-1]
    start = data.index[data["Date"] >= f"{year}-01-01"][0]
    train_end = data.index[data["Date"] >= f"{year - 1}-01-01"][0]
    upto = data.iloc[: end + 1].reset_index(drop=True)
    close = upto["Close"].astype(float)
    p_value, t_value = _pareto_knee_params(close.iloc[:train_end])
    points = [p for p in detect_turning_points(close, P=p_value, T=t_value)[:-1] if start <= int(p["index"]) <= end]
    return {"x": np.arange(end + 1 - start), "dates": upto["Date"].iloc[start:end + 1].reset_index(drop=True),
            "price": close.iloc[start:end + 1].to_numpy(), "triangle": _known_triangle(close, p_value, t_value)[start:end + 1],
            "points": [(int(p["index"]) - start, p["type"] == "bottom") for p in points], "P": p_value, "T": t_value}


def _month_ticks(ax, dates: pd.Series, every: int = 1) -> None:
    """Trading-day x axis labelled with the first trading day of each month (a date axis would bend the straight lines at weekends)."""
    first = np.flatnonzero(dates.dt.month.diff().fillna(1).to_numpy() != 0)[::every]
    ax.set_xticks(first, [d.strftime("%Y-%m") for d in dates.iloc[first]])
    ax.set_xlim(-3, len(dates) + 2)


def _draw_triangle(ax, d: dict) -> None:
    ax.plot(d["x"], d["triangle"], color="#2b6ea8", linewidth=1.8)
    for i, bottom in d["points"]:
        ax.scatter(i, -1.0 if bottom else 1.0, marker="^" if bottom else "v", s=60, color=BUY if bottom else SELL, zorder=3,
                   edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#8a94a3", linewidth=0.8)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, 0, 1], ["−1 (매수)", "0", "+1 (매도)"])


def plot_triangle(ticker: str, year: int, out: Path = OUT) -> Path:
    """The CPM triangle target of one year alone, turning points marked at +-1."""
    d = _triangle_year(ticker, year)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    _draw_triangle(ax, d)
    _month_ticks(ax, d["dates"])
    ax.scatter([], [], marker="^", color=BUY, label="CPM 저점 (−1, 매수)")
    ax.scatter([], [], marker="v", color=SELL, label="CPM 고점 (+1, 매도)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=9)
    fig.suptitle(f"{ticker} {year}년 CPM 삼각파 목표 — 전환점 사이를 직선으로 이음 (P={d['P']:.0%}, T={d['T']}, 확정되지 않은 끝부분은 비워 둠)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ticker}_cpm_triangle_{year}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_price_to_triangle(ticker: str, year: int, out: Path = OUT) -> Path:
    """Left: the price with its CPM turning points; arrow; right: the triangle target the ESN is trained on."""
    d = _triangle_year(ticker, year)
    fig = plt.figure(figsize=(14, 4.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[10, 1.6, 10], wspace=0.05)
    left, mid, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[2])

    left.plot(d["x"], d["price"], color=INK, linewidth=1.2)
    for i, bottom in d["points"]:
        left.scatter(i, d["price"][i], marker="^" if bottom else "v", s=60, color=BUY if bottom else SELL, zorder=3,
                     edgecolor="white", linewidth=0.8)
    left.scatter([], [], marker="^", color=BUY, label="저점")
    left.scatter([], [], marker="v", color=SELL, label="고점")
    left.legend(loc="lower right", ncol=2, fontsize=9)
    left.set_title(f"① 종가와 CPM 전환점 (P={d['P']:.0%}, T={d['T']})", loc="left", fontsize=11)
    _month_ticks(left, d["dates"], every=2)

    mid.axis("off")
    mid.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                 arrowprops={"arrowstyle": "-|>,head_width=0.5,head_length=0.9", "color": "#555c66", "linewidth": 2.2})

    _draw_triangle(right, d)
    right.set_title("② ESN 학습 목표 (삼각파)", loc="left", fontsize=11)
    _month_ticks(right, d["dates"], every=2)

    fig.suptitle(f"{ticker} {year}년: CPM 전환점을 삼각파 목표로 바꾸기", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.82, bottom=0.12)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ticker}_cpm_price_to_triangle_{year}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_targets(ax, dates, triangle, labels) -> None:
    ax.plot(dates, triangle, color="#2b6ea8", linewidth=1.6, label="CPM 삼각파 목표")
    ax.step(dates, labels, where="mid", color="#c27c00", linewidth=1.0, alpha=0.9, label="트리플배리어 라벨")
    ax.axhline(0, color="#8a94a3", linewidth=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1], ["−1 (매수)", "0", "+1 (매도)"])
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("ESN이 맞히려는 목표값 (확정되지 않은 끝부분은 비워 둠)", loc="left", fontsize=10.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2020])
    parser.add_argument("--no-targets", action="store_true", help="leave out the bottom panel (the ESN targets); file name gets _marks")
    parser.add_argument("--triangle-only", action="store_true", help="one panel with the CPM triangle target alone")
    parser.add_argument("--price-to-triangle", action="store_true", help="price with CPM points -> arrow -> triangle target, side by side")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for year in args.years:
        if args.price_to_triangle:
            print(plot_price_to_triangle(args.ticker, year, out=args.out_dir))
        elif args.triangle_only:
            print(plot_triangle(args.ticker, year, out=args.out_dir))
        else:
            print(plot_year(args.ticker, year, targets=not args.no_targets, out=args.out_dir))


if __name__ == "__main__":
    main()
