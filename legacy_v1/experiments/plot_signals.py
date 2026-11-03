"""Trading signals of the CPM-label and triple-barrier-label models, and the triangle wave their trades draw.

    python -m legacy_v1.experiments.plot_signals [--tickers SPY QQQ AAPL GLD] [--start 2018-01-01] [--end 2020-12-31]

Source: the per-stock walk-forward runs of compare_labels.py (results/legacy_v1/labels/<ticker>_<label>/, no overlay, next-day
open fills). A trade triangle puts -1 on every buy and +1 on every sell and joins them with straight lines, the same way the
CPM target joins troughs (-1) and peaks (+1); it shows where the model's own 'turning points' fall. The reference is the CPM
triangle of data/labels/cpm (P, T chosen on prices up to 2018). The CPM model's score is its direct estimate of that triangle.
Outputs: results/legacy_v1/signal_summary.csv and results/legacy_v1/figures/<ticker>_signal_triangles_<start>_<end>.png
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

RUNS = Path("results/legacy_v1/labels")
FIGURES = Path("results/legacy_v1/figures")
LABELS = {"cpm_causal": ("CPM 라벨 모델", "#2b6ea8"), "triple_barrier": ("트리플배리어 라벨 모델", "#c27c00")}
BUY, SELL, INK, REF = "#1a9e5a", "#d1453b", "#222222", "#9aa3ae"
plt.rcParams.update({"font.family": ["Malgun Gothic", "sans-serif"], "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e7eb", "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 130})


def triangle(dates: pd.Series, points: list[tuple[pd.Timestamp, float]]) -> np.ndarray:
    """-1 / +1 at the given dates, straight lines in between, NaN outside the first and last point."""
    out = np.full(len(dates), np.nan)
    pos = [(int(np.searchsorted(dates.to_numpy(), np.datetime64(d))), v) for d, v in points]
    pos = [(p, v) for p, v in pos if p < len(dates)]
    for (a, va), (b, vb) in zip(pos, pos[1:]):
        if b > a:
            out[a:b + 1] = np.linspace(va, vb, b - a + 1)
    return out


def load(ticker: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = RUNS / f"{ticker}_{label}"
    predictions = pd.read_csv(folder / "walkforward_predictions.csv", parse_dates=["date"])
    trades = pd.read_csv(folder / "trades.csv", parse_dates=["date"])
    return predictions, trades


def summary(ticker: str, label: str, predictions: pd.DataFrame, trades: pd.DataFrame) -> dict:
    buys, sells = trades[trades["action"] == "buy"].reset_index(drop=True), trades[trades["action"] == "sell"].reset_index(drop=True)
    n = min(len(buys), len(sells))
    returns = sells["execution_price"].iloc[:n].to_numpy() / buys["execution_price"].iloc[:n].to_numpy() - 1
    holding = (sells["date"].iloc[:n].to_numpy() - buys["date"].iloc[:n].to_numpy()).astype("timedelta64[D]").astype(float)
    signal = predictions["signal"]
    return {"ticker": ticker, "label": label, "days": len(predictions), "buy_signal_days": float((signal == -1).mean()), "sell_signal_days": float((signal == 1).mean()),
            "hold_signal_days": float((signal == 0).mean()), "round_trips": int(n), "round_trips_per_year": n / (len(predictions) / 252),
            "median_holding_calendar_days": float(np.median(holding)) if n else np.nan, "trade_win_rate": float((returns > 0).mean()) if n else np.nan,
            "median_trade_return": float(np.median(returns)) if n else np.nan}


def plot(ticker: str, start: str, end: str, runs: dict) -> Path:
    marks = pd.read_csv(Path("data/labels/cpm") / f"{ticker}.csv", parse_dates=["Date"])
    final = marks[(marks["cpm_point_type"].fillna("") != "") & marks["final"]]
    reference_points = [(d, -1.0 if t == "bottom" else 1.0) for d, t in zip(final["Date"], final["cpm_point_type"])]
    base = runs["cpm_causal"][0]
    window = base[(base["date"] >= start) & (base["date"] <= end)].reset_index(drop=True)
    dates = window["date"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True, gridspec_kw={"height_ratios": [2.4, 2.4, 2.6]})
    texts = []
    reference = triangle(dates, reference_points)
    for ax, (label, (name, colour)) in zip(axes[:2], LABELS.items()):
        predictions, trades = runs[label]
        part = trades[(trades["date"] >= start) & (trades["date"] <= end)]
        ax.plot(dates, window["close"], color=INK, linewidth=1.1)
        for action, marker, c in (("buy", "^", BUY), ("sell", "v", SELL)):
            rows = part[part["action"] == action]
            ax.scatter(rows["date"], rows["execution_price"], marker=marker, s=60, color=c, edgecolor="white", linewidth=0.7, zorder=3,
                       label=f"{'매수' if action == 'buy' else '매도'} {len(rows)}회")
        held = predictions.set_index("date")["signal"]
        ax.legend(loc="upper left", ncol=2, fontsize=9)
        ax.set_title(f"{name}: 실제 체결 (다음 날 시가)", loc="left", fontsize=10.5)
        points = [(d, -1.0 if a == "buy" else 1.0) for d, a in zip(trades["date"], trades["action"])]
        tri = triangle(dates, points)
        axes[2].plot(dates, tri, color=colour, linewidth=1.6, label=f"{name}의 거래 삼각파")
        both = np.isfinite(tri) & np.isfinite(reference)
        texts.append(f"{name.split(' ')[0]} {np.corrcoef(tri[both], reference[both])[0, 1]:+.2f}" if both.sum() > 30 else f"{name.split(' ')[0]} –")
    score = window["trading_score"]
    axes[2].plot(dates, reference, color=REF, linewidth=3.2, alpha=0.75, label="실제 CPM 삼각파 (사후 정답)", zorder=1)
    axes[2].plot(dates, score, color=LABELS["cpm_causal"][1], linewidth=0.9, linestyle=":", label="CPM 모델 점수 (삼각파 예측값)")
    axes[2].axhline(0, color="#8a94a3", linewidth=0.8)
    axes[2].set_ylim(-1.15, 1.15)
    axes[2].set_yticks([-1, 0, 1], ["−1 (매수/저점)", "0", "+1 (매도/고점)"])
    axes[2].legend(loc="upper left", ncol=2, fontsize=8.5)
    axes[2].set_title("삼각파 비교 — 실제 정답과의 상관: " + ", ".join(texts), loc="left", fontsize=10.5)
    fig.suptitle(f"{ticker} {start[:4]}~{end[:4]}: 모델 거래 신호와 그 신호로 그린 삼각파", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{ticker}_signal_triangles_{start[:4]}_{end[:4]}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "GLD"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2020-12-31")
    args = parser.parse_args()
    rows = []
    for ticker in args.tickers:
        runs = {label: load(ticker, label) for label in LABELS}
        rows += [summary(ticker, label, *runs[label]) for label in LABELS]
        print(plot(ticker, args.start, args.end, runs))
    table = pd.DataFrame(rows)
    table.to_csv(Path("results/legacy_v1/signal_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
