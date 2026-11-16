"""9 x 9 max-drawdown comparison of the select_esn_backtest.py runs: 3 inputs x 3 ESN setups, 4 stocks. Added 2026-11-09.

    python -m legacy_v1.experiments.plot_mdd_matrix [--out results/legacy_v1/figures/esn_mdd_matrix.png]

The 9 strategies (rows and columns, the same order):
  inputs  ta (esn_backtest_*), ta_candle (esn_backtest_ta_candle_*), ga_ta_candle (esn_backtest_ga_ta_candle_*)
  ESN     esn       the default ESN, seed 0                 (<run>_s0/daily_default.csv)
          esn탐색   the grid-selected ESN, seed 0           (<run>_s0/daily_selected.csv)
          esn앙상블 the default ESN, 10-seed ensemble       (<run>_ens10/daily_default.csv)
MDD of each stock over the whole test period (2016-03 ~ the end, next-open fills, costs as in the runs).
Cell (row i, column j), i != j: median over the 4 stocks of MDD_i - MDD_j (MDD is negative, so > 0 = row i falls less).
Diagonal: the median MDD of strategy i itself. Holding's median MDD is in the title. Also saved as a CSV next to the figure,
with the per-stock MDD of every strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results/legacy_v1")
INPUTS = {"ta": "esn_backtest", "ta_candle": "esn_backtest_ta_candle", "ga_ta_candle": "esn_backtest_ga_ta_candle"}
ESN = {"esn": ("_s0", "default"), "esn탐색": ("_s0", "selected"), "esn앙상블": ("_ens10", "default")}
STOCKS = ["SPY", "QQQ", "AAPL", "GLD"]


def mdd(returns: pd.DataFrame) -> pd.Series:
    equity = (1 + returns.fillna(0.0)).cumprod()
    return (equity / equity.cummax() - 1).min()


def load() -> tuple[pd.DataFrame, pd.Series]:
    """(strategies x stocks MDD, holding MDD per stock)."""
    rows = {}
    for inp, run in INPUTS.items():
        for esn, (suffix, kind) in ESN.items():
            daily = pd.read_csv(RESULTS / f"{run}{suffix}" / f"daily_{kind}.csv", index_col="date", parse_dates=True)
            rows[f"{inp} / {esn}"] = mdd(daily[STOCKS])
    hold = pd.read_csv(RESULTS / "esn_backtest_s0" / "daily_hold.csv", index_col="date", parse_dates=True)
    return pd.DataFrame(rows).T, mdd(hold[STOCKS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS / "figures" / "esn_mdd_matrix.png"))
    args = parser.parse_args()
    table, hold = load()
    names = list(table.index)
    values = table.to_numpy()
    n = len(names)
    diff = np.array([[np.median(values[i] - values[j]) for j in range(n)] for i in range(n)])
    own = np.median(values, axis=1)

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 9.5))
    off = np.where(np.eye(n, dtype=bool), np.nan, diff * 100)
    lim = np.nanmax(np.abs(off))
    im = ax.imshow(off, cmap="RdYlGn", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#e6e6e6"))
                ax.text(j, i, f"{own[i] * 100:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, f"{diff[i, j] * 100:+.1f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    for k in (2.5, 5.5):                                   # separate the three inputs
        ax.axhline(k, color="black", lw=1.2)
        ax.axvline(k, color="black", lw=1.2)
    ax.set_title(f"MDD 비교 (SPY·QQQ·AAPL·GLD, 테스트 2016-03~)\n"
                 f"칸: 4종목 중앙값 MDD(행) − MDD(열), %p (+ = 행이 덜 빠짐) / 대각: 그 전략의 MDD 중앙값 / 보유 {np.median(hold) * 100:.1f}%")
    fig.colorbar(im, ax=ax, fraction=0.04, label="MDD 차이 중앙값 (%p)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)

    report = table.copy()
    report.loc["hold"] = hold
    report["median"] = report[STOCKS].median(axis=1)
    report.to_csv(out.with_suffix(".csv"))
    print((report * 100).round(1).to_string())
    print(f"saved {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
