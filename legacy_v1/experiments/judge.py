"""The declared win rule for comparing a strategy with a baseline over many stocks (fixed 2026-11-01, before the pooled results).

Primary verdict: for every stock, d = Sharpe(strategy) - Sharpe(baseline) over the whole test period; the statistic is the
MEDIAN of d across stocks. Its 95% interval comes from a calendar-month block bootstrap: whole months are drawn with
replacement and every stock is recomputed on the same drawn months, so stocks that move together are not counted as
independent evidence. Verdict: interval above 0 = "wins", below 0 = "loses", otherwise "not distinguishable".
Shown next to it, never used for the verdict: the share of stocks with d > 0, the median CAGR and max-drawdown differences,
and in how many test years the median yearly Sharpe difference is above 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _sharpe_from_sums(total: np.ndarray, squares: np.ndarray, count: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        var = squares / count - mean ** 2
        return np.where((count > 20) & (var > 0), mean / np.sqrt(var) * np.sqrt(TRADING_DAYS), np.nan)


def _month_sums(returns: pd.DataFrame, months: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stocks x months) sum, sum of squares and count of the daily returns."""
    values = returns.to_numpy(float)
    ok = np.isfinite(values)
    filled = np.where(ok, values, 0.0)
    shape = (values.shape[1], len(months))
    total, squares, count = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for m in range(len(months)):
        rows = codes == m
        total[:, m] = filled[rows].sum(axis=0)
        squares[:, m] = (filled[rows] ** 2).sum(axis=0)
        count[:, m] = ok[rows].sum(axis=0)
    return total, squares, count


def _cagr_mdd(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    equity = (1 + returns.fillna(0.0)).cumprod()
    years = returns.notna().sum() / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years.replace(0, np.nan)) - 1
    mdd = (equity / equity.cummax() - 1).min()
    return cagr, mdd


def judge(strategy: pd.DataFrame, baseline: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """strategy, baseline: daily returns, dates x stocks, same shape (NaN where a stock is not traded)."""
    strategy, baseline = strategy.align(baseline, join="inner")
    keep = strategy.notna().sum() > TRADING_DAYS
    strategy, baseline = strategy.loc[:, keep], baseline.loc[:, keep]
    period = strategy.index.to_period("M")
    months, codes = np.unique(period.astype(str), return_inverse=True)
    sums_s, sums_b = _month_sums(strategy, months, codes), _month_sums(baseline, months, codes)
    diff = _sharpe_from_sums(*(x.sum(axis=1) for x in sums_s)) - _sharpe_from_sums(*(x.sum(axis=1) for x in sums_b))
    rng = np.random.default_rng(seed)
    draws = np.stack([np.bincount(rng.integers(0, len(months), len(months)), minlength=len(months)) for _ in range(n_boot)]).astype(float)
    boot_s = _sharpe_from_sums(*(draws @ x.T for x in sums_s))           # (draws x stocks)
    boot_b = _sharpe_from_sums(*(draws @ x.T for x in sums_b))
    medians = np.nanmedian(boot_s - boot_b, axis=1)
    lo, hi = np.nanpercentile(medians, [2.5, 97.5])
    cagr_s, mdd_s = _cagr_mdd(strategy)
    cagr_b, mdd_b = _cagr_mdd(baseline)
    year = strategy.index.year
    yearly = []
    for y in np.unique(year):
        rows = year == y
        if rows.sum() < 60:
            continue
        ys, yb = strategy[rows], baseline[rows]
        s = ys.mean() / ys.std() * np.sqrt(TRADING_DAYS)
        b = yb.mean() / yb.std() * np.sqrt(TRADING_DAYS)
        yearly.append(float(np.nanmedian((s - b).to_numpy())))
    verdict = "wins" if lo > 0 else "loses" if hi < 0 else "not distinguishable"
    return {"stocks": int(len(diff)), "median_sharpe_diff": float(np.nanmedian(diff)), "ci_lo": float(lo), "ci_hi": float(hi), "verdict": verdict,
            "share_stocks_ahead": float(np.nanmean(diff > 0)), "median_cagr_diff": float(np.nanmedian((cagr_s - cagr_b).to_numpy())),
            "median_mdd_diff": float(np.nanmedian((mdd_s - mdd_b).to_numpy())), "years_ahead": int(sum(v > 0 for v in yearly)), "years": len(yearly)}
