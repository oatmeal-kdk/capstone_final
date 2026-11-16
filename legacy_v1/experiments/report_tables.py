"""Markdown result tables (Sharpe, annual return with the cumulative return, MDD, and the same per fold) of finished runs.
Added 2026-11-12.

    python -m legacy_v1.experiments.report_tables combos      # 3 inputs x 3 ESN setups (select_esn_backtest.py runs)
    python -m legacy_v1.experiments.report_tables reservoir   # compare_reservoir.py (reservoir x spectral radius)

Reads the saved daily returns only (nothing is rerun). Whole period = the stitched test years, as in the runs. "평균" = the
mean over the stocks. A fold = one test year (its start is the fold's test start in folds.csv); per fold, the return is
compounded inside the test year and the MDD is measured from the test year's start (the position carried in from the previous
year is kept, as in the stitched backtest). Saved as results/legacy_v1/tables_<preset>.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/legacy_v1")
STOCKS = ["SPY", "QQQ", "AAPL", "GLD"]
FOLD_SOURCE = R / "esn_backtest_s0" / "folds.csv"          # test start dates per stock and fold (the same folds everywhere)


def combos() -> list[tuple[str, str, Path]]:
    inputs = [("ta", "esn_backtest"), ("ta + 캔들", "esn_backtest_ta_candle"), ("ta_ga + 캔들", "esn_backtest_ga_ta_candle"),
              ("ta_ga v2 + 캔들 (규칙 v2)", "esn_backtest_ga2_ta_candle")]
    esn = [("기본값", "_s0", "default"), ("탐색", "_s0", "selected"), ("앙상블 10", "_ens10", "default")]
    return [(i, e, R / f"{run}{suffix}" / f"daily_{kind}.csv") for i, run in inputs for e, suffix, kind in esn]


def reservoir() -> list[tuple[str, str, Path]]:
    d = R / "reservoir"
    rows = [(f"reservoir {r}", f"sr {sr}", d / f"daily_r{r}_sr{sr}.csv") for r in (500, 1000, 3000) for sr in (0.1, 0.3, 0.5, 0.7, 0.9)]
    return rows + [("기본 ESN (100, 0.9)", "", d / "daily_default.csv")]


PRESETS = {"combos": combos, "reservoir": reservoir}

INPUT_NOTES = [   # combos only: what each input row is
    ("ta", 20, "ta 지표 20개, 기본 기간"),
    ("ta + 캔들", 25, "ta 20개 (기본 기간) + 캔들 패턴 5개 (기본 파라미터, -1/0/+1)"),
    ("ta_ga + 캔들", 25, "ta 20개 (규칙 v1의 GA 기간, data/params) + 캔들 패턴 5개 (GA 파라미터)"),
    ("ta_ga v2 + 캔들 (규칙 v2)", 24, "ta 23개 (규칙 v2의 GA 기간, data/params_v2: ROC 단기·장기, MA 간격, MA 이격 추가) + 통합 캔들 신호 1개"),
]
ESN_NOTES = [("기본값", "reservoir 100, spectral radius 0.9, input scale 0.3, leak 0.3, ridge 200, seed 0"),
             ("탐색", "폴드마다 243개 후보 중 선택 연도 4개 Sharpe 최고, seed 0"),
             ("앙상블 10", "기본값 설정, seed 0~9 점수 평균")]


def combos_summary(daily: dict, whole: dict) -> list[str]:
    """Inputs legend, the 4-stock means in one table, v2 against v1 per ESN setup, and the GA fitness of the rules."""
    from legacy_v1.experiments.judge import judge

    out = ["## 요약", "### 입력과 ESN", "| 입력 | 피처 수 | 내용 |", "| --- | --- | --- |"]
    out += [f"| {n} | {k} | {d} |" for n, k, d in INPUT_NOTES]
    out += ["", "| ESN | 내용 |", "| --- | --- |"] + [f"| {n} | {d} |" for n, d in ESN_NOTES]
    esns = [n for n, _ in ESN_NOTES]
    out += ["", "### 4종목 평균 (Sharpe / 연수익률 / MDD)", "| 입력 | " + " | ".join(esns) + " |", "| --- | " + "--- | " * len(esns)]
    mean = lambda k, col: np.mean([whole[k][t][col] for t in STOCKS])          # noqa: E731
    for n, _, _ in INPUT_NOTES:
        out.append(f"| {n} | " + " | ".join(f"{mean((n, e), 'sharpe'):.3f} / {mean((n, e), 'cagr') * 100:.1f}% / {mean((n, e), 'mdd') * 100:.1f}%"
                                            for e in esns) + " |")
    h = ("보유", "")
    out.append(f"| 보유 | {mean(h, 'sharpe'):.3f} / {mean(h, 'cagr') * 100:.1f}% / {mean(h, 'mdd') * 100:.1f}% (ESN 무관) | | |")
    v1, v2 = INPUT_NOTES[2][0], INPUT_NOTES[3][0]
    out += ["", "### 규칙 v2 vs v1 (ta_ga v2 + 캔들 − ta_ga + 캔들)",
            "종목 중앙값 Sharpe 차이, 95% 월 블록 부트스트랩 구간 (legacy_v1/experiments/judge.py). 구간이 0을 포함하면 구분 안 됨.", "",
            "| ESN | Sharpe 차이 | 95% 구간 | 판정 | v2가 앞선 종목 | 연수익률 차이 | MDD 차이 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in esns:
        j = judge(daily[(v2, e)][STOCKS], daily[(v1, e)][STOCKS])
        verdict = {"not distinguishable": "구분 안 됨"}.get(j["verdict"], j["verdict"])
        out.append(f"| {e} | {j['median_sharpe_diff']:+.3f} | {j['ci_lo']:+.3f} ~ {j['ci_hi']:+.3f} | {verdict} | "
                   f"{round(j['share_stocks_ahead'] * len(STOCKS))}/{len(STOCKS)} | {j['median_cagr_diff'] * 100:+.1f}%p | {j['median_mdd_diff'] * 100:+.1f}%p |")
    fit = {}
    for version, folder in (("v1", "data/params"), ("v2", "data/params_v2")):
        for t in STOCKS:
            path = Path(folder) / f"{t}.json"
            if path.exists():
                for entry in json.loads(path.read_text(encoding="utf-8"))["folds"].values():
                    for rule, x in entry["rules"].items():
                        fit.setdefault((version, rule), []).append(x["train_fitness"])
    if fit:
        med = {k: float(np.median(v)) for k, v in fit.items()}
        pairs = [("rsi", "rsi"), ("roc", "roc"), ("stoch", "stoch"), ("bollinger", "bollinger"), ("mfi", "mfi"), ("cmf", "cmf"),
                 ("trix", "trix"), ("aroon", "aroon"), ("keltner", "squeeze"), ("adx", "adx"), ("williams_r", "williams_r"),
                 ("cci", "cci"), ("stochrsi", "stochrsi"), ("ultimate", "ultimate"), (None, "ma_cross"), (None, "ma_envelope"),
                 ("candle_* (5개)", "candle")]
        out += ["", "### 규칙별 GA 학습 fitness (4종목 x 11폴드 중앙값, 낮을수록 좋음, 1 = 신호 없음과 같음)",
                "| 규칙 v1 | v1 | 규칙 v2 | v2 |", "| --- | --- | --- | --- |"]
        for a, b in pairs:
            if a == "candle_* (5개)":
                v1s = " / ".join(f"{med[('v1', r)]:.3f}" for r in ("candle_hammer", "candle_dark_cloud", "candle_piercing",
                                                                  "candle_bullish_engulfing", "candle_bearish_engulfing") if ("v1", r) in med)
            else:
                v1s = f"{med[('v1', a)]:.3f}" if a and ("v1", a) in med else "—"
            v2s = f"{med[('v2', b)]:.3f}" if ("v2", b) in med else "—"
            out.append(f"| {a or '—'} | {v1s} | {b} | {v2s} |")
        out.append(f"| **전체** | **{np.median([x for (v, _), xs in fit.items() if v == 'v1' for x in xs]):.3f}** | | "
                   f"**{np.median([x for (v, _), xs in fit.items() if v == 'v2' for x in xs]):.3f}** |")
    return ["\n".join(out)]


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else float("nan")


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    equity = (1 + r).cumprod()
    return {"sharpe": sharpe(r), "cagr": float(equity.iloc[-1] ** (252 / len(r)) - 1), "total": float(equity.iloc[-1] - 1),
            "mdd": float((equity / equity.cummax() - 1).min())}


def fold_bounds(ticker: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp | None]]:
    starts = pd.read_csv(FOLD_SOURCE).query("ticker == @ticker")["test_year"].map(pd.Timestamp).tolist()
    return [(f"{s.year}", s, starts[k + 1] if k + 1 < len(starts) else None) for k, s in enumerate(starts)]


def table(rows: list[tuple[str, str, dict]], fmt, columns: list[str], first: tuple[str, str], bold_mean: str) -> str:
    """rows: (group, name, {column: value}), holding last; the group is written only on its first row; the best mean of the
    strategies (holding left out) in bold. NaN (a year spent in cash: no return variation) is shown as —."""
    means = [np.nanmean([v[c] for c in columns]) for _, _, v in rows]
    best = int((np.nanargmax if bold_mean == "max" else np.nanargmin)(means[:-1]))
    out = [f"| {first[0]} | {first[1]} | " + " | ".join(columns) + " | **평균** |", "| --- | --- | " + "--- | " * (len(columns) + 1)]
    last = None
    for k, ((group, name, values), mean) in enumerate(zip(rows, means)):
        cells = ["—" if np.isnan(values[c]) else fmt(values[c]) for c in columns]
        m = fmt(mean, mean_only=True)
        out.append(f"| {group if group != last else ''} | {name} | " + " | ".join(cells) + f" | {f'**{m}**' if k == best else m} |")
        last = group
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=list(PRESETS))
    args = parser.parse_args()
    runs = PRESETS[args.preset]()
    hold_path = R / ("reservoir" if args.preset == "reservoir" else "esn_backtest_s0") / "daily_hold.csv"
    daily = {(g, n): pd.read_csv(p, index_col="date", parse_dates=True) for g, n, p in runs}
    daily[("보유", "")] = pd.read_csv(hold_path, index_col="date", parse_dates=True)
    keys = list(daily)

    whole = {k: {t: metrics(daily[k][t]) for t in STOCKS} for k in keys}
    pct = lambda v, mean_only=False: f"{v * 100:.1f}%"                       # noqa: E731
    num = lambda v, mean_only=False: f"{v:.3f}"                              # noqa: E731
    first = ("입력", "ESN") if args.preset == "combos" else ("reservoir", "spectral radius")
    parts = [f"전체 기간 = 테스트 구간을 이어 붙인 것 ({min(d.index.min() for d in daily.values()).date()} ~ "
             f"{max(d.index.max() for d in daily.values()).date()}). 평균 = 4종목 평균. 굵게 = 전략 중 최고 (보유 제외).",
             *(combos_summary(daily, whole) if args.preset == "combos" else []),
             "## Sharpe", table([(g, n, {t: whole[(g, n)][t]["sharpe"] for t in STOCKS}) for g, n in keys], num, STOCKS, first, "max")]
    ret_rows = []
    for g, n in keys:
        cells = {t: whole[(g, n)][t] for t in STOCKS}
        ret_rows.append((g, n, {t: (cells[t]["cagr"], cells[t]["total"]) for t in STOCKS}))
    def ret_fmt(v, mean_only=False):
        return f"{v * 100:.1f}%" if mean_only else f"{v[0] * 100:.1f}% ({v[1] * 100:.0f}%)"
    # cells carry (annual, cumulative) but the mean is of the annual returns only, so this table is written here
    lines = [f"| {first[0]} | {first[1]} | " + " | ".join(STOCKS) + " | **평균** |", "| --- | --- | " + "--- | " * (len(STOCKS) + 1)]
    means = [np.mean([v[t][0] for t in STOCKS]) for _, _, v in ret_rows]
    best, last = int(np.argmax(means[:-1])), None
    for k, ((g, n, v), m) in enumerate(zip(ret_rows, means)):
        mean = f"{m * 100:.1f}%"
        lines.append(f"| {g if g != last else ''} | {n} | " + " | ".join(ret_fmt(v[t]) for t in STOCKS) + f" | {f'**{mean}**' if k == best else mean} |")
        last = g
    parts += ["## 연수익률 (괄호는 전체 기간 누적 수익률)", "\n".join(lines)]
    parts += ["## MDD (0에 가까울수록 덜 빠짐)", table([(g, n, {t: whole[(g, n)][t]["mdd"] for t in STOCKS}) for g, n in keys], pct, STOCKS, first, "max")]

    folds = {t: fold_bounds(t) for t in STOCKS}
    years = [y for y, _, _ in folds[STOCKS[0]]]
    for title, col, fmt, best in (("Sharpe", "sharpe", lambda v, mean_only=False: f"{v:.2f}", "max"),
                                  ("수익률", "total", pct, "max"), ("MDD", "mdd", pct, "max")):
        parts.append(f"## 폴드별 {title}")
        parts.append(f"열 = 테스트 연도 (그해 3월 초 ~ 다음 해 3월 초, 마지막 {years[-1]}은 데이터 끝까지). "
                     + ("— = 그해 내내 현금 (Sharpe 정의 안 됨), 평균에서 제외." if col == "sharpe" else ""))
        for t in STOCKS:
            rows = []
            for g, n in keys:
                r = daily[(g, n)][t].dropna()
                rows.append((g, n, {y: metrics(r[(r.index >= s) & ((r.index < e) if e is not None else True)])[col] for y, s, e in folds[t]}))
            parts += [f"### {t}", table(rows, fmt, years, first, best)]
    text = "\n\n".join(parts) + "\n"
    out = R / f"tables_{args.preset}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
