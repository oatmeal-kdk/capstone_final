"""Triple barrier vs CPM (Pareto-knee P, T) in the v1 recipe, against holding the stock, with and without the risk overlay.

    python -m legacy_v1.experiments.compare_labels [--per-market 30] [--jobs 6]

Labels compared, everything else fixed (ta 20 indicators, ESN, threshold on validation accuracy, 5y/1y/1y walk-forward,
costs 0.05% + 0.05%, fills at the next day's open):
  triple_barrier  the locked Phase 12 label
  cpm_causal      the old CPM triangle target, built per fold from data the fold could see: P, T chosen by the Pareto knee on
                  the training rows only. (Choosing P, T on the whole period is not allowed, so cpm / cpm_global are not run.)
Baselines on the same dates: hold the stock, and hold it under the same volatility-target overlay. The overlay baseline answers
whether the overlay's gain needs the ESN signal at all.
Stocks: SPY, QQQ, AAPL, GLD (the Phase 9-12 tickers) and a random sample (fixed seed) of the old screening universe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from legacy_v1.backtest.simple_backtest import run_long_only_backtest_vol_target
from legacy_v1.experiments.compare_execution import CORE, sample
from legacy_v1.pipeline import production_config as cfg

OUT = Path("results/legacy_v1/labels")
TARGETS = ("triple_barrier", "cpm_causal")      # whole-series CPM variants are excluded: their P, T would be chosen with future prices


def stats(returns: pd.Series) -> dict:
    years = len(returns) / 252
    equity = (1 + returns).cumprod()
    return {"sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan,
            "cagr": float(equity.iloc[-1] ** (1 / years) - 1), "mdd": float((equity / equity.cummax() - 1).min())}


def run(symbol: str, target: str, overlay: bool) -> dict:
    out = OUT / f"{symbol}_{target}{'_overlay' if overlay else ''}"
    cmd = [sys.executable, "-W", "ignore", "-m", "legacy_v1.pipeline.run_walkforward", "--tp-path", f"data/processed/{symbol}/1d/{symbol}_with_tp.csv",
           "--target-method", target, "--train-years", str(cfg.TRAIN_YEARS), "--fee-rate", str(cfg.FEE_RATE), "--slippage-rate", str(cfg.SLIPPAGE_RATE),
           "--execution", "next_open", "--output-dir", str(out)]
    if overlay:
        cmd += ["--risk-overlay", "--target-vol", str(cfg.RISK_OVERLAY_TARGET_VOL), "--rebalance-band", str(cfg.RISK_OVERLAY_REBALANCE_BAND)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not (out / "equity_curve.csv").exists():
        return {"symbol": symbol, "label": target, "overlay": overlay, "error": result.stderr.strip().splitlines()[-1][:200] if result.stderr else "no output"}
    equity = pd.read_csv(out / "equity_curve.csv")
    row = {"symbol": symbol, "label": target, "overlay": overlay, "start": equity["date"].iloc[0], "end": equity["date"].iloc[-1],
           "exposure": float((equity["position"] > 0).mean()), **stats(equity["portfolio_value"].pct_change().fillna(0.0))}
    if overlay and target == "triple_barrier":                    # baselines on the same dates and volatility series
        predictions = pd.read_csv(out / "walkforward_predictions.csv")
        hold = stats(predictions["close"].pct_change().fillna(0.0))
        always = predictions.assign(signal=-1)
        curve, _, _ = run_long_only_backtest_vol_target(always, price_col="close", signal_col="signal", vol_col="vol", target_vol=cfg.RISK_OVERLAY_TARGET_VOL,
                                                        rebalance_band=cfg.RISK_OVERLAY_REBALANCE_BAND, initial_cash=1.0, fee_rate=cfg.FEE_RATE,
                                                        slippage_rate=cfg.SLIPPAGE_RATE, execution="next_open")
        overlaid = stats(curve["portfolio_value"].pct_change().fillna(0.0))
        row.update({f"hold_{k}": v for k, v in hold.items()})
        row.update({f"hold_overlay_{k}": v for k, v in overlaid.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-market", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picked = sample(args.per_market)
    groups = {s: "core" for s in CORE} | {s: m for m, symbols in picked.items() for s in symbols if s not in CORE}
    jobs = [(s, t, o) for s in groups for t in TARGETS for o in (False, True)]
    rows = Parallel(n_jobs=args.jobs)(delayed(run)(s, t, o) for s, t, o in jobs)
    table = pd.DataFrame(rows)
    table["group"] = table["symbol"].map(groups)
    table.to_csv(OUT.parent / "label_compare.csv", index=False)

    ok = table[table["error"].isna()] if "error" in table else table
    base = ok[(ok["label"] == "triple_barrier") & ok["overlay"]].set_index("symbol")[[c for c in ok.columns if c.startswith("hold")]]
    pd.set_option("display.width", 220)
    for group in ("core", "US", "KR"):
        part = ok[ok["group"] == group]
        if part.empty:
            continue
        wide = part.pivot_table(index="symbol", columns=["overlay", "label"], values="sharpe")
        print(f"\n== {group}: Sharpe by stock (next-day open fills) ==" if group == "core" else f"\n== {group}: {len(wide)} random stocks, median ==")
        cells = {}
        for overlay in (False, True):
            for label in TARGETS:
                if (overlay, label) in wide:
                    cells[f"{label}{' + overlay' if overlay else ''}"] = wide[(overlay, label)]
        cells["hold"] = base["hold_sharpe"].reindex(wide.index)
        cells["hold + overlay"] = base["hold_overlay_sharpe"].reindex(wide.index)
        frame = pd.DataFrame(cells)
        print((frame if group == "core" else frame.median().to_frame("median Sharpe").T).round(3).to_string())
        if group != "core":
            print("share of stocks where the label beats hold + overlay (both with the overlay): "
                  + ", ".join(f"{label} {(frame[f'{label} + overlay'] > frame['hold + overlay']).mean():.0%}" for label in TARGETS if f"{label} + overlay" in frame))
    errors = table[table["error"].notna()] if "error" in table else table.iloc[0:0]
    print(f"\n{len(errors)} runs failed" + (f": {errors[['symbol', 'label', 'error']].head(8).to_string(index=False)}" if len(errors) else ""))


if __name__ == "__main__":
    main()
