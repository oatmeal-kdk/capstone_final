# legacy_v1 copy of src/pipeline/run_esn_ta_walkforward.py (frozen v1 recipe). Only import paths were changed.
"""Walk-forward validation for the 20-raw-ta-indicator ESN input set.

Mirrors config D from run_esn_year_splits.py (8y train / 1y val / 1y test,
expanding window, retrains every year through the end of the data), but
uses the 20 raw `ta` indicators (src/indicators/ta_features.py) scaled with
MinMaxUnitScaler instead of the 9 GA-tuned signals scaled with
StandardScaler.

The `ta` indicators are plain causal rolling-window formulas (not fit/tuned
against future data the way GA indicator parameters were), so unlike
run_esn_full_walkforward.py this does not need to recompute anything
data-dependent per fold except the scaler, the ESN, and the threshold
selection.

The training/eval target is pluggable via --target-method:
- "cpm" (default): triangle_target built from CPM turning points, the
  single global run (same known lookahead caveat as the rest of this
  project's walk-forward work).
- "cpm_global" (added 2026-11-01): the same triangle target, but CPM's P and T are chosen here by the Pareto knee of the
  P x T grid (as prepare_ticker_data.py / run_cpm.py did) and CPM is run on the whole series, so any ticker works. Like "cpm",
  every label sees the whole history: a look-ahead reference, not a tradable result.
- "cpm_causal" (added 2026-11-01): per fold, P and T are chosen by the Pareto knee on the training rows only; training targets
  come from a CPM run on the training rows, validation targets from a CPM run ending at the validation end. The last turning
  point CPM reports is not final, so it and every row after it are dropped instead of being extrapolated.
- "triple_barrier": ATR-scaled triple-barrier label (src/cpm/triple_barrier.py).
  Every label only depends on at most --tb-max-holding future rows. In
  walk-forward mode labels are rebuilt per fold and purged at train/validation
  boundaries so a training label never looks into validation/test rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from legacy_v1.backtest.simple_backtest import run_long_only_backtest, run_long_only_backtest_vol_target
from legacy_v1.labels.triple_barrier import label_triple_barrier
from legacy_v1.labels.cpm import detect_turning_points, grid_search_cpm, select_best_cpm_params
from legacy_v1.labels.cpm_target import build_triangle_target
from legacy_v1.model.metrics import score_to_signal
from legacy_v1.model.esn import EchoStateNetwork
from legacy_v1.features.ohlcv_features import build_ohlcv_features
from legacy_v1.features.ta_features import TA_FEATURE_COLUMNS, build_ta_features
from legacy_v1.model.thresholds import _select_thresholds
from legacy_v1.utils.io import save_dataframe, save_json
from legacy_v1.model.scaling import MinMaxUnitScaler

ESN_CONFIG = {
    "reservoir_size": 100,
    "spectral_radius": 0.9,
    "sparsity": 0.15,
    "input_scale": 0.3,
    "leaking_rate": 0.3,
    "ridge_alpha": 200.0,
    "washout": 30,
    "random_state": 42,
    "threshold_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


class EnsembleESN:
    """Averages predictions across several ESNs that differ only in random reservoir seed.

    A single fixed random reservoir can be "lucky" or "unlucky" for a given
    fold (see model.py's independent Win/W seed streams) -- averaging
    predictions across an ensemble of otherwise-identical models reduces
    that seed-luck variance, the same way bagging does. With ensemble_size=1
    this is numerically identical to using a single EchoStateNetwork.
    """

    def __init__(self, models: list[EchoStateNetwork]) -> None:
        self.models = models

    def predict(self, X: object) -> np.ndarray:
        preds = [np.asarray(m.predict(X), dtype=float) for m in self.models]
        return np.mean(preds, axis=0)


def _fit_esn(
    X_train: np.ndarray, y_train: np.ndarray, washout: int,
    random_state: int | None = None, ensemble_size: int = 1, readout_inputs: bool = True,
) -> EnsembleESN:
    base_seed = int(ESN_CONFIG["random_state"]) if random_state is None else int(random_state)
    models = []
    for i in range(max(1, ensemble_size)):
        model = EchoStateNetwork(
            input_dim=X_train.shape[1],
            reservoir_size=int(ESN_CONFIG["reservoir_size"]),
            spectral_radius=float(ESN_CONFIG["spectral_radius"]),
            sparsity=float(ESN_CONFIG["sparsity"]),
            input_scale=float(ESN_CONFIG["input_scale"]),
            leaking_rate=float(ESN_CONFIG["leaking_rate"]),
            ridge_alpha=float(ESN_CONFIG["ridge_alpha"]),
            washout=washout,
            random_state=base_seed + i,
            readout_inputs=readout_inputs,
        )
        model.fit(X_train, y_train)
        models.append(model)
    return EnsembleESN(models)


def _turning_points_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    points = []
    for idx, point_type in df["cpm_point_type"].fillna("").items():
        if str(point_type).strip():
            points.append({"index": idx, "type": str(point_type)})
    return points


CPM_GRID = {"p_values": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12], "t_values": [2, 4, 6, 8, 10, 12, 14, 16, 18],
            "selection_method": "pareto_knee"}       # the grid and selection of prepare_ticker_data.py / configs/spy_1d.yaml


def _pareto_knee_params(close: pd.Series) -> tuple[float, int]:
    best = select_best_cpm_params(grid_search_cpm(close.reset_index(drop=True), CPM_GRID["p_values"], CPM_GRID["t_values"]), CPM_GRID["selection_method"])
    return float(best["P"]), int(best["T"])


def _known_triangle(close: pd.Series, p_value: float, t_value: int) -> np.ndarray:
    """Triangle target of a CPM run on `close`, NaN where it is not determined: before the first turning point, and from the
    last reported point on (that point can still move, and nothing after it is known)."""
    points = detect_turning_points(close.reset_index(drop=True), P=p_value, T=t_value)[:-1]
    out = np.full(len(close), np.nan)
    if len(points) < 2:
        return out
    triangle = build_triangle_target(pd.DataFrame(index=pd.RangeIndex(len(close))), points).to_numpy(dtype=float)
    first, last = int(points[0]["index"]), int(points[-1]["index"])
    out[first:last + 1] = triangle[first:last + 1]
    return out


def _full_horizon_mask(length: int, max_holding: int) -> np.ndarray:
    """Rows whose full triple-barrier horizon stays inside the segment."""

    mask = np.zeros(length, dtype=bool)
    valid_until = length - int(max_holding)
    if valid_until > 0:
        mask[:valid_until] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-path", default="data/processed/SPY/1d/SPY_with_tp.csv")
    parser.add_argument("--output-dir", default="results/esn_ta_walkforward")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--val-years", type=int, default=1)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--target-method", default="cpm", choices=["cpm", "cpm_global", "cpm_causal", "triple_barrier"])
    parser.add_argument("--tb-atr-mult", type=float, default=2.0)
    parser.add_argument("--tb-max-holding", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=None, help="Override ESN_CONFIG's fixed random_state (for seed-sensitivity checks).")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Average predictions across this many reservoir seeds (reduces seed variance).")
    parser.add_argument(
        "--risk-overlay", action="store_true",
        help="Apply volatility-targeted fractional position sizing to the final aggregated result (see run_esn_ta_walkforward_auto.py).",
    )
    parser.add_argument("--target-vol", type=float, default=0.10, help="Annualized realized-vol target for --risk-overlay.")
    parser.add_argument("--rebalance-band", type=float, default=0.1, help="Only rebalance when weight drift exceeds this.")
    parser.add_argument("--allow-lookahead", action="store_true",
                        help="needed for --target-method cpm / cpm_global, whose CPM settings and labels use the whole series (reproduction only)")
    parser.add_argument("--execution", choices=["same_close", "next_open"], default="next_open",
                        help="next_open = fill at the next day's open (added 2026-10-15; the default since 2026-11-04); same_close = the original v1 "
                             "fill at the signal day's close, needed to reproduce the v1 numbers.")
    parser.add_argument("--train-window", choices=["rolling", "expanding"], default="rolling",
                        help="rolling = train on only the last --train-years before validation (the default since 2026-11-04); "
                             "expanding = every row since the start of the data (the original v1 window, needed to reproduce the v1 numbers).")
    parser.add_argument("--readout", choices=["input_state", "state"], default="input_state",
                        help="input_state = ridge on [1, 20 inputs, 100 states] (the v1 readout, default); state = ridge on [1, 100 states] only. "
                             "Added 2026-11-07.")
    parser.add_argument("--threshold-rule", choices=["accuracy", "quantile", "zero"], default="accuracy",
                        help="accuracy = the grid value with the best 3-class validation accuracy (v1, default); quantile = the "
                             "(1 - --signal-share) quantile of |training score|, so a buy or sell signal is on for about that share of "
                             "training days; zero = threshold 0 (the sign of the score). Added 2026-11-07.")
    parser.add_argument("--signal-share", type=float, default=0.2, help="Share of training days with a signal for --threshold-rule quantile.")
    args = parser.parse_args()

    if args.target_method in ("cpm", "cpm_global") and not args.allow_lookahead:
        raise SystemExit(f"--target-method {args.target_method} chooses CPM's P and T (and places turning points) with the whole series, i.e. with "
                         "prices after every test year. Use cpm_causal; pass --allow-lookahead only to reproduce the old numbers.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tp_path)
    df = build_ta_features(df)

    global_cpm = None
    if args.target_method == "cpm":
        turning_points = _turning_points_from_df(df)
        df["target"] = build_triangle_target(df, turning_points)
    elif args.target_method == "cpm_global":
        p_value, t_value = _pareto_knee_params(df["Close"].astype(float))
        df["target"] = build_triangle_target(df, detect_turning_points(df["Close"].astype(float), P=p_value, T=t_value))
        global_cpm = {"P": p_value, "T": t_value}
    elif args.target_method == "cpm_causal":
        df["target"] = np.nan                                         # built per fold
    else:
        df["target"] = label_triple_barrier(df, k=args.tb_atr_mult, max_holding=args.tb_max_holding)

    subset_cols = [*TA_FEATURE_COLUMNS]
    if args.target_method in ("cpm", "cpm_global"):
        subset_cols.append("target")
    df = df.dropna(subset=subset_cols).reset_index(drop=True)

    vol_series = build_ohlcv_features(df)["ohlcv_realized_vol_20"].to_numpy() if args.risk_overlay else None

    dates_ts = pd.to_datetime(df["Date"])
    dates_np = dates_ts.to_numpy()
    n_rows = len(df)
    start_date = dates_ts.iloc[0]

    def idx_for(ts: pd.Timestamp) -> int:
        return int(np.searchsorted(dates_np, np.datetime64(ts)))

    X_full = df[TA_FEATURE_COLUMNS].astype(float).to_numpy()
    y_full = df["target"].astype(float).to_numpy() if args.target_method in ("cpm", "cpm_global") else None
    close_series = df["Close"].astype(float)
    price_full = df["Close"].astype(float).to_numpy()
    open_full = df["Open"].astype(float).to_numpy()
    date_full = df["Date"].to_numpy()

    washout = int(ESN_CONFIG["washout"])
    threshold_values = list(ESN_CONFIG["threshold_values"])
    val_years = int(args.val_years)

    fold_frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    fold_idx = 0
    cur_date = start_date + pd.DateOffset(years=args.train_years + val_years)

    while idx_for(cur_date) < n_rows:
        fold_idx += 1
        fold_test_start_date = cur_date
        fold_test_end_date = cur_date + pd.DateOffset(years=args.step_years)
        fold_val_start_date = fold_test_start_date - pd.DateOffset(years=val_years)

        fold_test_start = idx_for(fold_test_start_date)
        fold_test_end = min(idx_for(fold_test_end_date), n_rows)
        fold_val_start = idx_for(fold_val_start_date)
        train_fit_end = fold_val_start
        # rolling (default): only the last train_years before validation; expanding (v1): every row from the start
        train_start = idx_for(fold_val_start_date - pd.DateOffset(years=args.train_years)) if args.train_window == "rolling" else 0
        if fold_test_end <= fold_test_start:
            break

        scaler = MinMaxUnitScaler()
        if args.target_method == "triple_barrier":
            train_target = label_triple_barrier(       # labelled on every row up to the training end (ATR needs no warm-up), then cut
                df.iloc[:train_fit_end],
                k=args.tb_atr_mult,
                max_holding=args.tb_max_holding,
            ).iloc[train_start:]
            val_target = label_triple_barrier(
                df.iloc[:fold_test_start],
                k=args.tb_atr_mult,
                max_holding=args.tb_max_holding,
            )
            train_mask = train_target.notna().to_numpy() & _full_horizon_mask(
                len(train_target), args.tb_max_holding
            )
            val_target_slice = val_target.iloc[fold_val_start:fold_test_start]
            val_positions = np.arange(fold_val_start, fold_test_start)
            val_mask = (
                val_target_slice.notna().to_numpy()
                & ((val_positions + int(args.tb_max_holding)) < fold_test_start)
            )

            if int(train_mask.sum()) <= washout:
                raise ValueError(
                    f"Fold {fold_idx}: purged train rows ({int(train_mask.sum())}) "
                    f"must exceed washout ({washout})."
                )
            if int(val_mask.sum()) == 0:
                raise ValueError(f"Fold {fold_idx}: no validation rows remain after purging.")

            X_train = scaler.fit_transform(X_full[train_start:train_fit_end][train_mask])
            y_train = train_target.to_numpy(dtype=float)[train_mask]
            X_val = scaler.transform(X_full[fold_val_start:fold_test_start][val_mask])
            y_val = val_target_slice.to_numpy(dtype=float)[val_mask]
            target_for_output = label_triple_barrier(
                df.iloc[:fold_test_end],
                k=args.tb_atr_mult,
                max_holding=args.tb_max_holding,
            ).iloc[fold_test_start:fold_test_end].to_numpy(dtype=float)
        elif args.target_method == "cpm_causal":
            # P, T chosen on the training rows only; every CPM run starts at train_start (0 with --train-window expanding)
            p_value, t_value = _pareto_knee_params(close_series.iloc[train_start:train_fit_end])
            train_target = _known_triangle(close_series.iloc[train_start:train_fit_end], p_value, t_value)
            val_slice = _known_triangle(close_series.iloc[train_start:fold_test_start], p_value, t_value)[fold_val_start - train_start:]
            train_mask, val_mask = ~np.isnan(train_target), ~np.isnan(val_slice)
            if int(train_mask.sum()) <= washout:
                raise ValueError(f"Fold {fold_idx}: only {int(train_mask.sum())} training rows have a determined CPM target.")
            X_train = scaler.fit_transform(X_full[train_start:train_fit_end][train_mask])
            y_train = train_target[train_mask]
            X_val = scaler.transform(X_full[fold_val_start:fold_test_start][val_mask]) if val_mask.any() else None   # None: no turning
            y_val = val_slice[val_mask]                                                                              # point is settled yet
            target_for_output = _known_triangle(close_series.iloc[train_start:fold_test_end], p_value, t_value)[fold_test_start - train_start:]
        else:
            X_train = scaler.fit_transform(X_full[train_start:train_fit_end])
            y_train = y_full[train_start:train_fit_end]
            X_val = scaler.transform(X_full[fold_val_start:fold_test_start])
            y_val = y_full[fold_val_start:fold_test_start]
            target_for_output = y_full[fold_test_start:fold_test_end]
        X_test = scaler.transform(X_full[fold_test_start:fold_test_end])

        model = _fit_esn(
            X_train, y_train, washout,
            random_state=args.random_state, ensemble_size=args.ensemble_size, readout_inputs=args.readout == "input_state",
        )

        if args.threshold_rule == "quantile":      # from the training scores only; no labels used
            train_scores = np.asarray(model.predict(X_train), dtype=float)[washout:]
            selected = {"threshold": float(np.quantile(np.abs(train_scores), 1 - args.signal_share)), "validation_accuracy": None}
        elif args.threshold_rule == "zero":
            selected = {"threshold": 0.0, "validation_accuracy": None}
        elif X_val is None:                        # cpm_causal only: keep the previous fold's threshold (0.1 for the first fold)
            selected = {"threshold": fold_summaries[-1]["selected_threshold"] if fold_summaries else 0.1, "validation_accuracy": None}
        else:
            val_pred = np.asarray(model.predict(X_val), dtype=float)
            selected = _select_thresholds(y_val, val_pred, threshold_values)

        test_pred = np.asarray(model.predict(X_test), dtype=float)
        test_signal = np.asarray(score_to_signal(test_pred, selected["threshold"]), dtype=int)

        fold_df = pd.DataFrame(
            {
                "date": date_full[fold_test_start:fold_test_end],
                "close": price_full[fold_test_start:fold_test_end],
                "open": open_full[fold_test_start:fold_test_end],
                "target": target_for_output,
                "trading_score": test_pred,
                "signal": test_signal,
                "fold": fold_idx,
            }
        )
        if vol_series is not None:
            fold_df["vol"] = vol_series[fold_test_start:fold_test_end]
        fold_frames.append(fold_df)

        fold_summary = {
            "fold": fold_idx,
            "train_rows": train_fit_end - train_start,
            "train_start_date": str(date_full[train_start]),
            "val_rows": fold_test_start - fold_val_start,
            "test_rows": fold_test_end - fold_test_start,
            "test_start_date": str(date_full[fold_test_start]),
            "test_end_date": str(date_full[fold_test_end - 1]),
            "selected_threshold": selected["threshold"],
            "val_accuracy": selected["validation_accuracy"],
        }
        if args.target_method in ("triple_barrier", "cpm_causal"):
            fold_summary["purged_train_rows"] = int(train_mask.sum())
            fold_summary["purged_val_rows"] = int(val_mask.sum())
        if args.target_method == "cpm_causal":
            fold_summary["cpm"] = {"P": p_value, "T": t_value}
        fold_summaries.append(fold_summary)
        accuracy = (f"n/a (--threshold-rule {args.threshold_rule})" if args.threshold_rule != "accuracy"
                    else "n/a (no settled validation target)" if selected["validation_accuracy"] is None else f"{selected['validation_accuracy']:.4f}")
        print(f"fold {fold_idx}: {fold_summary['test_start_date']} ~ {fold_summary['test_end_date']} "
              f"threshold={selected['threshold']} val_acc={accuracy}" + (f" CPM P={p_value} T={t_value}" if args.target_method == "cpm_causal" else ""))

        cur_date = fold_test_end_date

    predictions_df = pd.concat(fold_frames, ignore_index=True)
    if args.risk_overlay:
        equity_curve, trades, summary = run_long_only_backtest_vol_target(
            predictions_df, price_col="close", signal_col="signal", vol_col="vol",
            target_vol=args.target_vol, rebalance_band=args.rebalance_band,
            initial_cash=args.initial_cash, fee_rate=args.fee_rate, slippage_rate=args.slippage_rate, execution=args.execution,
        )
    else:
        equity_curve, trades, summary = run_long_only_backtest(
            predictions_df,
            price_col="close",
            signal_col="signal",
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            execution=args.execution,
        )

    result = {
        "label": f"ta-feature walk-forward: {args.train_years}y train / {val_years}y val / {args.step_years}y test",
        "target_method": args.target_method,
        "target_params": (
            {"atr_mult": args.tb_atr_mult, "max_holding": args.tb_max_holding}
            if args.target_method == "triple_barrier"
            else {}
        ),
        "num_folds": len(fold_summaries),
        "folds": fold_summaries,
        "test_period": [str(predictions_df["date"].iloc[0]), str(predictions_df["date"].iloc[-1])],
        "test_rows": int(len(predictions_df)),
        "execution": args.execution,
        "train_window": args.train_window,
        "readout": args.readout,
        "threshold_rule": args.threshold_rule,
        "signal_share": args.signal_share if args.threshold_rule == "quantile" else None,
        "cpm": global_cpm,
        "strategy": summary,
    }

    save_dataframe(predictions_df, output_dir / "walkforward_predictions.csv")
    save_dataframe(equity_curve, output_dir / "equity_curve.csv")
    save_dataframe(trades, output_dir / "trades.csv")
    save_json(result, output_dir / "summary.json")

    print(f"\nTest period: {result['test_period']}, rows: {result['test_rows']}, folds: {len(fold_summaries)}")
    print(f"Total return: {summary['total_return']:.4f}, annualized: {summary['annualized_return']:.4f}, "
          f"sharpe: {summary['sharpe_ratio']:.4f}, mdd: {summary['max_drawdown']:.4f}, trades: {summary['num_trades']}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
