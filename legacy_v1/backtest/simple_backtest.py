# legacy_v1 copy of src/backtest/simple_backtest.py (frozen v1 recipe). Only import paths were changed.
"""Simple long-only backtest for ESN trading-score signals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["Close", "close", "Adj Close", "adj_close"]
DATE_COLUMNS = ["date", "Date", "datetime", "Datetime"]


def infer_price_col(df: pd.DataFrame, price_col: str | None = None) -> str:
    """Return an available close-price column."""

    candidates = [price_col] if price_col else []
    candidates.extend(col for col in PRICE_COLUMNS if col not in candidates)
    for col in candidates:
        if col and col in df.columns:
            return col
    raise ValueError(f"No price column found. Checked: {candidates}")


def infer_date_values(df: pd.DataFrame) -> pd.Series:
    """Return date values from a common date column or from the index."""

    for col in DATE_COLUMNS:
        if col in df.columns:
            return df[col].reset_index(drop=True)
    return pd.Series(df.index, name="date").reset_index(drop=True)


def normalize_signal(value: Any) -> int:
    """Normalize numeric or string trade signals to {-1, 0, 1}."""

    if pd.isna(value):
        return 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"buy", "-1"}:
            return -1
        if lowered in {"sell", "1"}:
            return 1
        if lowered in {"hold", "0", ""}:
            return 0
    numeric = float(value)
    if numeric < 0:
        return -1
    if numeric > 0:
        return 1
    return 0


def run_long_only_backtest(
    df: pd.DataFrame,
    price_col: str | None = None,
    signal_col: str = "signal",
    initial_cash: float = 1.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    capital_gains_tax_rate: float = 0.0,
    allow_fractional: bool = True,
    execution: str = "next_open",
    open_col: str = "open",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run a long-only all-in/all-out backtest from ESN signals.

    ``execution``: "same_close" (the original v1 behaviour) trades at the close of the very day whose close produced the signal;
    "next_open" (added 2026-10-15, the default since 2026-11-04) trades a signal of day t at the open of day t+1, which is when it
    could really be acted on.
    The portfolio is marked at each close in both cases.

    Cost model:
    - ``fee_rate`` is charged on executed notional for buys and sells.
    - ``slippage_rate`` worsens execution price: buys execute higher and sells lower.
    - ``sell_tax_rate`` is charged on sell notional.
    - ``capital_gains_tax_rate`` is charged on positive realized gains when selling.

    The tax model is intentionally generic and simplified. Use it for scenario
    analysis, not jurisdiction-specific tax reporting.
    """

    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")
    if initial_cash <= 0.0:
        raise ValueError(f"initial_cash must be positive, got {initial_cash}.")
    if fee_rate < 0.0:
        raise ValueError(f"fee_rate must be non-negative, got {fee_rate}.")
    if slippage_rate < 0.0:
        raise ValueError(f"slippage_rate must be non-negative, got {slippage_rate}.")
    if sell_tax_rate < 0.0:
        raise ValueError(f"sell_tax_rate must be non-negative, got {sell_tax_rate}.")
    if capital_gains_tax_rate < 0.0:
        raise ValueError(
            f"capital_gains_tax_rate must be non-negative, got {capital_gains_tax_rate}."
        )

    resolved_price_col = infer_price_col(df, price_col)
    working = df.copy().reset_index(drop=True)
    dates = infer_date_values(working)
    close = working[resolved_price_col].astype(float).reset_index(drop=True)
    signals = working[signal_col].map(normalize_signal).astype(int).reset_index(drop=True)
    next_open = _check_execution(execution, working, open_col)
    opens = working[open_col].astype(float).reset_index(drop=True) if next_open else close

    cash = float(initial_cash)
    shares = 0.0
    cost_basis = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for i, price in enumerate(close):
        signal = int(signals.iloc[i])
        acted = (int(signals.iloc[i - 1]) if i > 0 else 0) if next_open else signal       # the signal traded today
        mark_price = float(opens.iloc[i]) if next_open else float(price)                  # the price it is traded at
        action = "hold"
        execution_price = mark_price
        fee = 0.0
        sell_tax = 0.0
        capital_gains_tax = 0.0
        realized_gain = 0.0

        if acted == -1 and shares <= 0.0 and cash > 0.0:
            execution_price = mark_price * (1.0 + slippage_rate)
            if allow_fractional:
                buy_shares = cash / (execution_price * (1.0 + fee_rate))
            else:
                buy_shares = math.floor(cash / (execution_price * (1.0 + fee_rate)))
            if buy_shares > 0.0:
                gross_cost = buy_shares * execution_price
                fee = gross_cost * fee_rate
                cash -= gross_cost + fee
                shares += buy_shares
                cost_basis = gross_cost + fee
                action = "buy"
        elif acted == 1 and shares > 0.0:
            sold_shares = shares
            execution_price = mark_price * (1.0 - slippage_rate)
            gross_value = shares * execution_price
            fee = gross_value * fee_rate
            sell_tax = gross_value * sell_tax_rate
            net_before_capital_gains_tax = gross_value - fee - sell_tax
            realized_gain = net_before_capital_gains_tax - cost_basis
            capital_gains_tax = max(0.0, realized_gain) * capital_gains_tax_rate
            cash += net_before_capital_gains_tax - capital_gains_tax
            shares = 0.0
            cost_basis = 0.0
            action = "sell"

        portfolio_value = cash + shares * float(price)
        position = 1 if shares > 0.0 else 0
        if action in {"buy", "sell"}:
            trades.append(
                {
                    "date": dates.iloc[i],
                    "action": action,
                    "price": mark_price,
                    "execution_price": float(execution_price),
                    "shares": float(shares if action == "buy" else sold_shares),
                    "gross_notional": float(gross_cost if action == "buy" else gross_value),
                    "fee": float(fee),
                    "sell_tax": float(sell_tax),
                    "capital_gains_tax": float(capital_gains_tax),
                    "realized_gain": float(realized_gain),
                    "cash_after": float(cash),
                    "portfolio_value_after": float(portfolio_value),
                }
            )
        rows.append(
            {
                "date": dates.iloc[i],
                "close": float(price),
                "signal": signal,
                "position": position,
                "cash": float(cash),
                "shares": float(shares),
                "cost_basis": float(cost_basis),
                "portfolio_value": float(portfolio_value),
            }
        )

    equity_curve = pd.DataFrame(rows)
    if equity_curve.empty:
        equity_curve = _empty_equity_curve()
    equity_curve["daily_return"] = equity_curve["portfolio_value"].pct_change().fillna(0.0)
    equity_curve["cumulative_return"] = equity_curve["portfolio_value"] / float(initial_cash) - 1.0
    trades_df = pd.DataFrame(
        trades,
        columns=[
            "date",
            "action",
            "price",
            "execution_price",
            "shares",
            "gross_notional",
            "fee",
            "sell_tax",
            "capital_gains_tax",
            "realized_gain",
            "cash_after",
            "portfolio_value_after",
        ],
    )
    summary = _strategy_summary(equity_curve, trades_df, initial_cash)
    summary["cost_model"] = {
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
        "sell_tax_rate": float(sell_tax_rate),
        "capital_gains_tax_rate": float(capital_gains_tax_rate),
    }
    summary["buy_and_hold"] = buy_and_hold_summary(
        df=working,
        price_col=resolved_price_col,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        sell_tax_rate=sell_tax_rate,
        capital_gains_tax_rate=capital_gains_tax_rate,
        allow_fractional=allow_fractional,
    )
    return equity_curve, trades_df, summary


def run_long_only_backtest_vol_target(
    df: pd.DataFrame,
    price_col: str | None = None,
    signal_col: str = "signal",
    vol_col: str = "vol",
    target_vol: float = 0.15,
    rebalance_band: float = 0.1,
    initial_cash: float = 1.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    capital_gains_tax_rate: float = 0.0,
    execution: str = "next_open",
    open_col: str = "open",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Long-only backtest with volatility-targeted fractional position sizing.

    ``execution`` as in run_long_only_backtest: with "next_open" the target weight decided from day t's signal and volatility is
    traded at the open of day t+1.

    A risk overlay on top of run_long_only_backtest's binary all-in/all-out
    logic: instead of always going 100% long on a buy signal, the desired
    exposure is ``min(1.0, target_vol / vol_col[t])`` -- shrinking position
    size when recent realized volatility is above the target and staying
    full-size otherwise. Once in a position (from a buy signal), exposure is
    re-targeted every day using the *current* day's volatility even on
    "hold" days, so a vol spike while already holding still de-risks the
    position; a sell signal always exits fully to 0. Actual holdings only
    move toward the target when the drift exceeds ``rebalance_band``, to
    avoid daily vol wiggle generating excessive rebalancing trades/fees.
    """

    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")
    if vol_col not in df.columns:
        raise ValueError(f"Volatility column not found: {vol_col}")
    if initial_cash <= 0.0:
        raise ValueError(f"initial_cash must be positive, got {initial_cash}.")
    if target_vol <= 0.0:
        raise ValueError(f"target_vol must be positive, got {target_vol}.")
    if not 0.0 <= rebalance_band < 1.0:
        raise ValueError(f"rebalance_band must be in [0, 1), got {rebalance_band}.")

    resolved_price_col = infer_price_col(df, price_col)
    working = df.copy().reset_index(drop=True)
    dates = infer_date_values(working)
    close = working[resolved_price_col].astype(float).reset_index(drop=True)
    signals = working[signal_col].map(normalize_signal).astype(int).reset_index(drop=True)
    vol = working[vol_col].astype(float).reset_index(drop=True)
    next_open = _check_execution(execution, working, open_col)
    opens = working[open_col].astype(float).reset_index(drop=True) if next_open else close

    cash = float(initial_cash)
    shares = 0.0
    cost_basis = 0.0
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for i, price in enumerate(close):
        signal = int(signals.iloc[i])
        if next_open:                                                   # yesterday's decision, traded at today's open
            acted = int(signals.iloc[i - 1]) if i > 0 else 0
            v = float(vol.iloc[i - 1]) if i > 0 else float("nan")
            mark_price = float(opens.iloc[i])
        else:
            acted, v, mark_price = signal, float(vol.iloc[i]), float(price)
        portfolio_value_pre = cash + shares * mark_price
        current_weight = (shares * mark_price / portfolio_value_pre) if portfolio_value_pre > 0 else 0.0

        if acted == 1:
            desired_weight = 0.0
        elif acted == -1 or shares > 0.0:
            desired_weight = min(1.0, target_vol / v) if (not np.isnan(v) and v > 0.0) else current_weight
        else:
            desired_weight = 0.0

        action = "hold"
        execution_price = mark_price
        fee = sell_tax = capital_gains_tax = realized_gain = traded_shares = 0.0

        if desired_weight > current_weight + rebalance_band and cash > 0.0:
            target_value = desired_weight * portfolio_value_pre
            buy_notional = min(target_value - shares * mark_price, cash)
            if buy_notional > 0.0:
                execution_price = mark_price * (1.0 + slippage_rate)
                traded_shares = buy_notional / (execution_price * (1.0 + fee_rate))
                gross_cost = traded_shares * execution_price
                fee = gross_cost * fee_rate
                cash -= gross_cost + fee
                shares += traded_shares
                cost_basis += gross_cost + fee
                action = "buy" if current_weight <= 1e-9 else "rebalance_up"
        elif desired_weight < current_weight - rebalance_band and shares > 0.0:
            target_value = desired_weight * portfolio_value_pre
            sell_shares = min(shares, max(0.0, (shares * mark_price - target_value) / mark_price))
            if sell_shares > 0.0:
                execution_price = mark_price * (1.0 - slippage_rate)
                gross_value = sell_shares * execution_price
                fee = gross_value * fee_rate
                sell_tax = gross_value * sell_tax_rate
                net_before_cgt = gross_value - fee - sell_tax
                proportion = sell_shares / shares
                basis_sold = cost_basis * proportion
                realized_gain = net_before_cgt - basis_sold
                capital_gains_tax = max(0.0, realized_gain) * capital_gains_tax_rate
                cash += net_before_cgt - capital_gains_tax
                shares -= sell_shares
                cost_basis -= basis_sold
                traded_shares = sell_shares
                action = "sell" if desired_weight <= 1e-9 else "rebalance_down"

        portfolio_value = cash + shares * float(price)
        position = 1 if shares > 0.0 else 0
        if action != "hold":
            trades.append(
                {
                    "date": dates.iloc[i],
                    "action": "buy" if action in ("buy", "rebalance_up") else "sell",
                    "price": mark_price,
                    "execution_price": float(execution_price),
                    "shares": float(traded_shares),
                    "gross_notional": float(traded_shares * execution_price),
                    "fee": float(fee),
                    "sell_tax": float(sell_tax),
                    "capital_gains_tax": float(capital_gains_tax),
                    "realized_gain": float(realized_gain),
                    "cash_after": float(cash),
                    "portfolio_value_after": float(portfolio_value),
                }
            )
        rows.append(
            {
                "date": dates.iloc[i],
                "close": float(price),
                "signal": signal,
                "vol": v,
                "target_weight": float(desired_weight),
                "actual_weight": float(shares * float(price) / portfolio_value) if portfolio_value > 0 else 0.0,
                "position": position,
                "cash": float(cash),
                "shares": float(shares),
                "portfolio_value": float(portfolio_value),
            }
        )

    equity_curve = pd.DataFrame(rows)
    if equity_curve.empty:
        equity_curve = _empty_equity_curve()
    equity_curve["daily_return"] = equity_curve["portfolio_value"].pct_change().fillna(0.0)
    equity_curve["cumulative_return"] = equity_curve["portfolio_value"] / float(initial_cash) - 1.0
    trades_df = pd.DataFrame(
        trades,
        columns=[
            "date", "action", "price", "execution_price", "shares", "gross_notional",
            "fee", "sell_tax", "capital_gains_tax", "realized_gain", "cash_after", "portfolio_value_after",
        ],
    )
    summary = _strategy_summary(equity_curve, trades_df, initial_cash)
    summary["cost_model"] = {
        "fee_rate": float(fee_rate), "slippage_rate": float(slippage_rate),
        "sell_tax_rate": float(sell_tax_rate), "capital_gains_tax_rate": float(capital_gains_tax_rate),
    }
    summary["risk_overlay"] = {"target_vol": float(target_vol), "rebalance_band": float(rebalance_band)}
    summary["buy_and_hold"] = buy_and_hold_summary(
        df=working, price_col=resolved_price_col, initial_cash=initial_cash,
        fee_rate=fee_rate, slippage_rate=slippage_rate,
        sell_tax_rate=sell_tax_rate, capital_gains_tax_rate=capital_gains_tax_rate,
    )
    return equity_curve, trades_df, summary


def _check_execution(execution: str, df: pd.DataFrame, open_col: str) -> bool:
    if execution not in ("same_close", "next_open"):
        raise ValueError(f"execution must be 'same_close' or 'next_open', got {execution!r}.")
    if execution == "next_open" and open_col not in df.columns:
        raise ValueError(f"execution='next_open' needs an open-price column {open_col!r}.")
    return execution == "next_open"


def buy_and_hold_summary(
    df: pd.DataFrame,
    price_col: str | None = None,
    initial_cash: float = 1.0,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    capital_gains_tax_rate: float = 0.0,
    allow_fractional: bool = True,
) -> dict[str, float]:
    """Return buy-and-hold metrics over the same period.

    The final value is marked to market, matching the strategy equity curve.
    Therefore exit-only costs/taxes are not deducted unless a sell signal occurs.
    """

    resolved_price_col = infer_price_col(df, price_col)
    close = df[resolved_price_col].astype(float).reset_index(drop=True)
    if close.empty:
        return _metric_summary(pd.Series(dtype=float), initial_cash)

    first_price = float(close.iloc[0])
    execution_price = first_price * (1.0 + slippage_rate)
    if allow_fractional:
        shares = initial_cash / (execution_price * (1.0 + fee_rate))
    else:
        shares = math.floor(initial_cash / (execution_price * (1.0 + fee_rate)))
    cash = initial_cash - shares * execution_price * (1.0 + fee_rate)
    equity = cash + shares * close
    metrics = _metric_summary(equity.astype(float), initial_cash)
    metrics["cost_model"] = {
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
        "sell_tax_rate": float(sell_tax_rate),
        "capital_gains_tax_rate": float(capital_gains_tax_rate),
    }
    return metrics


def _strategy_summary(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    metrics = _metric_summary(equity_curve["portfolio_value"], initial_cash)
    num_buy = int((trades["action"] == "buy").sum()) if not trades.empty else 0
    num_sell = int((trades["action"] == "sell").sum()) if not trades.empty else 0
    return {
        "initial_cash": float(initial_cash),
        **metrics,
        "num_trades": int(len(trades)),
        "num_buy": num_buy,
        "num_sell": num_sell,
        "exposure_ratio": float(equity_curve["position"].mean()) if len(equity_curve) else 0.0,
        "win_rate": _win_rate(trades),
    }


def _metric_summary(equity: pd.Series, initial_cash: float) -> dict[str, float]:
    if equity.empty:
        return {
            "final_value": float(initial_cash),
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    final_value = float(equity.iloc[-1])
    total_return = final_value / float(initial_cash) - 1.0
    n_days = len(equity)
    annualized_return = (final_value / float(initial_cash)) ** (252.0 / n_days) - 1.0
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    daily_return = equity.pct_change().fillna(0.0)
    std = float(daily_return.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * daily_return.mean() / std) if std > 0.0 else 0.0
    return {
        "final_value": final_value,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe,
    }


def convert_equity_to_krw(
    equity_curve: pd.DataFrame,
    fx_rate: np.ndarray | pd.Series,
    initial_cash_usd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reprice a USD equity curve (from run_long_only_backtest*) into KRW.

    Assumes continuous FX exposure regardless of position: uninvested cash
    is assumed to sit in USD (a money-market account, say) rather than being
    converted back to KRW between trades, so every day's portfolio_value is
    simply multiplied by that day's USD/KRW rate. ``fx_rate`` must be aligned
    1:1 with ``equity_curve``'s rows (same length and order).
    """

    fx = np.asarray(fx_rate, dtype=float)
    if len(fx) != len(equity_curve):
        raise ValueError(f"fx_rate length {len(fx)} != equity_curve length {len(equity_curve)}.")

    krw_curve = equity_curve.copy()
    krw_curve["fx_rate"] = fx
    krw_curve["portfolio_value_usd"] = equity_curve["portfolio_value"]
    krw_curve["portfolio_value"] = equity_curve["portfolio_value"].to_numpy() * fx

    initial_cash_krw = initial_cash_usd * fx[0] if len(fx) else initial_cash_usd
    metrics = _metric_summary(krw_curve["portfolio_value"], initial_cash_krw)
    metrics["initial_cash_krw"] = float(initial_cash_krw)
    metrics["fx_rate_start"] = float(fx[0]) if len(fx) else None
    metrics["fx_rate_end"] = float(fx[-1]) if len(fx) else None
    return krw_curve, metrics


def _win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    buys = trades[trades["action"] == "buy"].reset_index(drop=True)
    sells = trades[trades["action"] == "sell"].reset_index(drop=True)
    n_pairs = min(len(buys), len(sells))
    if n_pairs == 0:
        return 0.0
    wins = 0
    for i in range(n_pairs):
        if float(sells.loc[i, "portfolio_value_after"]) > float(buys.loc[i, "portfolio_value_after"]):
            wins += 1
    return float(wins / n_pairs)


def _empty_equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "close",
            "signal",
            "position",
            "cash",
            "shares",
            "portfolio_value",
        ]
    )
