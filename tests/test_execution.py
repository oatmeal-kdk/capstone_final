"""Regression checks for the fill timing of simple_backtest.run_long_only_backtest (README section 6, "체결")."""

import pandas as pd
import pytest

from legacy_v1.backtest.simple_backtest import run_long_only_backtest

# signal convention: -1 buy, +1 sell, 0 keep the position
FRAME = pd.DataFrame({
    "date": pd.bdate_range("2020-01-01", periods=6).strftime("%Y-%m-%d"),
    "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
    "signal": [0, -1, 0, 0, 1, 0],
})


def test_next_open_trades_a_signal_at_the_following_days_open():
    curve, trades, _ = run_long_only_backtest(FRAME, price_col="close", execution="next_open")
    assert list(trades["action"]) == ["buy", "sell"]
    assert list(trades["date"]) == [FRAME["date"][2], FRAME["date"][5]]       # signal on day 1 -> day 2, day 4 -> day 5
    assert list(trades["execution_price"]) == [12.0, 15.0]
    assert curve["position"].tolist() == [0, 0, 1, 1, 1, 0]
    assert curve["portfolio_value"].iloc[-1] == pytest.approx(15.0 / 12.0)


def test_same_close_trades_at_the_signal_days_close():
    _, trades, _ = run_long_only_backtest(FRAME, price_col="close", execution="same_close")
    assert list(trades["date"]) == [FRAME["date"][1], FRAME["date"][4]]
    assert list(trades["execution_price"]) == [11.5, 14.5]


def test_costs_worsen_the_fill_price():
    fee, slip = 0.0005, 0.0005
    curve, trades, _ = run_long_only_backtest(FRAME, price_col="close", execution="next_open", fee_rate=fee, slippage_rate=slip)
    assert trades["execution_price"].iloc[0] == pytest.approx(12.0 * (1 + slip))
    assert trades["execution_price"].iloc[1] == pytest.approx(15.0 * (1 - slip))
    expected = (1 / (12.0 * (1 + slip) * (1 + fee))) * 15.0 * (1 - slip) * (1 - fee)
    assert curve["portfolio_value"].iloc[-1] == pytest.approx(expected)
