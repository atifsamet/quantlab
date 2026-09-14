"""Unit tests for backtest metrics (no network)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.backtest.metrics import compute_metrics, max_drawdown_pct, profit_factor
from app.backtest.models import EquityPoint, PositionSide, Trade


def _trade(net: float, trade_id: int = 1) -> Trade:
    return Trade(
        trade_id=trade_id,
        side=PositionSide.LONG,
        entry_timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        exit_timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
        entry_price=100.0,
        exit_price=110.0,
        quantity=1.0,
        gross_pnl=net,
        fees=0.0,
        net_pnl=net,
        return_pct=net,
    )


def test_max_drawdown() -> None:
    curve = [
        EquityPoint(pd.Timestamp("2024-01-01", tz="UTC"), 1000.0),
        EquityPoint(pd.Timestamp("2024-01-02", tz="UTC"), 1100.0),
        EquityPoint(pd.Timestamp("2024-01-03", tz="UTC"), 880.0),
        EquityPoint(pd.Timestamp("2024-01-04", tz="UTC"), 990.0),
    ]
    # Peak 1100 → 880 = 20% drawdown
    assert max_drawdown_pct(curve) == pytest.approx(20.0)


def test_max_drawdown_empty() -> None:
    assert max_drawdown_pct([]) == 0.0


def test_profit_factor_normal() -> None:
    assert profit_factor(150.0, -50.0) == pytest.approx(3.0)


def test_profit_factor_zero_loss_with_wins() -> None:
    assert math.isinf(profit_factor(100.0, 0.0))


def test_profit_factor_zero_trades() -> None:
    assert profit_factor(0.0, 0.0) == 0.0


def test_win_rate_and_best_worst() -> None:
    trades = [_trade(50.0, 1), _trade(-20.0, 2), _trade(10.0, 3)]
    curve = [
        EquityPoint(pd.Timestamp("2024-01-01", tz="UTC"), 1000.0),
        EquityPoint(pd.Timestamp("2024-01-02", tz="UTC"), 1040.0),
    ]
    result = compute_metrics(
        initial_capital=1000.0,
        final_equity=1040.0,
        trades=trades,
        equity_curve=curve,
    )
    assert result.total_trades == 3
    assert result.winning_trades == 2
    assert result.losing_trades == 1
    assert result.win_rate == pytest.approx(2 / 3 * 100.0)
    assert result.gross_profit == pytest.approx(60.0)
    assert result.gross_loss == pytest.approx(-20.0)
    assert result.profit_factor == pytest.approx(3.0)
    assert result.average_trade_pnl == pytest.approx(40.0 / 3.0)
    assert result.best_trade_pnl == pytest.approx(50.0)
    assert result.worst_trade_pnl == pytest.approx(-20.0)
    assert result.total_return_pct == pytest.approx(4.0)


def test_zero_trade_metrics() -> None:
    curve = [EquityPoint(pd.Timestamp("2024-01-01", tz="UTC"), 1000.0)]
    result = compute_metrics(
        initial_capital=1000.0,
        final_equity=1000.0,
        trades=[],
        equity_curve=curve,
    )
    assert result.total_trades == 0
    assert result.win_rate == 0.0
    assert result.profit_factor == 0.0
    assert result.best_trade_pnl is None
    assert result.worst_trade_pnl is None
    assert result.max_drawdown_pct == 0.0
