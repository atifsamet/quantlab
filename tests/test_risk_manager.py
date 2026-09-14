"""Unit tests for Phase 5 risk management (offline, deterministic)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    resolve_sl_tp_hit,
)
from app.backtest.models import PositionSide
from app.risk import RiskConfig, RiskDecision, RiskManager
from app.risk.manager import (
    compute_position_quantity,
    compute_stop_loss,
    compute_take_profit,
)
from app.strategy.baseline import REQUIRED_INDICATORS


def test_long_stop_loss_calculation() -> None:
    stop = compute_stop_loss("LONG", entry_price=100.0, atr=2.0, atr_multiplier=1.5)
    assert stop == pytest.approx(97.0)


def test_short_stop_loss_calculation() -> None:
    stop = compute_stop_loss("SHORT", entry_price=100.0, atr=2.0, atr_multiplier=1.5)
    assert stop == pytest.approx(103.0)


def test_long_take_profit() -> None:
    stop = 97.0
    tp = compute_take_profit("LONG", 100.0, stop, risk_reward_ratio=2.0)
    assert tp == pytest.approx(106.0)


def test_short_take_profit() -> None:
    stop = 103.0
    tp = compute_take_profit("SHORT", 100.0, stop, risk_reward_ratio=2.0)
    assert tp == pytest.approx(94.0)


def test_position_sizing_example_long() -> None:
    qty, risk_amount = compute_position_quantity(
        equity=1000.0,
        risk_per_trade=0.01,
        maximum_risk_per_trade=0.02,
        entry_price=100.0,
        stop_loss=97.0,
        max_position_size_pct=1.0,
    )
    assert risk_amount == pytest.approx(10.0)
    assert qty == pytest.approx(10.0 / 3.0)


def test_position_sizing_example_short() -> None:
    qty, risk_amount = compute_position_quantity(
        equity=1000.0,
        risk_per_trade=0.01,
        maximum_risk_per_trade=0.02,
        entry_price=100.0,
        stop_loss=103.0,
        max_position_size_pct=1.0,
    )
    assert risk_amount == pytest.approx(10.0)
    assert qty == pytest.approx(10.0 / 3.0)


def test_maximum_position_size_cap() -> None:
    qty, _ = compute_position_quantity(
        equity=1000.0,
        risk_per_trade=0.01,
        maximum_risk_per_trade=0.02,
        entry_price=100.0,
        stop_loss=99.9,  # tiny stop → huge raw qty
        max_position_size_pct=0.1,  # max notional 100 → qty 1
    )
    assert qty == pytest.approx(1.0)


def test_valid_trade_decision() -> None:
    mgr = RiskManager(RiskConfig())
    decision = mgr.evaluate(
        side="LONG",
        entry_price=100.0,
        atr=2.0,
        equity=1000.0,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert isinstance(decision, RiskDecision)
    assert decision.approved is True
    assert decision.side == "LONG"
    assert decision.stop_loss == pytest.approx(97.0)
    assert decision.take_profit == pytest.approx(106.0)
    assert decision.risk_amount == pytest.approx(10.0)
    assert decision.quantity == pytest.approx(10.0 / 3.0)
    assert decision.risk_reward_ratio == pytest.approx(2.0)


def test_invalid_atr_rejected() -> None:
    mgr = RiskManager(RiskConfig())
    decision = mgr.evaluate(
        side="LONG",
        entry_price=100.0,
        atr=0.0,
        equity=1000.0,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert decision.approved is False
    assert "ATR" in decision.reason


def test_invalid_entry_rejected() -> None:
    mgr = RiskManager(RiskConfig())
    decision = mgr.evaluate(
        side="SHORT",
        entry_price=-1.0,
        atr=2.0,
        equity=1000.0,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert decision.approved is False
    assert "entry" in decision.reason.lower()


def test_rejected_hold_signal() -> None:
    mgr = RiskManager(RiskConfig())
    decision = mgr.evaluate(
        side="HOLD",
        entry_price=100.0,
        atr=2.0,
        equity=1000.0,
        timestamp="2024-01-01T00:00:00Z",
    )
    assert decision.approved is False


def test_daily_loss_limit() -> None:
    mgr = RiskManager(RiskConfig(max_daily_loss_pct=0.03))
    ts = "2024-01-01T12:00:00Z"
    # Realize -30 on 1000 start (-3%)
    mgr.register_closed_trade(-30.0, timestamp=ts, equity=970.0)
    decision = mgr.evaluate(
        side="LONG",
        entry_price=100.0,
        atr=2.0,
        equity=970.0,
        timestamp="2024-01-01T13:00:00Z",
    )
    assert decision.approved is False
    assert "daily loss" in decision.reason

    # New day resets daily loss tracking
    decision2 = mgr.evaluate(
        side="LONG",
        entry_price=100.0,
        atr=2.0,
        equity=970.0,
        timestamp="2024-01-02T00:00:00Z",
    )
    assert decision2.approved is True


def test_consecutive_loss_limit_and_reset() -> None:
    cfg = RiskConfig(max_consecutive_losses=3)
    mgr = RiskManager(cfg)
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    for i in range(3):
        mgr.register_closed_trade(-5.0, timestamp=ts + pd.Timedelta(minutes=i), equity=1000.0)

    blocked = mgr.evaluate(
        side="LONG",
        entry_price=100.0,
        atr=2.0,
        equity=985.0,
        timestamp=ts + pd.Timedelta(minutes=10),
    )
    assert blocked.approved is False
    assert "consecutive loss" in blocked.reason

    # Winning trade resets the block
    mgr.register_closed_trade(8.0, timestamp=ts + pd.Timedelta(minutes=11), equity=993.0)
    allowed = mgr.evaluate(
        side="LONG",
        entry_price=100.0,
        atr=2.0,
        equity=993.0,
        timestamp=ts + pd.Timedelta(minutes=12),
    )
    assert allowed.approved is True


def _risk_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in REQUIRED_INDICATORS:
        if col not in frame.columns:
            frame[col] = 1.0
    if "atr_14" not in frame.columns:
        frame["atr_14"] = 2.0
    return frame


def test_backtest_long_stop_loss_hit() -> None:
    # Entry at bar1 open=100 → SL=97, TP=106. Bar1 low hits 96 → SL.
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": "LONG",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 100.5,
            "low": 96.0,
            "close": 97.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 97.0,
            "high": 98.0,
            "low": 96.5,
            "close": 97.5,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
    ]
    engine = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
        risk_manager=RiskManager(RiskConfig()),
    )
    result = engine.run(_risk_frame(rows), compute_indicators=False, compute_signals=False)
    assert result.total_trades == 1
    assert result.trades[0].exit_price == pytest.approx(97.0)
    assert result.trades[0].net_pnl < 0


def test_backtest_long_take_profit_hit() -> None:
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": "LONG",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 107.0,
            "low": 99.5,
            "close": 106.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 106.0,
            "high": 107.0,
            "low": 105.0,
            "close": 106.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
    ]
    engine = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
        risk_manager=RiskManager(RiskConfig()),
    )
    result = engine.run(_risk_frame(rows), compute_indicators=False, compute_signals=False)
    assert result.total_trades == 1
    assert result.trades[0].exit_price == pytest.approx(106.0)
    assert result.trades[0].net_pnl > 0


def test_backtest_short_stop_loss_hit() -> None:
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": "SHORT",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 104.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 103.0,
            "high": 104.0,
            "low": 102.0,
            "close": 103.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
    ]
    engine = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
        risk_manager=RiskManager(RiskConfig()),
    )
    result = engine.run(_risk_frame(rows), compute_indicators=False, compute_signals=False)
    assert result.trades[0].exit_price == pytest.approx(103.0)
    assert result.trades[0].net_pnl < 0


def test_backtest_short_take_profit_hit() -> None:
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": "SHORT",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 100.5,
            "low": 93.0,
            "close": 94.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 94.0,
            "high": 95.0,
            "low": 93.0,
            "close": 94.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
    ]
    engine = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
        risk_manager=RiskManager(RiskConfig()),
    )
    result = engine.run(_risk_frame(rows), compute_indicators=False, compute_signals=False)
    assert result.trades[0].exit_price == pytest.approx(94.0)
    assert result.trades[0].net_pnl > 0


def test_same_candle_sl_tp_assumes_sl_first() -> None:
    assert (
        resolve_sl_tp_hit(
            side=PositionSide.LONG,
            stop_loss=97.0,
            take_profit=106.0,
            high=110.0,
            low=90.0,
        )
        == "SL"
    )
    assert (
        resolve_sl_tp_hit(
            side=PositionSide.SHORT,
            stop_loss=103.0,
            take_profit=94.0,
            high=110.0,
            low=90.0,
        )
        == "SL"
    )

    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": "LONG",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 105.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
    ]
    engine = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
        risk_manager=RiskManager(RiskConfig()),
    )
    result = engine.run(_risk_frame(rows), compute_indicators=False, compute_signals=False)
    assert result.trades[0].exit_price == pytest.approx(97.0)


def test_risk_no_lookahead_uses_next_open() -> None:
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 50.0,
            "high": 999.0,
            "low": 50.0,
            "close": 999.0,
            "volume": 1.0,
            "signal": "LONG",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1.0,
            "signal": "HOLD",
            "atr_14": 2.0,
        },
    ]
    engine = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
        risk_manager=RiskManager(RiskConfig()),
    )
    result = engine.run(_risk_frame(rows), compute_indicators=False, compute_signals=False)
    assert result.trades[0].entry_price == pytest.approx(100.0)


def test_phase4_behavior_without_risk_manager_unchanged() -> None:
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": "LONG",
        },
        {
            "timestamp": "2024-01-01T00:01:00Z",
            "open": 100.0,
            "high": 106.0,
            "low": 99.0,
            "close": 105.0,
            "volume": 1.0,
            "signal": "HOLD",
        },
        {
            "timestamp": "2024-01-01T00:02:00Z",
            "open": 105.0,
            "high": 111.0,
            "low": 104.0,
            "close": 110.0,
            "volume": 1.0,
            "signal": "HOLD",
        },
    ]
    frame = _risk_frame(rows)
    engine = BacktestEngine(config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(frame, compute_indicators=False, compute_signals=False)
    assert result.total_trades == 1
    assert result.trades[0].entry_price == pytest.approx(100.0)
    assert result.trades[0].exit_price == pytest.approx(110.0)
    assert result.final_equity == pytest.approx(1100.0)
    assert math.isclose(result.trades[0].quantity, 10.0)
