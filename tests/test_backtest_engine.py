"""Unit tests for the Phase 4 backtest engine (no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.backtest.engine import (
    BacktestConfig,
    BacktestDataError,
    BacktestEngine,
    load_ohlcv_csv,
)
from app.backtest.models import PositionSide
from app.strategy.baseline import REQUIRED_INDICATORS, BaselineStrategy, Signal


def _ts(i: int) -> str:
    return f"2024-01-01 00:{i:02d}:00+00:00"


def _frame(
    rows: list[tuple[float, float, str]],
) -> pd.DataFrame:
    """
    Build OHLCV + dummy indicators + preset signals.

    Each row tuple: (open, close, signal)
    high/low bracket the open/close; volume=1.
    """
    records = []
    for i, (open_px, close_px, signal) in enumerate(rows):
        high = max(open_px, close_px) + 1.0
        low = min(open_px, close_px) - 1.0
        records.append(
            {
                "timestamp": _ts(i),
                "open": open_px,
                "high": high,
                "low": low,
                "close": close_px,
                "volume": 1.0,
                "signal": signal,
            }
        )
    frame = pd.DataFrame.from_records(records)
    for col in REQUIRED_INDICATORS:
        frame[col] = 1.0
    return frame


def _run(
    rows: list[tuple[float, float, str]],
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    initial_capital: float = 1000.0,
):
    engine = BacktestEngine(
        config=BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        )
    )
    return engine.run(
        _frame(rows),
        compute_indicators=False,
        compute_signals=False,
    )


def test_simple_profitable_long() -> None:
    # Signal on bar0 → enter at bar1 open=100; force-close at last close=110
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 105.0, "HOLD"),
            (105.0, 110.0, "HOLD"),
        ],
    )
    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.side is PositionSide.LONG
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(110.0)
    assert trade.net_pnl == pytest.approx(100.0)
    assert result.final_equity == pytest.approx(1100.0)


def test_simple_profitable_short() -> None:
    result = _run(
        [
            (100.0, 100.0, "SHORT"),
            (100.0, 95.0, "HOLD"),
            (95.0, 90.0, "HOLD"),
        ],
    )
    assert result.total_trades == 1
    trade = result.trades[0]
    assert trade.side is PositionSide.SHORT
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(90.0)
    assert trade.net_pnl == pytest.approx(100.0)
    assert result.final_equity == pytest.approx(1100.0)


def test_losing_long() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 95.0, "HOLD"),
            (95.0, 90.0, "HOLD"),
        ],
    )
    assert result.trades[0].net_pnl == pytest.approx(-100.0)
    assert result.final_equity == pytest.approx(900.0)


def test_losing_short() -> None:
    result = _run(
        [
            (100.0, 100.0, "SHORT"),
            (100.0, 105.0, "HOLD"),
            (105.0, 110.0, "HOLD"),
        ],
    )
    assert result.trades[0].net_pnl == pytest.approx(-100.0)
    assert result.final_equity == pytest.approx(900.0)


def test_long_to_short_reversal() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 105.0, "HOLD"),
            (105.0, 110.0, "SHORT"),  # close long at next open, open short
            (110.0, 110.0, "HOLD"),
            (110.0, 100.0, "HOLD"),  # force-close short at 100
        ],
    )
    assert result.total_trades == 2
    assert result.trades[0].side is PositionSide.LONG
    assert result.trades[0].entry_price == pytest.approx(100.0)
    assert result.trades[0].exit_price == pytest.approx(110.0)
    assert result.trades[0].net_pnl == pytest.approx(100.0)
    assert result.trades[1].side is PositionSide.SHORT
    assert result.trades[1].entry_price == pytest.approx(110.0)
    assert result.trades[1].exit_price == pytest.approx(100.0)
    assert result.trades[1].net_pnl == pytest.approx(100.0)


def test_short_to_long_reversal() -> None:
    result = _run(
        [
            (100.0, 100.0, "SHORT"),
            (100.0, 95.0, "HOLD"),
            (95.0, 90.0, "LONG"),
            (90.0, 90.0, "HOLD"),
            (90.0, 100.0, "HOLD"),
        ],
    )
    assert result.total_trades == 2
    assert result.trades[0].side is PositionSide.SHORT
    assert result.trades[0].net_pnl == pytest.approx(100.0)
    assert result.trades[1].side is PositionSide.LONG
    assert result.trades[1].entry_price == pytest.approx(90.0)
    assert result.trades[1].exit_price == pytest.approx(100.0)
    assert result.trades[1].net_pnl > 0


def test_hold_while_position_open() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 101.0, "HOLD"),
            (101.0, 102.0, "HOLD"),
            (102.0, 103.0, "HOLD"),
            (103.0, 120.0, "HOLD"),
        ],
    )
    assert result.total_trades == 1
    assert result.trades[0].entry_price == pytest.approx(100.0)
    assert result.trades[0].exit_price == pytest.approx(120.0)


def test_trading_fees() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 110.0, "HOLD"),
            (110.0, 110.0, "HOLD"),
        ],
        fee_rate=0.001,
    )
    trade = result.trades[0]
    # notional≈999.001 after fee fit; check fees > 0 and net < gross price move
    assert trade.fees > 0
    assert trade.net_pnl < trade.gross_pnl
    assert result.final_equity < 1000.0 + trade.gross_pnl


def test_slippage() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 110.0, "HOLD"),
            (110.0, 110.0, "HOLD"),
        ],
        slippage_rate=0.0005,
    )
    trade = result.trades[0]
    # Buy high, sell low vs raw open/close references
    assert trade.entry_price == pytest.approx(100.0 * 1.0005)
    assert trade.exit_price == pytest.approx(110.0 * (1.0 - 0.0005))
    assert trade.net_pnl < 100.0


def test_no_future_data_leakage() -> None:
    # Bar0 close is huge, but execution must use bar1 open=100, not 999.
    result = _run(
        [
            (50.0, 999.0, "LONG"),
            (100.0, 100.0, "HOLD"),
            (100.0, 100.0, "HOLD"),
        ],
    )
    assert result.trades[0].entry_price == pytest.approx(100.0)


def test_final_candle_signal_not_executed() -> None:
    result = _run(
        [
            (100.0, 100.0, "HOLD"),
            (100.0, 100.0, "HOLD"),
            (100.0, 100.0, "LONG"),  # no N+1 → cannot open
        ],
    )
    assert result.total_trades == 0
    assert result.final_equity == pytest.approx(1000.0)


def test_final_open_position_force_closed() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 105.0, "HOLD"),
        ],
    )
    assert result.total_trades == 1
    assert result.trades[0].exit_timestamp == pd.Timestamp(_ts(1))


def test_insufficient_indicator_data_zero_trades() -> None:
    # Real baseline strategy + short history → HOLD during NaNs → no trades
    rows = []
    for i in range(40):
        rows.append(
            {
                "timestamp": _ts(i),
                "open": 100.0 + i * 0.1,
                "high": 101.0 + i * 0.1,
                "low": 99.0 + i * 0.1,
                "close": 100.0 + i * 0.1,
                "volume": 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    engine = BacktestEngine(
        strategy=BaselineStrategy(),
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0),
    )
    result = engine.run(frame, compute_indicators=True, compute_signals=True)
    assert result.total_trades == 0
    assert result.final_equity == pytest.approx(1000.0)


def test_empty_dataset_raises() -> None:
    engine = BacktestEngine()
    with pytest.raises(BacktestDataError, match="empty"):
        engine.run(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))


def test_invalid_timestamps_raise() -> None:
    frame = _frame([(100.0, 100.0, "HOLD"), (100.0, 100.0, "HOLD")])
    frame.loc[1, "timestamp"] = "not-a-date"
    engine = BacktestEngine()
    with pytest.raises(BacktestDataError, match="timestamp"):
        engine.run(frame, compute_indicators=False, compute_signals=False)


def test_duplicate_timestamps_raise() -> None:
    frame = _frame([(100.0, 100.0, "HOLD"), (100.0, 100.0, "HOLD")])
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    engine = BacktestEngine()
    with pytest.raises(BacktestDataError, match="Duplicate"):
        engine.run(frame, compute_indicators=False, compute_signals=False)


def test_zero_trade_backtest() -> None:
    result = _run(
        [
            (100.0, 100.0, "HOLD"),
            (100.0, 101.0, "HOLD"),
            (101.0, 102.0, "HOLD"),
        ],
    )
    assert result.total_trades == 0
    assert result.final_equity == pytest.approx(1000.0)
    assert result.total_return_pct == pytest.approx(0.0)
    assert len(result.equity_curve) == 3


def test_equity_curve_length() -> None:
    result = _run(
        [
            (100.0, 100.0, "LONG"),
            (100.0, 100.0, "HOLD"),
            (100.0, 100.0, "HOLD"),
        ],
    )
    assert len(result.equity_curve) == 3
    assert list(result.equity_curve_frame().columns) == ["timestamp", "equity"]


def test_load_ohlcv_csv(tmp_path: Path) -> None:
    path = tmp_path / "candles.csv"
    frame = pd.DataFrame(
        {
            "timestamp": [_ts(0), _ts(1)],
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10.0, 11.0],
        }
    )
    frame.to_csv(path, index=False)
    loaded = load_ohlcv_csv(path)
    assert list(loaded.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert len(loaded) == 2
