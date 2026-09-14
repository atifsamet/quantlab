"""Phase 7 robust backtest / train-test / walk-forward tests (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.analysis import (
    compute_robust_metrics,
    max_consecutive_streak,
    monthly_returns_from_equity,
    side_breakdown,
)
from app.backtest.models import BacktestResult, EquityPoint, PositionSide, Trade
from app.backtest.robust import (
    assert_warmup_available,
    generate_walk_forward_windows,
    load_signaled_history,
    run_period,
    slice_evaluation,
)
from app.indicators import WARMUP_BARS
from app.strategy.baseline import REQUIRED_INDICATORS


def _trade(
    net: float,
    side: PositionSide = PositionSide.LONG,
    trade_id: int = 1,
    entry: str = "2024-01-01T00:00:00Z",
    exit_: str = "2024-01-02T00:00:00Z",
) -> Trade:
    return Trade(
        trade_id=trade_id,
        side=side,
        entry_timestamp=pd.Timestamp(entry),
        exit_timestamp=pd.Timestamp(exit_),
        entry_price=100.0,
        exit_price=110.0,
        quantity=1.0,
        gross_pnl=net,
        fees=0.1,
        net_pnl=net,
        return_pct=net,
    )


def _synthetic_ohlcv_csv(path: Path, n: int = 400) -> Path:
    """Create enough bars for EMA200 warm-up + evaluation."""
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for i in range(n):
        ts = start + pd.Timedelta(hours=i)
        # Mild deterministic drift + oscillation for mixed signals
        price = 100.0 + 0.05 * i + 2.0 * np.sin(i / 7.0)
        rows.append(
            {
                "timestamp": ts.isoformat().replace("+00:00", "+00:00"),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.2 * np.cos(i / 5.0),
                "volume": 10.0 + (i % 5),
            }
        )
    frame = pd.DataFrame(rows)
    # Normalize timestamp strings for CSV
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_max_consecutive_streaks() -> None:
    trades = [
        _trade(1, trade_id=1),
        _trade(2, trade_id=2),
        _trade(-1, trade_id=3),
        _trade(-2, trade_id=4),
        _trade(-3, trade_id=5),
        _trade(1, trade_id=6),
    ]
    assert max_consecutive_streak(trades, wins=True) == 2
    assert max_consecutive_streak(trades, wins=False) == 3


def test_long_short_metrics() -> None:
    trades = [
        _trade(10, PositionSide.LONG, 1),
        _trade(-4, PositionSide.LONG, 2),
        _trade(5, PositionSide.SHORT, 3),
        _trade(-2, PositionSide.SHORT, 4),
        _trade(-1, PositionSide.SHORT, 5),
    ]
    long = side_breakdown(trades, PositionSide.LONG)
    short = side_breakdown(trades, PositionSide.SHORT)
    assert long.trade_count == 2
    assert long.win_rate == pytest.approx(50.0)
    assert long.total_pnl == pytest.approx(6.0)
    assert short.trade_count == 3
    assert short.winning_trades == 1
    assert short.losing_trades == 2


def test_monthly_aggregation() -> None:
    curve = [
        EquityPoint(pd.Timestamp("2024-01-01", tz="UTC"), 1000.0),
        EquityPoint(pd.Timestamp("2024-01-15", tz="UTC"), 1010.0),
        EquityPoint(pd.Timestamp("2024-01-31", tz="UTC"), 1100.0),
        EquityPoint(pd.Timestamp("2024-02-10", tz="UTC"), 1050.0),
        EquityPoint(pd.Timestamp("2024-02-29", tz="UTC"), 1080.0),
    ]
    months = monthly_returns_from_equity(curve)
    assert [m.month for m in months] == ["2024-01", "2024-02"]
    assert months[0].return_pct == pytest.approx(10.0)
    assert months[1].equity_start == pytest.approx(1100.0)
    assert months[1].return_pct == pytest.approx((1080 / 1100 - 1) * 100)


def test_walk_forward_windows() -> None:
    windows = generate_walk_forward_windows(
        overall_start="2024-01-01",
        overall_end="2024-12-31",
        train_days=90,
        test_days=30,
        step_days=30,
    )
    assert len(windows) >= 3
    for tr_s, tr_e, te_s, te_e in windows:
        assert tr_e == te_s
        assert (tr_e - tr_s).days == 90
        assert (te_e - te_s).days == 30
        assert te_e <= pd.Timestamp("2024-12-31", tz="UTC")


def test_warmup_and_no_future_leakage(tmp_path: Path) -> None:
    csv_path = _synthetic_ohlcv_csv(tmp_path / "BTC-USDT_1h.csv", n=500)
    history = load_signaled_history(csv_path)

    eval_start = history["timestamp"].iloc[WARMUP_BARS]
    eval_end = history["timestamp"].iloc[WARMUP_BARS + 50]

    # Warm-up must be present
    assert_warmup_available(history, eval_start, warmup_bars=WARMUP_BARS)

    # Insufficient warm-up raises
    too_early = history["timestamp"].iloc[10]
    with pytest.raises(ValueError, match="Insufficient warm-up"):
        assert_warmup_available(history, too_early, warmup_bars=WARMUP_BARS)

    frame = slice_evaluation(history, eval_start, eval_end)
    assert frame["timestamp"].min() >= eval_start
    assert frame["timestamp"].max() <= eval_end
    # No future relative to eval_end
    assert not (frame["timestamp"] > eval_end).any()
    # Indicators ready at start
    for col in REQUIRED_INDICATORS:
        assert not pd.isna(frame.iloc[0][col])

    # Indicators on eval start equal full-history values (no recompute leakage)
    full_row = history.loc[history["timestamp"] == eval_start].iloc[0]
    for col in ("ema_200", "rsi_14", "macd"):
        assert frame.iloc[0][col] == pytest.approx(full_row[col])


def test_train_test_separate_and_no_cross_leakage(tmp_path: Path) -> None:
    csv_path = _synthetic_ohlcv_csv(tmp_path / "BTC-USDT_1h.csv", n=600)
    history = load_signaled_history(csv_path)

    train_start = history["timestamp"].iloc[WARMUP_BARS]
    train_end = history["timestamp"].iloc[WARMUP_BARS + 100]
    test_start = history["timestamp"].iloc[WARMUP_BARS + 101]
    test_end = history["timestamp"].iloc[WARMUP_BARS + 200]

    train = run_period(
        history,
        label="train",
        start=train_start,
        end=train_end,
        instrument="BTC-USDT",
        timeframe="1h",
        use_risk=True,
    )
    test = run_period(
        history,
        label="test",
        start=test_start,
        end=test_end,
        instrument="BTC-USDT",
        timeframe="1h",
        use_risk=True,
    )

    assert train.frame["timestamp"].max() <= train_end
    assert test.frame["timestamp"].min() >= test_start
    # Disjoint evaluation frames
    assert train.frame["timestamp"].max() < test.frame["timestamp"].min()
    # Test equity curve confined to test window
    if test.result.equity_curve:
        assert test.result.equity_curve[0].timestamp >= test_start
        assert test.result.equity_curve[-1].timestamp <= test_end


def test_robust_metrics_expectancy() -> None:
    trades = [_trade(10, trade_id=1), _trade(-5, trade_id=2), _trade(0, trade_id=3)]
    result = BacktestResult(
        initial_capital=1000.0,
        final_equity=1005.0,
        total_return_pct=0.5,
        total_pnl=5.0,
        total_trades=3,
        winning_trades=1,
        losing_trades=1,
        win_rate=1 / 3 * 100,
        gross_profit=10.0,
        gross_loss=-5.0,
        profit_factor=2.0,
        max_drawdown_pct=1.0,
        average_trade_pnl=5.0 / 3,
        best_trade_pnl=10.0,
        worst_trade_pnl=-5.0,
        trades=trades,
        equity_curve=[
            EquityPoint(pd.Timestamp("2024-01-01", tz="UTC"), 1000.0),
            EquityPoint(pd.Timestamp("2024-01-03", tz="UTC"), 1005.0),
        ],
    )
    metrics = compute_robust_metrics(
        result,
        period_start="2024-01-01",
        period_end="2024-01-31",
    )
    assert metrics.average_winning_trade == pytest.approx(10.0)
    assert metrics.average_losing_trade == pytest.approx(-5.0)
    # expectancy = (1/3)*10 + (1/3)*(-5) + 0 for flat in rates used
    assert metrics.expectancy_per_trade == pytest.approx((1 / 3) * 10 + (1 / 3) * (-5))
    assert metrics.total_fees == pytest.approx(0.3)
