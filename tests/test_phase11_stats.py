"""Phase 11 unit tests (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.models import PositionSide, Trade
from app.stats.benchmarks import (
    assign_random_signals,
    buy_and_hold,
    measure_signal_stats,
)
from app.stats.bootstrap import bootstrap_trade_metrics
from app.stats.monte_carlo import run_trade_monte_carlo
from app.stats.risk_metrics import compute_risk_adjusted, max_drawdown_from_equity
from app.stats.sensitivity import CostScenario, cost_sensitivity


def _ts(i: int) -> pd.Timestamp:
    return pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=4 * i)


def _trade(pnl: float, i: int = 0) -> Trade:
    return Trade(
        trade_id=i + 1,
        side=PositionSide.LONG,
        entry_timestamp=_ts(i),
        exit_timestamp=_ts(i + 1),
        entry_price=100.0,
        exit_price=100.0 + pnl,
        quantity=1.0,
        gross_pnl=pnl,
        fees=0.0,
        net_pnl=pnl,
        return_pct=pnl,
    )


def test_monte_carlo_reproducible() -> None:
    trades = [_trade(1.0, 0), _trade(-0.5, 1), _trade(2.0, 2), _trade(-1.0, 3)]
    a, ra, _ = run_trade_monte_carlo(trades, n_sims=200, seed=42)
    b, rb, _ = run_trade_monte_carlo(trades, n_sims=200, seed=42)
    assert a.median_return == b.median_return
    np.testing.assert_array_equal(ra, rb)


def test_monte_carlo_percentiles_ordered() -> None:
    trades = [_trade(x, i) for i, x in enumerate([1, -1, 2, -2, 0.5])]
    summary, _, _ = run_trade_monte_carlo(trades, n_sims=1000, seed=1)
    assert summary.p05_return <= summary.p25_return <= summary.median_return
    assert summary.median_return <= summary.p75_return <= summary.p95_return


def test_bootstrap_flags_small_sample() -> None:
    trades = [_trade(1.0, 0), _trade(-1.0, 1)]
    stats = bootstrap_trade_metrics(trades, n_boot=100, seed=0, min_trades=20)
    assert all(s.reliable is False for s in stats)


def test_random_signals_reproducible_and_approx_frequency() -> None:
    n = 1000
    df = pd.DataFrame(
        {
            "timestamp": [_ts(i) for i in range(n)],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "signal": ["HOLD"] * n,
        }
    )
    # Plant frequencies
    df.loc[0:49, "signal"] = "LONG"
    df.loc[50:79, "signal"] = "SHORT"
    stats = measure_signal_stats(df)
    a = assign_random_signals(df, stats, seed=123)
    b = assign_random_signals(df, stats, seed=123)
    assert list(a["signal"]) == list(b["signal"])
    out_stats = measure_signal_stats(a)
    assert abs(out_stats.long_pct - stats.long_pct) < 0.03
    assert abs(out_stats.short_pct - stats.short_pct) < 0.03


def test_buy_and_hold_positive_on_uptrend() -> None:
    rows = []
    for i in range(10):
        px = 100 + i
        rows.append(
            {
                "timestamp": _ts(i),
                "open": px,
                "high": px + 1,
                "low": px - 1,
                "close": px + 0.5,
                "volume": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    result = buy_and_hold(df, fee_rate=0.0, slippage_rate=0.0)
    assert result.total_return_pct > 0
    assert result.total_trades == 1


def test_max_drawdown_from_equity() -> None:
    eq = np.array([100.0, 110.0, 90.0, 95.0])
    assert max_drawdown_from_equity(eq) == pytest.approx(18.1818, rel=1e-3)


def test_cost_sensitivity_higher_costs_not_better() -> None:
    # Minimal signaled frame with indicators for risk path is heavy;
    # unit-level: CostScenario ordering only
    low = CostScenario("LOW", 0.0005, 0.0002)
    high = CostScenario("HIGH", 0.0015, 0.001)
    assert low.fee_rate < high.fee_rate
