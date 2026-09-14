"""
Run deterministic Phase 4 / Phase 5 backtest demos (no network, no orders).

Usage:
    python -m app.backtest
"""

from __future__ import annotations

import pandas as pd

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.risk import RiskConfig, RiskManager
from app.strategy.baseline import REQUIRED_INDICATORS


def _demo_frame() -> pd.DataFrame:
    """Small hand-built series with preset signals (no indicator warm-up needed)."""
    rows = [
        ("2024-01-01 00:00:00+00:00", 100.0, 101.0, 99.0, 100.0, 10.0, "LONG", 2.0),
        ("2024-01-01 00:01:00+00:00", 100.0, 106.0, 99.5, 105.0, 10.0, "HOLD", 2.0),
        ("2024-01-01 00:02:00+00:00", 105.0, 112.0, 104.0, 110.0, 10.0, "HOLD", 2.0),
        ("2024-01-01 00:03:00+00:00", 110.0, 111.0, 108.0, 109.0, 10.0, "SHORT", 2.0),
        ("2024-01-01 00:04:00+00:00", 109.0, 109.5, 100.0, 101.0, 10.0, "HOLD", 2.0),
        ("2024-01-01 00:05:00+00:00", 101.0, 102.0, 95.0, 96.0, 10.0, "HOLD", 2.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "signal",
            "atr_14",
        ],
    )
    for col in REQUIRED_INDICATORS:
        if col not in frame.columns:
            frame[col] = 1.0
    return frame


def _print_result(title: str, result) -> None:
    print(title)
    print(f"Initial capital : {result.initial_capital:.2f} USDT")
    print(f"Final equity    : {result.final_equity:.2f} USDT")
    print(f"Total return    : {result.total_return_pct:.4f}%")
    print(f"Total PnL       : {result.total_pnl:.4f}")
    print(f"Trades          : {result.total_trades}")
    print(f"Win rate        : {result.win_rate:.2f}%")
    print(f"Profit factor   : {result.profit_factor}")
    print(f"Max drawdown    : {result.max_drawdown_pct:.4f}%")
    print(f"Avg trade PnL   : {result.average_trade_pnl:.4f}")
    for trade in result.trades:
        print(
            f"  #{trade.trade_id} {trade.side.value} "
            f"entry={trade.entry_price:.4f} exit={trade.exit_price:.4f} "
            f"qty={trade.quantity:.4f} net={trade.net_pnl:.4f}"
        )


def main() -> None:
    frame = _demo_frame()

    legacy = BacktestEngine(config=BacktestConfig()).run(
        frame,
        compute_indicators=False,
        compute_signals=False,
    )
    _print_result(
        "Phase 4 synthetic backtest demo (full capital, no risk manager)",
        legacy,
    )
    print()

    risked = BacktestEngine(
        config=BacktestConfig(),
        risk_manager=RiskManager(RiskConfig()),
    ).run(
        frame,
        compute_indicators=False,
        compute_signals=False,
    )
    _print_result(
        "Phase 5 synthetic backtest demo (risk sizing + ATR SL/TP) - not live trading",
        risked,
    )


if __name__ == "__main__":
    main()
