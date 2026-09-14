"""Backtesting package (Phase 4 — historical simulation only, no live orders)."""

from app.backtest.engine import (
    BacktestConfig,
    BacktestDataError,
    BacktestEngine,
    load_ohlcv_csv,
)
from app.backtest.metrics import compute_metrics
from app.backtest.models import (
    BacktestResult,
    EquityPoint,
    PositionSide,
    Trade,
)
from app.backtest.report import BacktestReport, build_report

__all__ = [
    "BacktestConfig",
    "BacktestDataError",
    "BacktestEngine",
    "BacktestReport",
    "BacktestResult",
    "EquityPoint",
    "PositionSide",
    "Trade",
    "build_report",
    "compute_metrics",
    "load_ohlcv_csv",
]
