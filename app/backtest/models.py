"""Data models for the Phase 4 backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class PositionSide(str, Enum):
    """Open position direction (simulator state also uses FLAT separately)."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(slots=True)
class Trade:
    """One completed round-trip (entry → exit)."""

    trade_id: int
    side: PositionSide
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    funding: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "side": self.side.value,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "gross_pnl": self.gross_pnl,
            "fees": self.fees,
            "funding": self.funding,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
        }


@dataclass(slots=True)
class EquityPoint:
    """Equity snapshot at a point in time (no future information)."""

    timestamp: pd.Timestamp
    equity: float


@dataclass(slots=True)
class BacktestResult:
    """Aggregated outcome of a historical simulation."""

    initial_capital: float
    final_equity: float
    total_return_pct: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    max_drawdown_pct: float
    average_trade_pnl: float
    best_trade_pnl: float | None
    worst_trade_pnl: float | None
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    def equity_curve_frame(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])
        return pd.DataFrame(
            {
                "timestamp": [p.timestamp for p in self.equity_curve],
                "equity": [p.equity for p in self.equity_curve],
            }
        )
