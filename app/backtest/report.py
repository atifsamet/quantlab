"""Backtest performance reporting and CSV exports (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.models import BacktestResult, PositionSide


@dataclass(slots=True)
class BacktestReport:
    """Human-readable / exportable summary of a historical backtest."""

    instrument: str
    timeframe: str
    evaluation_start: str
    evaluation_end: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    average_trade: float
    best_trade: float | None
    worst_trade: float | None
    total_fees: float
    long_trades: int
    short_trades: int
    warmup_bars: int
    candle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "average_trade": self.average_trade,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "total_fees": self.total_fees,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "warmup_bars": self.warmup_bars,
            "candle_count": self.candle_count,
        }

    def format_text(self) -> str:
        pf = self.profit_factor
        pf_text = "inf" if pf == float("inf") else f"{pf:.4f}"
        best = "n/a" if self.best_trade is None else f"{self.best_trade:.4f}"
        worst = "n/a" if self.worst_trade is None else f"{self.worst_trade:.4f}"
        lines = [
            "=== Backtest report ===",
            f"Instrument          : {self.instrument}",
            f"Timeframe           : {self.timeframe}",
            f"Evaluation start    : {self.evaluation_start}",
            f"Evaluation end      : {self.evaluation_end}",
            f"Warm-up bars        : {self.warmup_bars}",
            f"Candles (eval)      : {self.candle_count}",
            f"Initial capital     : {self.initial_capital:.2f} USDT",
            f"Final equity        : {self.final_equity:.2f} USDT",
            f"Total return        : {self.total_return_pct:.4f}%",
            f"Total PnL           : {self.total_pnl:.4f}",
            f"Total trades        : {self.total_trades}",
            f"Winning trades      : {self.winning_trades}",
            f"Losing trades       : {self.losing_trades}",
            f"LONG trades         : {self.long_trades}",
            f"SHORT trades        : {self.short_trades}",
            f"Win rate            : {self.win_rate:.2f}%",
            f"Profit factor       : {pf_text}",
            f"Max drawdown        : {self.max_drawdown_pct:.4f}%",
            f"Average trade       : {self.average_trade:.4f}",
            f"Best trade          : {best}",
            f"Worst trade         : {worst}",
            f"Total fees          : {self.total_fees:.4f}",
            "",
            "Historical backtest results do not guarantee future performance.",
            "Fees/slippage assumptions affect results. No live trading enabled.",
        ]
        return "\n".join(lines)


def build_report(
    result: BacktestResult,
    *,
    instrument: str,
    timeframe: str,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    warmup_bars: int,
    candle_count: int,
) -> BacktestReport:
    long_trades = sum(1 for t in result.trades if t.side is PositionSide.LONG)
    short_trades = sum(1 for t in result.trades if t.side is PositionSide.SHORT)
    total_fees = sum(t.fees for t in result.trades)
    return BacktestReport(
        instrument=instrument,
        timeframe=timeframe,
        evaluation_start=str(evaluation_start),
        evaluation_end=str(evaluation_end),
        initial_capital=result.initial_capital,
        final_equity=result.final_equity,
        total_return_pct=result.total_return_pct,
        total_pnl=result.total_pnl,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        max_drawdown_pct=result.max_drawdown_pct,
        average_trade=result.average_trade_pnl,
        best_trade=result.best_trade_pnl,
        worst_trade=result.worst_trade_pnl,
        total_fees=total_fees,
        long_trades=long_trades,
        short_trades=short_trades,
        warmup_bars=warmup_bars,
        candle_count=candle_count,
    )


def export_equity_curve(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = result.equity_curve_frame()
    if not frame.empty:
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        frame["timestamp"] = frame["timestamp"].str.replace(
            r"(\d{2})(\d{2})$", r"\1:\2", regex=True
        )
    frame.to_csv(path, index=False)
    return path


def export_trades(result: BacktestResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not result.trades:
        pd.DataFrame(
            columns=[
                "trade_id",
                "side",
                "entry_timestamp",
                "exit_timestamp",
                "entry_price",
                "exit_price",
                "quantity",
                "gross_pnl",
                "fees",
                "net_pnl",
                "return_pct",
            ]
        ).to_csv(path, index=False)
        return path

    rows = [t.to_dict() for t in result.trades]
    frame = pd.DataFrame(rows)
    for col in ("entry_timestamp", "exit_timestamp"):
        frame[col] = pd.to_datetime(frame[col], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        frame[col] = frame[col].str.replace(r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    frame.to_csv(path, index=False)
    return path


def default_results_paths(instrument: str, timeframe: str, directory: str | Path = "data/results") -> tuple[Path, Path]:
    safe = instrument.replace("/", "-").upper()
    base = Path(directory)
    return (
        base / f"{safe}_{timeframe}_equity.csv",
        base / f"{safe}_{timeframe}_trades.csv",
    )
