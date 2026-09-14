"""Extended backtest analytics (Phase 7 — robust evaluation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import pandas as pd

from app.backtest.metrics import profit_factor
from app.backtest.models import BacktestResult, EquityPoint, PositionSide, Trade


@dataclass(slots=True)
class SideBreakdown:
    side: str
    trade_count: int
    win_rate: float
    total_pnl: float
    winning_trades: int
    losing_trades: int


@dataclass(slots=True)
class MonthlyReturn:
    month: str
    equity_start: float
    equity_end: float
    return_pct: float
    trades: int


@dataclass(slots=True)
class RobustMetrics:
    """Period-level metrics beyond the Phase 6 summary."""

    total_trades: int
    total_return_pct: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    average_winning_trade: float
    average_losing_trade: float
    expectancy_per_trade: float
    max_drawdown_pct: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_fees: float
    time_in_market_pct: float
    long: SideBreakdown
    short: SideBreakdown
    monthly_returns: list[MonthlyReturn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def max_consecutive_streak(trades: Sequence[Trade], *, wins: bool) -> int:
    best = 0
    current = 0
    for trade in trades:
        is_win = trade.net_pnl > 0
        is_loss = trade.net_pnl < 0
        matched = is_win if wins else is_loss
        if matched:
            current += 1
            best = max(best, current)
        else:
            # Break-even neither extends nor resets a streak of the opposite type
            # in a strict sense — reset when the desired outcome does not continue.
            if (wins and not is_win) or ((not wins) and not is_loss):
                current = 0
    return best


def side_breakdown(trades: Sequence[Trade], side: PositionSide) -> SideBreakdown:
    subset = [t for t in trades if t.side is side]
    wins = [t for t in subset if t.net_pnl > 0]
    losses = [t for t in subset if t.net_pnl < 0]
    count = len(subset)
    win_rate = (len(wins) / count * 100.0) if count else 0.0
    return SideBreakdown(
        side=side.value,
        trade_count=count,
        win_rate=win_rate,
        total_pnl=sum(t.net_pnl for t in subset),
        winning_trades=len(wins),
        losing_trades=len(losses),
    )


def time_in_market_pct(
    trades: Sequence[Trade],
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> float:
    """Fraction of the evaluation window spent in a position (single-position model)."""
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    total = (end - start).total_seconds()
    if total <= 0:
        return 0.0
    held = 0.0
    for trade in trades:
        entry = pd.Timestamp(trade.entry_timestamp)
        exit_ = pd.Timestamp(trade.exit_timestamp)
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        if exit_.tzinfo is None:
            exit_ = exit_.tz_localize("UTC")
        # Clip to evaluation window
        a = max(entry, start)
        b = min(exit_, end)
        if b > a:
            held += (b - a).total_seconds()
    return min(100.0, held / total * 100.0)


def monthly_returns_from_equity(
    equity_curve: Sequence[EquityPoint],
    trades: Sequence[Trade] | None = None,
) -> list[MonthlyReturn]:
    if not equity_curve:
        return []

    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(p.timestamp) for p in equity_curve],
            "equity": [p.equity for p in equity_curve],
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.set_index("timestamp")

    # Month-end last equity; prepend first point as starting equity of first month.
    month_end = frame["equity"].resample("ME").last().dropna()
    if month_end.empty:
        return []

    trade_counts: dict[str, int] = {}
    if trades:
        for trade in trades:
            key = pd.Timestamp(trade.exit_timestamp)
            if key.tzinfo is None:
                key = key.tz_localize("UTC")
            else:
                key = key.tz_convert("UTC")
            label = key.strftime("%Y-%m")
            trade_counts[label] = trade_counts.get(label, 0) + 1

    rows: list[MonthlyReturn] = []
    prev_equity = float(frame["equity"].iloc[0])
    for ts, equity in month_end.items():
        label = pd.Timestamp(ts).strftime("%Y-%m")
        end_eq = float(equity)
        ret = ((end_eq / prev_equity) - 1.0) * 100.0 if prev_equity else 0.0
        rows.append(
            MonthlyReturn(
                month=label,
                equity_start=prev_equity,
                equity_end=end_eq,
                return_pct=ret,
                trades=trade_counts.get(label, 0),
            )
        )
        prev_equity = end_eq
    return rows


def compute_robust_metrics(
    result: BacktestResult,
    *,
    period_start: pd.Timestamp | str,
    period_end: pd.Timestamp | str,
) -> RobustMetrics:
    trades = result.trades
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0
    n = len(trades)
    win_rate_frac = (len(wins) / n) if n else 0.0
    loss_rate_frac = (len(losses) / n) if n else 0.0
    expectancy = win_rate_frac * avg_win + loss_rate_frac * avg_loss

    return RobustMetrics(
        total_trades=n,
        total_return_pct=result.total_return_pct,
        total_pnl=result.total_pnl,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        average_winning_trade=avg_win,
        average_losing_trade=avg_loss,
        expectancy_per_trade=expectancy,
        max_drawdown_pct=result.max_drawdown_pct,
        max_consecutive_wins=max_consecutive_streak(trades, wins=True),
        max_consecutive_losses=max_consecutive_streak(trades, wins=False),
        total_fees=sum(t.fees for t in trades),
        time_in_market_pct=time_in_market_pct(
            trades,
            period_start=pd.Timestamp(period_start),
            period_end=pd.Timestamp(period_end),
        ),
        long=side_breakdown(trades, PositionSide.LONG),
        short=side_breakdown(trades, PositionSide.SHORT),
        monthly_returns=monthly_returns_from_equity(result.equity_curve, trades),
    )


def format_robust_section(title: str, metrics: RobustMetrics) -> str:
    pf = (
        "inf"
        if metrics.profit_factor == float("inf")
        else f"{metrics.profit_factor:.4f}"
    )
    lines = [
        f"--- {title} ---",
        f"Total return           : {metrics.total_return_pct:.4f}%",
        f"Total PnL              : {metrics.total_pnl:.4f}",
        f"Total trades           : {metrics.total_trades}",
        f"Win rate               : {metrics.win_rate:.2f}%",
        f"Profit factor          : {pf}",
        f"Avg winning trade      : {metrics.average_winning_trade:.4f}",
        f"Avg losing trade       : {metrics.average_losing_trade:.4f}",
        f"Expectancy / trade     : {metrics.expectancy_per_trade:.4f}",
        f"Max drawdown           : {metrics.max_drawdown_pct:.4f}%",
        f"Max consecutive wins   : {metrics.max_consecutive_wins}",
        f"Max consecutive losses : {metrics.max_consecutive_losses}",
        f"Total fees             : {metrics.total_fees:.4f}",
        f"Time in market         : {metrics.time_in_market_pct:.2f}%",
        (
            f"LONG  trades/win%/PnL  : {metrics.long.trade_count} / "
            f"{metrics.long.win_rate:.2f}% / {metrics.long.total_pnl:.4f}"
        ),
        (
            f"SHORT trades/win%/PnL  : {metrics.short.trade_count} / "
            f"{metrics.short.win_rate:.2f}% / {metrics.short.total_pnl:.4f}"
        ),
    ]
    if metrics.monthly_returns:
        lines.append("Monthly returns:")
        for row in metrics.monthly_returns:
            lines.append(
                f"  {row.month}: {row.return_pct:+.4f}% "
                f"(trades={row.trades}, equity {row.equity_start:.2f} -> {row.equity_end:.2f})"
            )
    return "\n".join(lines)


# Re-export for tests that may want PF helper consistency
__all__ = [
    "MonthlyReturn",
    "RobustMetrics",
    "SideBreakdown",
    "compute_robust_metrics",
    "format_robust_section",
    "max_consecutive_streak",
    "monthly_returns_from_equity",
    "profit_factor",
    "side_breakdown",
    "time_in_market_pct",
]
