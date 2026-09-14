"""Performance metrics derived from closed trades and the equity curve."""

from __future__ import annotations

from typing import Sequence

from app.backtest.models import BacktestResult, EquityPoint, Trade


def max_drawdown_pct(equity_curve: Sequence[EquityPoint]) -> float:
    """
    Maximum peak-to-trough percentage decline of the equity curve.

    Returns 0.0 when the curve is empty or never declines from a peak.
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0].equity
    max_dd = 0.0
    for point in equity_curve:
        equity = point.equity
        if equity > peak:
            peak = equity
        if peak <= 0:
            continue
        drawdown = (peak - equity) / peak * 100.0
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    """
    Gross profit / abs(gross loss).

    * No losing trades and some winners → +inf
    * No trades / both zero → 0.0
    """
    loss_abs = abs(gross_loss)
    if loss_abs == 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return gross_profit / loss_abs


def compute_metrics(
    *,
    initial_capital: float,
    final_equity: float,
    trades: Sequence[Trade],
    equity_curve: Sequence[EquityPoint],
) -> BacktestResult:
    """Build a BacktestResult from simulation outputs."""
    total_pnl = final_equity - initial_capital
    total_return_pct = (
        (final_equity / initial_capital - 1.0) * 100.0 if initial_capital else 0.0
    )

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]

    total_trades = len(trades)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = sum(t.net_pnl for t in losses)

    pnls = [t.net_pnl for t in trades]
    average_trade_pnl = sum(pnls) / total_trades if total_trades else 0.0
    best_trade_pnl = max(pnls) if pnls else None
    worst_trade_pnl = min(pnls) if pnls else None

    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        total_pnl=total_pnl,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor(gross_profit, gross_loss),
        max_drawdown_pct=max_drawdown_pct(equity_curve),
        average_trade_pnl=average_trade_pnl,
        best_trade_pnl=best_trade_pnl,
        worst_trade_pnl=worst_trade_pnl,
        trades=list(trades),
        equity_curve=list(equity_curve),
    )
