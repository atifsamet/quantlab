"""Risk-adjusted and distributional metrics for Phase 11."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from app.backtest.models import BacktestResult, EquityPoint, Trade

# Bars per year for annualization by timeframe label.
BARS_PER_YEAR: dict[str, float] = {
    "15m": 365.25 * 24 * 4,
    "1h": 365.25 * 24,
    "4h": 365.25 * 6,
    "1d": 365.25,
}


@dataclass(slots=True)
class RiskAdjustedMetrics:
    """Documented risk metrics; annualized only when sample is adequate."""

    total_return_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    recovery_factor: float | None
    profit_factor: float | None
    expectancy: float
    average_win: float
    average_loss: float
    payoff_ratio: float | None
    win_rate: float
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    annualized: bool
    annualization_note: str
    n_trades: int
    n_bars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def equity_returns(equity_curve: Sequence[EquityPoint]) -> np.ndarray:
    if len(equity_curve) < 2:
        return np.array([], dtype=float)
    eq = np.array([p.equity for p in equity_curve], dtype=float)
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, eq[1:] / prev - 1.0, 0.0)
    return rets.astype(float)


def max_drawdown_duration_bars(equity_curve: Sequence[EquityPoint]) -> int:
    if not equity_curve:
        return 0
    peak = equity_curve[0].equity
    peak_i = 0
    worst = 0
    for i, point in enumerate(equity_curve):
        if point.equity >= peak:
            peak = point.equity
            peak_i = i
        else:
            worst = max(worst, i - peak_i)
    return int(worst)


def compute_risk_adjusted(
    result: BacktestResult,
    *,
    timeframe: str = "4h",
    min_trades_for_annualization: int = 20,
    min_bars_for_annualization: int = 100,
    risk_free_per_bar: float = 0.0,
) -> RiskAdjustedMetrics:
    """
    Sharpe/Sortino/Calmar annualization assumptions
    ----------------------------------------------
    * Per-bar returns from the equity curve.
    * Annualization factor = sqrt(bars_per_year) for Sharpe/Sortino.
    * Calmar = annualized return / max drawdown (when annualized).
    * If trade count or bar count is too small, Sharpe/Sortino/Calmar are None
      and ``annualized=False`` — we do not fabricate annualized stats.
    """
    trades = result.trades
    n_trades = len(trades)
    n_bars = len(result.equity_curve)
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
    expectancy = float(np.mean([t.net_pnl for t in trades])) if trades else 0.0
    win_rate = 100.0 * len(wins) / n_trades if n_trades else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss == 0:
        pf = float("inf") if gross_profit > 0 else None
    else:
        pf = gross_profit / gross_loss

    max_dd = float(result.max_drawdown_pct)
    recovery = (
        (result.total_pnl / (abs(max_dd) / 100.0 * result.initial_capital))
        if max_dd > 0 and result.initial_capital > 0
        else None
    )
    dd_dur = max_drawdown_duration_bars(result.equity_curve)

    bars_py = BARS_PER_YEAR.get(timeframe)
    can_annualize = (
        bars_py is not None
        and n_trades >= min_trades_for_annualization
        and n_bars >= min_bars_for_annualization
    )
    sharpe = sortino = calmar = None
    note = "Insufficient sample for reliable annualization."
    if can_annualize and bars_py is not None:
        rets = equity_returns(result.equity_curve)
        note = (
            f"Annualized with bars_per_year={bars_py} for timeframe={timeframe}; "
            f"risk_free_per_bar={risk_free_per_bar}."
        )
        if len(rets) > 1 and np.std(rets, ddof=1) > 0:
            excess = rets - risk_free_per_bar
            sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(bars_py))
            downside = excess[excess < 0]
            if len(downside) > 1 and np.std(downside, ddof=1) > 0:
                sortino = float(
                    np.mean(excess) / np.std(downside, ddof=1) * np.sqrt(bars_py)
                )
            ann_ret = float((1.0 + result.total_return_pct / 100.0) ** (bars_py / max(n_bars, 1)) - 1.0)
            if max_dd > 0:
                calmar = float((ann_ret * 100.0) / max_dd)

    return RiskAdjustedMetrics(
        total_return_pct=float(result.total_return_pct),
        max_drawdown_pct=max_dd,
        max_drawdown_duration_bars=dd_dur,
        recovery_factor=recovery,
        profit_factor=pf,
        expectancy=expectancy,
        average_win=avg_win,
        average_loss=avg_loss,
        payoff_ratio=payoff,
        win_rate=win_rate,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        annualized=can_annualize,
        annualization_note=note,
        n_trades=n_trades,
        n_bars=n_bars,
    )


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak * 100.0)
    return float(max_dd)
