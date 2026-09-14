"""Benchmark strategies for Phase 11 statistical comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.backtest.models import BacktestResult, EquityPoint, PositionSide, Trade
from app.indicators import WARMUP_BARS
from app.research.indicators_flex import RESEARCH_REQUIRED_COLUMNS, add_research_indicators
from app.research.params import ResearchParams
from app.research.variants import build_strategy
from app.risk import RiskConfig, RiskManager
from app.stats import RANDOM_SEED


@dataclass(slots=True)
class SignalStats:
    long_pct: float
    short_pct: float
    hold_pct: float
    n_bars: int
    n_long: int
    n_short: int


def measure_signal_stats(signaled: pd.DataFrame) -> SignalStats:
    sig = signaled["signal"].astype(str).str.upper()
    n = len(sig)
    n_long = int((sig == "LONG").sum())
    n_short = int((sig == "SHORT").sum())
    n_hold = n - n_long - n_short
    return SignalStats(
        long_pct=n_long / n if n else 0.0,
        short_pct=n_short / n if n else 0.0,
        hold_pct=n_hold / n if n else 1.0,
        n_bars=n,
        n_long=n_long,
        n_short=n_short,
    )


def prepare_breakout_frame(ohlcv: pd.DataFrame, params: ResearchParams | None = None) -> pd.DataFrame:
    params = params or ResearchParams(variant="breakout", breakout_lookback=20)
    framed = add_research_indicators(ohlcv, params)
    return build_strategy(params).generate_signals(framed)


def assign_constant_signal(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = signal
    out["signal_reason"] = f"benchmark_{signal.lower()}"
    return out


def assign_random_signals(
    df: pd.DataFrame,
    stats: SignalStats,
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Independent random LONG/SHORT/HOLD with matched frequencies.

    No look-ahead; each bar drawn i.i.d. from the empirical mix.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    probs = np.array([stats.long_pct, stats.short_pct, stats.hold_pct], dtype=float)
    if probs.sum() <= 0:
        probs = np.array([0.0, 0.0, 1.0])
    else:
        probs = probs / probs.sum()
    choices = rng.choice(["LONG", "SHORT", "HOLD"], size=len(out), p=probs)
    out["signal"] = choices
    out["signal_reason"] = "random_matched_frequency"
    return out


def assign_random_entry_matched_count(
    df: pd.DataFrame,
    *,
    target_trade_signals: int,
    long_share: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sparse random entries aiming for ~target_trade_signals non-HOLD bars.

    Approximates matched trade *activity* without copying strategy logic.
    """
    out = df.copy()
    n = len(out)
    out["signal"] = "HOLD"
    out["signal_reason"] = "random_matched_count"
    if n < 2 or target_trade_signals <= 0:
        return out
    rng = np.random.default_rng(seed + 7)
    k = min(target_trade_signals, n - 1)
    idxs = rng.choice(np.arange(0, n - 1), size=k, replace=False)
    long_share = float(np.clip(long_share, 0.0, 1.0))
    for i in idxs:
        out.iat[i, out.columns.get_loc("signal")] = (
            "LONG" if rng.random() < long_share else "SHORT"
        )
    return out


def run_signaled_backtest(
    signaled: pd.DataFrame,
    *,
    use_risk: bool = True,
    config: BacktestConfig | None = None,
    risk_config: RiskConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    risk = RiskManager(risk_config or RiskConfig()) if use_risk else None
    engine = BacktestEngine(config=cfg, risk_manager=risk)
    return engine.run(signaled, compute_indicators=False, compute_signals=False)


def buy_and_hold(
    df: pd.DataFrame,
    *,
    initial_capital: float = 1000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> BacktestResult:
    """
    Buy at first evaluation open, hold to last close (single round-trip).

    Fair spot benchmark; 1x, no leverage. Fees on entry and exit notionals.
    """
    if len(df) < 2:
        raise ValueError("Need at least 2 bars for buy-and-hold")
    entry_open = float(df.iloc[0]["open"])
    exit_close = float(df.iloc[-1]["close"])
    entry_fill = entry_open * (1.0 + slippage_rate)
    exit_fill = exit_close * (1.0 - slippage_rate)
    affordable = initial_capital / (1.0 + fee_rate)
    qty = affordable / entry_fill
    entry_fee = qty * entry_fill * fee_rate
    exit_fee = qty * exit_fill * fee_rate
    gross = qty * (exit_fill - entry_fill)
    fees = entry_fee + exit_fee
    net = gross - fees
    final_equity = max(0.0, initial_capital + net)

    trade = Trade(
        trade_id=1,
        side=PositionSide.LONG,
        entry_timestamp=pd.Timestamp(df.iloc[0]["timestamp"]),
        exit_timestamp=pd.Timestamp(df.iloc[-1]["timestamp"]),
        entry_price=entry_fill,
        exit_price=exit_fill,
        quantity=qty,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        return_pct=net / initial_capital * 100.0,
    )
    equity_curve: list[EquityPoint] = []
    for pos in range(len(df)):
        row = df.iloc[pos]
        mark = float(row["close"])
        if pos == 0:
            eq = initial_capital
        else:
            unreal = qty * (mark - entry_fill)
            eq = max(0.0, initial_capital - entry_fee + unreal)
        equity_curve.append(EquityPoint(timestamp=pd.Timestamp(row["timestamp"]), equity=eq))
    equity_curve[-1] = EquityPoint(
        timestamp=equity_curve[-1].timestamp, equity=final_equity
    )

    from app.backtest.metrics import compute_metrics

    return compute_metrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=[trade],
        equity_curve=equity_curve,
    )


def slice_eval_frame(history: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    from app.backtest.robust import slice_evaluation

    return slice_evaluation(
        history,
        start,
        end,
        warmup_bars=WARMUP_BARS,
        required_columns=RESEARCH_REQUIRED_COLUMNS,
    )
