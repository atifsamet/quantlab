"""
Simple multi-asset portfolio simulation (Phase 9I).

Conservative rules (configurable):
- leverage = 1x
- max risk per trade (default 1% of equity)
- max open positions (default 3)
- total portfolio risk cap (default 3%)

Never places live orders. Funding is not applied here (fee + slippage only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.backtest.engine import (
    BacktestConfig,
    _buy_price,
    _close_position,
    _exit_fill_from_open,
    _open_position_risk,
    _parse_signal,
    resolve_sl_tp_hit,
)
from app.backtest.metrics import compute_metrics
from app.backtest.models import BacktestResult, EquityPoint, PositionSide, Trade
from app.risk.config import RiskConfig
from app.risk.manager import RiskManager
from app.strategy.baseline import Signal


@dataclass(slots=True)
class PortfolioConfig:
    initial_capital: float = 1000.0
    max_open_positions: int = 3
    risk_per_trade: float = 0.01
    max_portfolio_risk_pct: float = 0.03
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    atr_multiplier: float = 1.5
    risk_reward_ratio: float = 2.0


@dataclass(slots=True)
class _PortPosition:
    symbol: str
    engine_pos: Any
    risk_amount: float


@dataclass(slots=True)
class PortfolioResult:
    result: BacktestResult
    open_position_peak: int = 0
    symbols_traded: list[str] = field(default_factory=list)
    notes: str = ""


def run_portfolio_backtest(
    asset_frames: dict[str, pd.DataFrame],
    *,
    config: PortfolioConfig | None = None,
) -> PortfolioResult:
    """
    Event-driven portfolio simulator across symbols on one timeframe.

    Each frame needs timestamp, OHLCV, atr_14, and signal.
    Signal on candle N executes at that symbol's N+1 open.
    """
    cfg = config or PortfolioConfig()
    if cfg.max_open_positions < 1:
        raise ValueError("max_open_positions must be >= 1")
    if not (0 < cfg.risk_per_trade <= 1):
        raise ValueError("risk_per_trade must be in (0, 1]")
    if cfg.max_portfolio_risk_pct < cfg.risk_per_trade:
        raise ValueError("max_portfolio_risk_pct must be >= risk_per_trade")
    if not asset_frames:
        raise ValueError("asset_frames is empty")

    indexed = {
        sym: frame.set_index(pd.to_datetime(frame["timestamp"], utc=True)).sort_index()
        for sym, frame in asset_frames.items()
    }
    all_ts = sorted({ts for frame in indexed.values() for ts in frame.index})

    bt_cfg = BacktestConfig(
        initial_capital=cfg.initial_capital,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        leverage=1.0,
    )
    risk = RiskManager(
        RiskConfig(
            initial_capital=cfg.initial_capital,
            risk_per_trade=cfg.risk_per_trade,
            atr_multiplier=cfg.atr_multiplier,
            risk_reward_ratio=cfg.risk_reward_ratio,
        )
    )

    cash = float(cfg.initial_capital)
    positions: dict[str, _PortPosition] = {}
    trades: list[Trade] = []
    equity_curve: list[EquityPoint] = []
    next_trade_id = 1
    peak_open = 0
    symbols_traded: set[str] = set()

    for i, ts in enumerate(all_ts):
        # SL/TP first
        for sym in list(positions.keys()):
            frame = indexed[sym]
            if ts not in frame.index:
                continue
            row = frame.loc[ts]
            p = positions[sym].engine_pos
            if p.stop_loss is None or p.take_profit is None:
                continue
            hit = resolve_sl_tp_hit(
                side=p.side,
                stop_loss=p.stop_loss,
                take_profit=p.take_profit,
                high=float(row["high"]),
                low=float(row["low"]),
            )
            if hit is None:
                continue
            fill = p.stop_loss if hit == "SL" else p.take_profit
            trade, cash = _close_position(
                position=p,
                exec_ts=ts,
                fill_price=float(fill),
                cash=cash,
                cfg=bt_cfg,
                trade_id=next_trade_id,
            )
            trades.append(trade)
            next_trade_id += 1
            risk.register_closed_trade(trade.net_pnl, timestamp=ts, equity=max(0.0, cash))
            del positions[sym]

        equity = cash
        for sym, pos in positions.items():
            frame = indexed[sym]
            if ts not in frame.index:
                continue
            close_px = float(frame.loc[ts, "close"])
            p = pos.engine_pos
            if p.side is PositionSide.LONG:
                equity += p.capital_at_entry + p.quantity * (close_px - p.entry_price)
            else:
                equity += p.capital_at_entry + p.quantity * (p.entry_price - close_px)
        equity_curve.append(EquityPoint(timestamp=ts, equity=max(0.0, equity)))
        peak_open = max(peak_open, len(positions))

        if i >= len(all_ts) - 1:
            continue
        next_ts = all_ts[i + 1]

        # New entries from signals on this bar
        if len(positions) >= cfg.max_open_positions:
            continue
        open_risk_frac = sum(p.risk_amount for p in positions.values()) / max(equity, 1.0)

        candidates: list[tuple[str, PositionSide, float]] = []
        for sym, frame in indexed.items():
            if sym in positions or ts not in frame.index or next_ts not in frame.index:
                continue
            signal = _parse_signal(frame.loc[ts, "signal"])
            if signal is Signal.HOLD:
                continue
            desired = PositionSide.LONG if signal is Signal.LONG else PositionSide.SHORT
            atr_raw = frame.loc[ts, "atr_14"] if "atr_14" in frame.columns else float("nan")
            candidates.append((sym, desired, float(atr_raw) if pd.notna(atr_raw) else float("nan")))

        for sym, desired, atr_raw in candidates:
            if len(positions) >= cfg.max_open_positions:
                break
            if open_risk_frac + cfg.risk_per_trade > cfg.max_portfolio_risk_pct + 1e-12:
                break
            exec_open = float(indexed[sym].loc[next_ts, "open"])
            if exec_open <= 0:
                continue
            entry_fill = (
                _buy_price(exec_open, cfg.slippage_rate)
                if desired is PositionSide.LONG
                else exec_open * (1.0 - cfg.slippage_rate)
            )
            decision = risk.evaluate(
                side=desired.value,
                entry_price=entry_fill,
                atr=atr_raw,
                equity=max(0.0, cash),
                timestamp=next_ts,
            )
            if not decision.approved or decision.quantity is None:
                continue
            opened_pos, cash, ok = _open_position_risk(
                side=desired,
                exec_ts=next_ts,
                entry_fill=entry_fill,
                quantity=float(decision.quantity),
                stop_loss=float(decision.stop_loss) if decision.stop_loss is not None else None,
                take_profit=float(decision.take_profit) if decision.take_profit is not None else None,
                cash=cash,
                cfg=bt_cfg,
            )
            if not ok or opened_pos is None:
                continue
            risk_amt = max(0.0, cash + opened_pos.capital_at_entry) * cfg.risk_per_trade
            if getattr(decision, "risk_amount", None) is not None:
                risk_amt = float(decision.risk_amount)
            positions[sym] = _PortPosition(
                symbol=sym, engine_pos=opened_pos, risk_amount=risk_amt
            )
            symbols_traded.add(sym)
            open_risk_frac = sum(p.risk_amount for p in positions.values()) / max(
                max(0.0, cash), 1.0
            )

    if positions:
        last_ts = all_ts[-1]
        for sym, pos in list(positions.items()):
            close_px = float(indexed[sym].iloc[-1]["close"])
            trade, cash = _close_position(
                position=pos.engine_pos,
                exec_ts=last_ts,
                fill_price=_exit_fill_from_open(pos.engine_pos.side, close_px, bt_cfg),
                cash=cash,
                cfg=bt_cfg,
                trade_id=next_trade_id,
            )
            trades.append(trade)
            next_trade_id += 1
            del positions[sym]
        if equity_curve:
            equity_curve[-1] = EquityPoint(timestamp=last_ts, equity=max(0.0, cash))

    result = compute_metrics(
        initial_capital=cfg.initial_capital,
        final_equity=max(0.0, cash),
        trades=trades,
        equity_curve=equity_curve,
    )
    return PortfolioResult(
        result=result,
        open_position_peak=peak_open,
        symbols_traded=sorted(symbols_traded),
        notes="fee+slippage only; funding not applied in portfolio path",
    )
