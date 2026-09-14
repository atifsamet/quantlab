"""
Risk manager: approve/reject trades and size positions (no order placement).

Consecutive-loss reset
----------------------
After ``max_consecutive_losses`` losing closed trades, new entries are rejected.
The block clears when a later closed trade is a **win** (net_pnl > 0).
Daily loss limits reset automatically at the start of each UTC trading day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from app.risk.config import RiskConfig


@dataclass(slots=True)
class RiskDecision:
    """Structured outcome of a risk evaluation (never places an order)."""

    approved: bool
    reason: str
    side: str | None
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    risk_amount: float | None
    quantity: float | None
    risk_reward_ratio: float | None


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_utc_date(timestamp: Any) -> date:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.date()


def compute_stop_loss(
    side: str,
    entry_price: float,
    atr: float,
    atr_multiplier: float,
) -> float:
    """ATR-based stop distance from entry."""
    side_u = side.upper()
    distance = atr * atr_multiplier
    if side_u == "LONG":
        return entry_price - distance
    if side_u == "SHORT":
        return entry_price + distance
    raise ValueError(f"Unsupported side: {side}")


def compute_take_profit(
    side: str,
    entry_price: float,
    stop_loss: float,
    risk_reward_ratio: float,
) -> float:
    """Take-profit from stop distance × reward multiple."""
    side_u = side.upper()
    if side_u == "LONG":
        risk = entry_price - stop_loss
        return entry_price + risk * risk_reward_ratio
    if side_u == "SHORT":
        risk = stop_loss - entry_price
        return entry_price - risk * risk_reward_ratio
    raise ValueError(f"Unsupported side: {side}")


def compute_position_quantity(
    *,
    equity: float,
    risk_per_trade: float,
    maximum_risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    max_position_size_pct: float,
) -> tuple[float, float]:
    """
    Return ``(quantity, risk_amount)`` from account risk and stop distance.

    Quantity is capped so notional does not exceed ``equity * max_position_size_pct``.
    """
    effective_risk = min(risk_per_trade, maximum_risk_per_trade)
    risk_amount = equity * effective_risk
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0 or risk_amount <= 0 or entry_price <= 0:
        return 0.0, risk_amount

    quantity = risk_amount / stop_distance
    max_notional = equity * max_position_size_pct
    max_qty = max_notional / entry_price if entry_price > 0 else 0.0
    if max_qty > 0:
        quantity = min(quantity, max_qty)
    return quantity, risk_amount


class RiskManager:
    """
    Evaluates strategy signals into approved/rejected risk decisions.

    Does not place orders or call exchange APIs.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._current_day: date | None = None
        self._day_start_equity: float = self.config.initial_capital
        self._day_realized_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._consecutive_block_active: bool = False

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def consecutive_block_active(self) -> bool:
        return self._consecutive_block_active

    @property
    def day_realized_pnl(self) -> float:
        return self._day_realized_pnl

    def evaluate(
        self,
        *,
        side: str,
        entry_price: float,
        atr: float,
        equity: float,
        timestamp: Any | None = None,
    ) -> RiskDecision:
        """Validate and size a prospective LONG/SHORT entry."""
        cfg = self.config
        side_u = str(side).upper()
        now = timestamp if timestamp is not None else pd.Timestamp.now(tz="UTC")
        self._roll_day(now, equity)

        def reject(reason: str) -> RiskDecision:
            return RiskDecision(
                approved=False,
                reason=reason,
                side=side_u if side_u in {"LONG", "SHORT"} else None,
                entry_price=_finite(entry_price),
                stop_loss=None,
                take_profit=None,
                risk_amount=None,
                quantity=None,
                risk_reward_ratio=cfg.risk_reward_ratio,
            )

        if side_u not in {"LONG", "SHORT"}:
            return reject("signal must be LONG or SHORT")

        entry = _finite(entry_price)
        if entry is None or entry <= 0:
            return reject("entry price must be a finite number > 0")

        atr_v = _finite(atr)
        if atr_v is None or atr_v <= 0:
            return reject("ATR must be a finite number > 0")

        if equity <= 0:
            return reject("equity must be > 0")

        if self._daily_loss_limit_breached():
            return reject("daily loss limit reached")

        if self._consecutive_block_active:
            return reject("consecutive loss limit reached")

        stop = compute_stop_loss(side_u, entry, atr_v, cfg.atr_multiplier)
        if side_u == "LONG" and stop >= entry:
            return reject("invalid stop-loss for LONG")
        if side_u == "SHORT" and stop <= entry:
            return reject("invalid stop-loss for SHORT")

        take = compute_take_profit(side_u, entry, stop, cfg.risk_reward_ratio)
        if side_u == "LONG" and take <= entry:
            return reject("invalid take-profit for LONG")
        if side_u == "SHORT" and take >= entry:
            return reject("invalid take-profit for SHORT")

        quantity, risk_amount = compute_position_quantity(
            equity=equity,
            risk_per_trade=cfg.risk_per_trade,
            maximum_risk_per_trade=cfg.maximum_risk_per_trade,
            entry_price=entry,
            stop_loss=stop,
            max_position_size_pct=cfg.max_position_size_pct,
        )
        if quantity <= 0 or not math.isfinite(quantity):
            return reject("position size must be > 0")

        max_notional = equity * cfg.max_position_size_pct
        if quantity * entry > max_notional + 1e-9:
            return reject("position size exceeds maximum")

        return RiskDecision(
            approved=True,
            reason="approved",
            side=side_u,
            entry_price=entry,
            stop_loss=stop,
            take_profit=take,
            risk_amount=risk_amount,
            quantity=quantity,
            risk_reward_ratio=cfg.risk_reward_ratio,
        )

    def register_closed_trade(self, net_pnl: float, *, timestamp: Any, equity: float) -> None:
        """Update daily loss and consecutive-loss state after a closed trade."""
        self._roll_day(timestamp, equity)
        pnl = float(net_pnl)
        self._day_realized_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._consecutive_block_active = True
        elif pnl > 0:
            # Reset condition: any winning trade clears the consecutive-loss block.
            self._consecutive_losses = 0
            self._consecutive_block_active = False
        # Break-even (pnl == 0) leaves the streak unchanged.

    def _roll_day(self, timestamp: Any, equity: float) -> None:
        day = _as_utc_date(timestamp)
        if self._current_day is None:
            self._current_day = day
            self._day_start_equity = equity
            self._day_realized_pnl = 0.0
            return
        if day != self._current_day:
            self._current_day = day
            self._day_start_equity = equity
            self._day_realized_pnl = 0.0

    def _daily_loss_limit_breached(self) -> bool:
        if self._day_start_equity <= 0:
            return True
        loss = min(0.0, self._day_realized_pnl)
        limit = self._day_start_equity * self.config.max_daily_loss_pct
        return abs(loss) >= limit - 1e-12
