"""Configurable risk parameters (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskConfig:
    """
    Risk controls applied between strategy signals and simulated execution.

    These values are defaults only — callers should pass an explicit config
    rather than scattering literals through the codebase.
    """

    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01
    maximum_risk_per_trade: float = 0.02
    risk_reward_ratio: float = 2.0
    atr_multiplier: float = 1.5
    max_position_size_pct: float = 1.0
    max_daily_loss_pct: float = 0.03
    max_consecutive_losses: int = 3

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be in (0, 1]")
        if not (0 < self.maximum_risk_per_trade <= 1):
            raise ValueError("maximum_risk_per_trade must be in (0, 1]")
        if self.risk_per_trade > self.maximum_risk_per_trade:
            raise ValueError("risk_per_trade cannot exceed maximum_risk_per_trade")
        if self.risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio must be > 0")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if not (0 < self.max_position_size_pct <= 1):
            raise ValueError("max_position_size_pct must be in (0, 1]")
        if not (0 < self.max_daily_loss_pct <= 1):
            raise ValueError("max_daily_loss_pct must be in (0, 1]")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
