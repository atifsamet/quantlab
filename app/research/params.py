"""Configurable research parameters (intentionally small search space)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# Phase 8 variants + Phase 9 strategy families
KNOWN_VARIANTS = frozenset(
    {
        "baseline",
        "trend_momentum",
        "ema_cross_rsi",
        "ema_macd",
        "breakout_momentum",
        # Phase 9 families
        "trend_following",
        "momentum",
        "breakout",
        "mean_reversion",
    }
)


@dataclass(frozen=True, slots=True)
class ResearchParams:
    """Strategy + risk knobs explored in research (not unrestricted brute force)."""

    variant: str = "baseline"
    ema_fast: int = 20
    ema_med: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_long_low: float = 50.0
    rsi_long_high: float = 70.0
    rsi_short_low: float = 30.0
    rsi_short_high: float = 50.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    risk_reward_ratio: float = 2.0
    breakout_lookback: int = 20
    # Mean-reversion knobs
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    mean_dev_pct: float = 0.015
    min_atr_pct: float = 0.002

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        return (
            f"{self.variant}|ema={self.ema_fast}/{self.ema_med}/{self.ema_slow}"
            f"|rsi={self.rsi_long_low:.0f}-{self.rsi_long_high:.0f}/"
            f"{self.rsi_short_low:.0f}-{self.rsi_short_high:.0f}"
            f"|atrx={self.atr_stop_multiplier}|rr={self.risk_reward_ratio}"
        )


def validate_research_params(params: ResearchParams) -> None:
    if params.variant not in KNOWN_VARIANTS:
        raise ValueError(
            f"Unknown variant '{params.variant}'. Known: {sorted(KNOWN_VARIANTS)}"
        )
    if params.ema_fast < 1 or params.ema_med < 1 or params.ema_slow < 1:
        raise ValueError("EMA periods must be >= 1")
    if not (params.ema_fast < params.ema_med < params.ema_slow):
        raise ValueError("Require ema_fast < ema_med < ema_slow")
    if params.rsi_period < 2:
        raise ValueError("rsi_period must be >= 2")
    if not (0 <= params.rsi_long_low < params.rsi_long_high <= 100):
        raise ValueError("Invalid RSI long thresholds")
    if not (0 <= params.rsi_short_low < params.rsi_short_high <= 100):
        raise ValueError("Invalid RSI short thresholds")
    if not (1 <= params.macd_fast < params.macd_slow):
        raise ValueError("Require 1 <= macd_fast < macd_slow")
    if params.macd_signal < 1:
        raise ValueError("macd_signal must be >= 1")
    if params.atr_period < 1:
        raise ValueError("atr_period must be >= 1")
    if params.atr_stop_multiplier <= 0:
        raise ValueError("atr_stop_multiplier must be > 0")
    if params.risk_reward_ratio <= 0:
        raise ValueError("risk_reward_ratio must be > 0")
    if params.breakout_lookback < 2:
        raise ValueError("breakout_lookback must be >= 2")
    if not (0 <= params.rsi_oversold < params.rsi_overbought <= 100):
        raise ValueError("Invalid RSI oversold/overbought")
    if params.mean_dev_pct <= 0:
        raise ValueError("mean_dev_pct must be > 0")
    if params.min_atr_pct < 0:
        raise ValueError("min_atr_pct must be >= 0")
