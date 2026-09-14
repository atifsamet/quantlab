"""Strategy package (Phase 3 — signals only, no order placement)."""

from app.strategy.baseline import (
    REQUIRED_INDICATORS,
    BaselineStrategy,
    Signal,
    StrategySignal,
    generate_signal,
    generate_signals,
)

__all__ = [
    "REQUIRED_INDICATORS",
    "BaselineStrategy",
    "Signal",
    "StrategySignal",
    "generate_signal",
    "generate_signals",
]
