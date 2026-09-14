"""
Baseline trend + momentum confirmation strategy (Phase 3).

Produces LONG / SHORT / HOLD signals only. Does not place orders.
This is an unvalidated educational baseline — not a claim of profitability.

Signals use only the current candle's indicator values (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import math

import pandas as pd

REQUIRED_INDICATORS = (
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
)

REASON_LONG = "bullish trend + momentum confirmation"
REASON_SHORT = "bearish trend + momentum confirmation"
REASON_HOLD = "no confirmed setup"
REASON_MISSING = "missing or invalid indicators"


class Signal(str, Enum):
    """Discrete strategy action for the current candle."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """Deterministic signal plus a short human-readable reason."""

    signal: Signal
    reason: str

    def __str__(self) -> str:
        return f"{self.signal.value}: {self.reason}"


def _as_finite_float(value: Any) -> float | None:
    """Coerce a value to float; return None if missing/invalid/non-finite."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _has_key(row: Mapping[str, Any] | pd.Series, key: str) -> bool:
    if isinstance(row, pd.Series):
        return key in row.index
    return key in row


def _extract_indicators(row: Mapping[str, Any] | pd.Series) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for key in REQUIRED_INDICATORS:
        if not _has_key(row, key):
            return None
        parsed = _as_finite_float(row[key])
        if parsed is None:
            return None
        values[key] = parsed
    return values


def _is_long(ind: Mapping[str, float]) -> bool:
    return (
        ind["ema_20"] > ind["ema_50"]
        and ind["ema_50"] > ind["ema_200"]
        and 50.0 <= ind["rsi_14"] <= 70.0
        and ind["macd"] > ind["macd_signal"]
        and ind["macd_hist"] > 0.0
    )


def _is_short(ind: Mapping[str, float]) -> bool:
    return (
        ind["ema_20"] < ind["ema_50"]
        and ind["ema_50"] < ind["ema_200"]
        and 30.0 <= ind["rsi_14"] <= 50.0
        and ind["macd"] < ind["macd_signal"]
        and ind["macd_hist"] < 0.0
    )


class BaselineStrategy:
    """
    Trend + momentum confirmation baseline.

    LONG when EMAs are stacked bullish, RSI in 50–70, and MACD confirms up.
    SHORT when EMAs are stacked bearish, RSI in 30–50, and MACD confirms down.
    Otherwise HOLD.
    """

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        """Evaluate a single indicator-enriched candle row."""
        indicators = _extract_indicators(row)
        if indicators is None:
            return StrategySignal(Signal.HOLD, REASON_MISSING)

        if _is_long(indicators):
            return StrategySignal(Signal.LONG, REASON_LONG)
        if _is_short(indicators):
            return StrategySignal(Signal.SHORT, REASON_SHORT)
        return StrategySignal(Signal.HOLD, REASON_HOLD)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of ``df`` with ``signal`` and ``signal_reason`` columns.

        Each row is evaluated independently using only that row's values
        (no future candles).
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        out = df.copy()
        signals: list[str] = []
        reasons: list[str] = []
        for _, row in out.iterrows():
            result = self.generate_signal(row)
            signals.append(result.signal.value)
            reasons.append(result.reason)
        out["signal"] = signals
        out["signal_reason"] = reasons
        return out


_DEFAULT_STRATEGY = BaselineStrategy()


def generate_signal(row: Mapping[str, Any] | pd.Series) -> StrategySignal:
    """Module-level helper using the default baseline strategy instance."""
    return _DEFAULT_STRATEGY.generate_signal(row)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Module-level helper that annotates a DataFrame with signals."""
    return _DEFAULT_STRATEGY.generate_signals(df)
