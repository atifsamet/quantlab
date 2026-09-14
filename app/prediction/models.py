"""Pydantic / dataclass models for QuantLab predictions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PredictionSide = Literal["LONG", "SHORT", "NEUTRAL"]
TrendLabel = Literal["bullish", "bearish", "sideways"]


class Explanation(BaseModel):
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class IndicatorSnapshot(BaseModel):
    ema_20: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    atr_pct: float | None = None
    volume: float | None = None
    volume_sma_20: float | None = None
    volume_ratio: float | None = None


class ModelLevels(BaseModel):
    """ATR-based analytical levels — not guaranteed prices."""

    entry_low: float | None = None
    entry_high: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward: float | None = None
    label: str = "Model levels (analytical only)"


class LiveMeta(BaseModel):
    """Market / prediction freshness (Phase 14)."""

    market_status: Literal["LIVE", "STALE", "ERROR"] = "LIVE"
    prediction_status: Literal["CURRENT", "STALE"] = "CURRENT"
    data_timestamp: str | None = None
    last_market_update: str | None = None
    prediction_candle_open: str | None = None
    prediction_candle_close: str | None = None
    candle_source: str = "closed"
    dropped_incomplete_candle: bool = False
    age_seconds: int | None = None
    stale_after_seconds: int | None = None
    engine_version: str | None = None


class PredictionResult(BaseModel):
    symbol: str
    timeframe: str
    timestamp: str
    current_price: float
    prediction: PredictionSide
    # Analytical agreement score 0-100 — NOT a calibrated probability of profit
    signal_strength: int = Field(ge=0, le=100)
    signal_strength_label: str = "Signal strength (analytical score, not win probability)"
    signal_strength_tooltip: str = (
        "Analytical signal strength derived from the current model inputs. "
        "It is not a probability of future return."
    )
    # Optional ML probability if a persisted model exists; else null
    ml_probability: float | None = None
    ml_probability_note: str | None = (
        "Phase 10 found no reliable ML edge; probability omitted unless a "
        "validated model artifact is present."
    )
    trend: TrendLabel
    strategy_signal: str
    levels: ModelLevels
    indicators: IndicatorSnapshot
    reasons: Explanation
    warnings: list[str] = Field(default_factory=list)
    research_conclusion: str
    disclaimer: str
    live: LiveMeta | None = None
    engine_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
