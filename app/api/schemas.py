"""FastAPI response schemas for QuantLab."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.prediction.models import PredictionResult


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str = "RESEARCH MODE"
    live_trading_enabled: bool = False
    product: str = "QuantLab"


class ErrorResponse(BaseModel):
    detail: str


class OverviewCard(BaseModel):
    symbol: str
    timeframe: str
    current_price: float | None = None
    prediction: str | None = None
    signal_strength: int | None = None
    trend: str | None = None
    timestamp: str | None = None
    market_status: str | None = None
    prediction_status: str | None = None
    last_market_update: str | None = None
    error: str | None = None


class OverviewResponse(BaseModel):
    mode: str = "RESEARCH MODE"
    live_trading_enabled: bool = False
    research_conclusion: str
    disclaimer: str
    market_data_disclaimer: str | None = None
    cards: list[OverviewCard]
    analytics_verdict: str | None = None
    monte_carlo_available: bool = False
    prediction_performance: dict[str, Any] | None = None
