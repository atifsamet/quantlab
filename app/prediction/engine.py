"""
QuantLab prediction engine.

Reuses existing indicators, breakout strategy, and ATR risk levels.
Does not place orders. Signal strength is an analytical score, not a
calibrated win probability (Phase 10 found no durable ML edge).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pandas as pd

from app.data.okx_historical import normalize_timeframe
from app.exchange.okx_client import OkxClient
from app.indicators import WARMUP_BARS, add_indicators
from app.prediction.config import (
    CANDLE_FETCH_LIMIT,
    CACHE_TTL_SECONDS,
    DISCLAIMER,
    ENGINE_VERSION,
    MARKET_DATA_DISCLAIMER,
    OKX_BAR,
    RESEARCH_CONCLUSION,
    SIGNAL_STRENGTH_TOOLTIP,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
)
from app.prediction.explanation import build_explanation
from app.prediction.freshness import select_closed_candles
from app.prediction.models import (
    IndicatorSnapshot,
    LiveMeta,
    ModelLevels,
    PredictionResult,
    PredictionSide,
    TrendLabel,
)
from app.research.indicators_flex import add_research_indicators
from app.research.params import ResearchParams
from app.research.variants import build_strategy
from app.risk.config import RiskConfig
from app.risk.manager import compute_stop_loss, compute_take_profit


class PredictionError(ValueError):
    """Invalid prediction request or insufficient market data."""


_cache: dict[str, tuple[float, PredictionResult]] = {}
_market_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def validate_symbol(symbol: str) -> str:
    sym = symbol.strip().upper()
    if sym not in SUPPORTED_SYMBOLS:
        raise PredictionError(
            f"Unsupported symbol '{symbol}'. Supported: {', '.join(SUPPORTED_SYMBOLS)}"
        )
    return sym


def validate_timeframe(timeframe: str) -> str:
    tf = normalize_timeframe(timeframe)
    if tf not in SUPPORTED_TIMEFRAMES:
        raise PredictionError(
            f"Unsupported timeframe '{timeframe}'. Supported: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return tf


def _score_conditions(
    *,
    trend: TrendLabel,
    strategy_signal: str,
    rsi: float | None,
    macd_hist: float | None,
    volume_ratio: float | None,
    ema20: float | None,
    ema50: float | None,
    ema200: float | None,
    close: float,
) -> tuple[PredictionSide, int]:
    """
    Deterministic directional vote → LONG/SHORT/NEUTRAL + strength 0-100.

    Strength reflects agreement among indicators. This is NOT P(profit).
    """
    bull = 0
    bear = 0
    votes = 0

    if trend == "bullish":
        bull += 1
        votes += 1
    elif trend == "bearish":
        bear += 1
        votes += 1

    if strategy_signal == "LONG":
        bull += 1
        votes += 1
    elif strategy_signal == "SHORT":
        bear += 1
        votes += 1

    if ema20 is not None and ema50 is not None:
        votes += 1
        if ema20 > ema50:
            bull += 1
        else:
            bear += 1
    if ema50 is not None and ema200 is not None:
        votes += 1
        if ema50 > ema200:
            bull += 1
        else:
            bear += 1
    if rsi is not None:
        if rsi >= 55:
            votes += 1
            bull += 1
        elif rsi <= 45:
            votes += 1
            bear += 1
    if macd_hist is not None:
        votes += 1
        if macd_hist > 0:
            bull += 1
        else:
            bear += 1
    if volume_ratio is not None and volume_ratio >= 1.0 and ema50 is not None:
        votes += 1
        if close >= ema50:
            bull += 1
        else:
            bear += 1

    if votes == 0:
        return "NEUTRAL", 0

    if bull > bear + 1:
        side: PredictionSide = "LONG"
        strength = int(round(100 * (bull - bear) / votes))
    elif bear > bull + 1:
        side = "SHORT"
        strength = int(round(100 * (bear - bull) / votes))
    else:
        side = "NEUTRAL"
        strength = int(round(100 * abs(bull - bear) / votes))

    return side, int(max(0, min(100, strength)))


def _trend_label(close: float, ema50: float | None, ema200: float | None) -> TrendLabel:
    if ema50 is None or ema200 is None:
        return "sideways"
    if close > ema50 > ema200:
        return "bullish"
    if close < ema50 < ema200:
        return "bearish"
    return "sideways"


def _levels(
    *,
    side: PredictionSide,
    price: float,
    atr: float | None,
) -> ModelLevels:
    if atr is None or atr <= 0 or side == "NEUTRAL":
        return ModelLevels(
            entry_low=round(price * 0.998, 6) if side != "NEUTRAL" else None,
            entry_high=round(price * 1.002, 6) if side != "NEUTRAL" else None,
            stop_loss=None,
            take_profit=None,
            risk_reward=None,
        )
    risk_cfg = RiskConfig()
    trade_side = "LONG" if side == "LONG" else "SHORT"
    stop = compute_stop_loss(trade_side, price, atr, risk_cfg.atr_multiplier)
    take = compute_take_profit(trade_side, price, stop, risk_cfg.risk_reward_ratio)
    half = atr * 0.25
    return ModelLevels(
        entry_low=round(price - half, 6),
        entry_high=round(price + half, 6),
        stop_loss=round(stop, 6),
        take_profit=round(take, 6),
        risk_reward=float(risk_cfg.risk_reward_ratio),
    )


def fetch_candles(symbol: str, timeframe: str, *, client: OkxClient | None = None) -> pd.DataFrame:
    symbol = validate_symbol(symbol)
    timeframe = validate_timeframe(timeframe)
    bar = OKX_BAR[timeframe]
    owns = client is None
    cli = client or OkxClient()
    try:
        frame = cli.get_candles(symbol, bar=bar, limit=CANDLE_FETCH_LIMIT)
    finally:
        if owns:
            cli.close()
    if frame.empty:
        raise PredictionError("OKX returned no candles")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def build_prediction_from_ohlcv(
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    now: datetime | None = None,
    use_closed_only: bool = True,
    freshness_meta: dict[str, Any] | None = None,
) -> PredictionResult:
    """
    Build prediction from OHLCV.

    When ``use_closed_only`` is True (default for live), incomplete trailing
    candles are dropped so the prediction candle is the latest closed bar.
    """
    symbol = validate_symbol(symbol)
    timeframe = validate_timeframe(timeframe)

    meta = freshness_meta
    frame = ohlcv
    if use_closed_only:
        try:
            frame, meta = select_closed_candles(ohlcv, timeframe, now=now)
        except ValueError as exc:
            raise PredictionError(str(exc)) from exc

    if len(frame) < WARMUP_BARS:
        raise PredictionError(
            f"Insufficient candles for indicators: need >={WARMUP_BARS}, got {len(frame)}"
        )

    # Phase-2 indicators + Phase-9 breakout research frame
    with_ind = add_indicators(frame)
    params = ResearchParams(variant="breakout", breakout_lookback=20)
    research = add_research_indicators(frame, params)
    signaled = build_strategy(params).generate_signals(research)

    last = with_ind.iloc[-1]
    last_sig = signaled.iloc[-1]
    close = float(last["close"])
    ts = pd.Timestamp(last["timestamp"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    def f(col: str) -> float | None:
        if col not in last.index or pd.isna(last[col]):
            return None
        return float(last[col])

    indicators = IndicatorSnapshot(
        ema_20=f("ema_20"),
        ema_50=f("ema_50"),
        ema_200=f("ema_200"),
        rsi_14=f("rsi_14"),
        macd=f("macd"),
        macd_signal=f("macd_signal"),
        macd_hist=f("macd_hist"),
        atr_14=f("atr_14"),
        atr_pct=(float(last["atr_14"]) / close) if f("atr_14") else None,
        volume=f("volume"),
        volume_sma_20=f("volume_sma_20"),
        volume_ratio=(
            float(last["volume"]) / float(last["volume_sma_20"])
            if f("volume") and f("volume_sma_20") and float(last["volume_sma_20"]) > 0
            else None
        ),
    )
    strategy_signal = str(last_sig["signal"]).upper()
    trend = _trend_label(close, indicators.ema_50, indicators.ema_200)
    prediction, strength = _score_conditions(
        trend=trend,
        strategy_signal=strategy_signal,
        rsi=indicators.rsi_14,
        macd_hist=indicators.macd_hist,
        volume_ratio=indicators.volume_ratio,
        ema20=indicators.ema_20,
        ema50=indicators.ema_50,
        ema200=indicators.ema_200,
        close=close,
    )
    levels = _levels(side=prediction, price=close, atr=indicators.atr_14)
    reasons = build_explanation(
        prediction=prediction,
        strategy_signal=strategy_signal,
        indicators=indicators,
        trend=trend,
    )
    if indicators.ema_200 is not None:
        if close > indicators.ema_200:
            reasons.positive.append("Price is above EMA200")
        else:
            reasons.negative.append("Price is below EMA200")

    warnings = [
        RESEARCH_CONCLUSION,
        "Signal strength is an analytical agreement score, not a calibrated probability of profit.",
        "Phase 10 ML filter research did not demonstrate a durable edge.",
        "Phase 11 statistical validation: no evidence of an edge vs random.",
        MARKET_DATA_DISCLAIMER,
    ]

    live = None
    if meta:
        live = LiveMeta(
            market_status=meta.get("market_status", "LIVE"),
            prediction_status=meta.get("prediction_status", "CURRENT"),
            data_timestamp=meta.get("data_timestamp"),
            last_market_update=meta.get("last_market_update"),
            prediction_candle_open=meta.get("prediction_candle_open"),
            prediction_candle_close=meta.get("prediction_candle_close"),
            candle_source=meta.get("candle_source", "closed"),
            dropped_incomplete_candle=bool(meta.get("dropped_incomplete_candle", False)),
            age_seconds=meta.get("age_seconds"),
            stale_after_seconds=meta.get("stale_after_seconds"),
            engine_version=ENGINE_VERSION,
        )

    return PredictionResult(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts.isoformat(),
        current_price=close,
        prediction=prediction,
        signal_strength=strength,
        signal_strength_tooltip=SIGNAL_STRENGTH_TOOLTIP,
        ml_probability=None,
        trend=trend,
        strategy_signal=strategy_signal,
        levels=levels,
        indicators=indicators,
        reasons=reasons,
        warnings=warnings,
        research_conclusion=RESEARCH_CONCLUSION,
        disclaimer=DISCLAIMER,
        live=live,
        engine_version=ENGINE_VERSION,
    )


def predict(
    symbol: str,
    timeframe: str,
    *,
    use_cache: bool = True,
    client: OkxClient | None = None,
    now: datetime | None = None,
) -> PredictionResult:
    symbol = validate_symbol(symbol)
    timeframe = validate_timeframe(timeframe)
    key = f"{symbol}:{timeframe}"
    now_m = time.monotonic()
    if use_cache and key in _cache:
        ts, cached = _cache[key]
        if now_m - ts < CACHE_TTL_SECONDS:
            return cached

    ohlcv = fetch_candles(symbol, timeframe, client=client)
    result = build_prediction_from_ohlcv(
        ohlcv,
        symbol=symbol,
        timeframe=timeframe,
        now=now,
        use_closed_only=True,
    )
    _cache[key] = (now_m, result)
    return result


def market_snapshot(
    symbol: str,
    timeframe: str,
    *,
    use_cache: bool = True,
    client: OkxClient | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recent OHLCV + indicators for charts (cached). Uses closed candles for series."""
    symbol = validate_symbol(symbol)
    timeframe = validate_timeframe(timeframe)
    key = f"mkt:{symbol}:{timeframe}"
    now_m = time.monotonic()
    if use_cache and key in _market_cache:
        ts, cached = _market_cache[key]
        if now_m - ts < CACHE_TTL_SECONDS:
            return cached

    ohlcv = fetch_candles(symbol, timeframe, client=client)
    try:
        closed, meta = select_closed_candles(ohlcv, timeframe, now=now)
    except ValueError as exc:
        raise PredictionError(str(exc)) from exc

    framed = add_indicators(closed)
    chart = framed.tail(120).copy()
    candles = []
    for _, row in chart.iterrows():
        candles.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "ema_20": float(row["ema_20"]) if pd.notna(row.get("ema_20")) else None,
                "ema_50": float(row["ema_50"]) if pd.notna(row.get("ema_50")) else None,
                "ema_200": float(row["ema_200"]) if pd.notna(row.get("ema_200")) else None,
            }
        )
    last = framed.iloc[-1]
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": pd.Timestamp(last["timestamp"]).isoformat(),
        "candles": candles,
        "last_price": float(last["close"]),
        "live": meta,
        "market_disclaimer": MARKET_DATA_DISCLAIMER,
    }
    _market_cache[key] = (now_m, payload)
    return payload


def clear_prediction_cache() -> None:
    _cache.clear()
    _market_cache.clear()
