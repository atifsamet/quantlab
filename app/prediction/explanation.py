"""Build human-readable explanation factors from indicator state."""

from __future__ import annotations

from app.prediction.models import Explanation, IndicatorSnapshot, PredictionSide


def build_explanation(
    *,
    prediction: PredictionSide,
    strategy_signal: str,
    indicators: IndicatorSnapshot,
    trend: str,
) -> Explanation:
    positive: list[str] = []
    negative: list[str] = []

    ema20, ema50, ema200 = indicators.ema_20, indicators.ema_50, indicators.ema_200
    rsi = indicators.rsi_14
    hist = indicators.macd_hist
    macd, signal = indicators.macd, indicators.macd_signal
    vol_ratio = indicators.volume_ratio

    if ema20 is not None and ema50 is not None:
        if ema20 > ema50:
            positive.append("EMA20 is above EMA50 (short-term trend supportive of upside)")
        else:
            negative.append("EMA20 is below EMA50 (short-term trend soft)")

    if ema50 is not None and ema200 is not None:
        if ema50 > ema200:
            positive.append("EMA50 is above EMA200 (intermediate trend structure bullish)")
        else:
            negative.append("EMA50 is below EMA200 (intermediate trend structure bearish)")

    if rsi is not None:
        if 50 <= rsi <= 70:
            positive.append(f"RSI ({rsi:.1f}) is in a constructive momentum band")
        elif rsi > 70:
            negative.append(f"RSI ({rsi:.1f}) is approaching/overbought")
        elif rsi < 30:
            negative.append(f"RSI ({rsi:.1f}) is oversold / weak momentum")
        elif rsi < 45:
            negative.append(f"RSI ({rsi:.1f}) is below neutral")
        else:
            positive.append(f"RSI ({rsi:.1f}) is near neutral")

    if hist is not None and macd is not None and signal is not None:
        if hist > 0 and macd > signal:
            positive.append("MACD histogram is positive (momentum expanding up)")
        elif hist < 0 and macd < signal:
            negative.append("MACD histogram is negative (momentum expanding down)")
        else:
            negative.append("MACD momentum is mixed / crossed")

    if vol_ratio is not None:
        if vol_ratio >= 1.1:
            positive.append(f"Volume is elevated vs SMA20 (ratio {vol_ratio:.2f})")
        elif vol_ratio < 0.7:
            negative.append(f"Volume is muted vs SMA20 (ratio {vol_ratio:.2f})")

    if strategy_signal == "LONG":
        positive.append("Breakout strategy signal is LONG on the latest closed setup")
    elif strategy_signal == "SHORT":
        negative.append("Breakout strategy signal is SHORT on the latest closed setup")
    else:
        negative.append("Breakout strategy signal is HOLD (no confirmed breakout)")

    if trend == "bullish":
        positive.append("Price structure vs EMAs reads bullish")
    elif trend == "bearish":
        negative.append("Price structure vs EMAs reads bearish")
    else:
        negative.append("Trend is sideways / inconclusive")

    # Align tone with prediction
    if prediction == "NEUTRAL":
        if not negative:
            negative.append("Insufficient confirmatory conditions for a directional call")
    return Explanation(positive=positive, negative=negative)
