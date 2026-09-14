"""Flexible indicator computation for research parameter sets."""

from __future__ import annotations

import pandas as pd

from app.indicators.technical import atr, ema, macd, prepare_ohlcv, rsi
from app.research.params import ResearchParams, validate_research_params

# Columns required at evaluation start for research frames.
RESEARCH_REQUIRED_COLUMNS = (
    "ema_fast",
    "ema_med",
    "ema_slow",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
)


def add_research_indicators(df: pd.DataFrame, params: ResearchParams) -> pd.DataFrame:
    """
    Compute indicator columns for a research parameter set.

    Also mirrors values into Phase-2/3 names when periods match defaults, and
    always writes ATR into ``atr_14`` for the Phase-5 risk engine.
    """
    validate_research_params(params)
    prepared = prepare_ohlcv(df)
    out = prepared.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["ema_fast"] = ema(close, params.ema_fast)
    out["ema_med"] = ema(close, params.ema_med)
    out["ema_slow"] = ema(close, params.ema_slow)
    out["rsi"] = rsi(close, params.rsi_period)

    macd_line, signal_line, hist = macd(
        close, params.macd_fast, params.macd_slow, params.macd_signal
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["atr_14"] = atr(high, low, close, params.atr_period)

    # Compatibility aliases for baseline diagnostics / warm-up checks.
    out["ema_20"] = out["ema_fast"] if params.ema_fast == 20 else ema(close, 20)
    out["ema_50"] = out["ema_med"] if params.ema_med == 50 else ema(close, 50)
    out["ema_200"] = out["ema_slow"] if params.ema_slow == 200 else ema(close, 200)
    out["rsi_14"] = out["rsi"] if params.rsi_period == 14 else rsi(close, 14)

    # Prior highs/lows for breakout (shifted so current bar does not peek).
    out["prior_high"] = high.rolling(params.breakout_lookback).max().shift(1)
    out["prior_low"] = low.rolling(params.breakout_lookback).min().shift(1)
    return out
