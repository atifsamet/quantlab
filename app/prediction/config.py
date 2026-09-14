"""QuantLab prediction config (analytical only — no trading)."""

from __future__ import annotations

SUPPORTED_SYMBOLS: tuple[str, ...] = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT")
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("15m", "1h", "4h")

# OKX public candle bar codes
OKX_BAR: dict[str, str] = {
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
}

# Enough bars for EMA200 warm-up within OKX candles limit (max 300)
CANDLE_FETCH_LIMIT = 300

CACHE_TTL_SECONDS = 60.0

# Phase 13 — prediction history / evaluation
ENGINE_VERSION = "prediction-v1"

# Forward candles used to resolve WIN / LOSS / TIMEOUT (after prediction candle N).
EVAL_HORIZONS: dict[str, int] = {
    "15m": 24,
    "1h": 12,
    "4h": 6,
}

# Backfill lookback windows (days) and sampling stride (bars between snapshots).
BACKFILL_WINDOWS_DAYS: dict[str, int] = {
    "15m": 90,
    "1h": 180,
    "4h": 365,
}
BACKFILL_STRIDE: dict[str, int] = {
    "15m": 16,  # every ~4h
    "1h": 6,  # every 6h
    "4h": 3,  # every 12h
}

SIGNAL_STRENGTH_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 39, "0-39"),
    (40, 59, "40-59"),
    (60, 79, "60-79"),
    (80, 100, "80-100"),
)

MIN_SAMPLE_FOR_RATE = 20

# Phase 14 — live market freshness (minutes after latest closed candle close)
STALE_AFTER_MINUTES: dict[str, int] = {
    "15m": 20,
    "1h": 75,
    "4h": 255,
}

# Candle open → close duration
BAR_DURATION_MINUTES: dict[str, int] = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

MARKET_DATA_DISCLAIMER = (
    "Market data is retrieved from public exchange endpoints and may be delayed "
    "or temporarily unavailable."
)

SIGNAL_STRENGTH_TOOLTIP = (
    "Analytical signal strength derived from the current model inputs. "
    "It is not a probability of future return."
)

RESEARCH_CONCLUSION = "NO ROBUST EDGE DETECTED"

DISCLAIMER = (
    "Predictions are analytical outputs from historical indicators and research "
    "models. They do not guarantee future performance. QuantLab does not place "
    "orders. LIVE TRADING is disabled."
)
