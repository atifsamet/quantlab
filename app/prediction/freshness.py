"""
Live market freshness helpers (Phase 14).

OKX candle timestamps are candle *open* times. A candle is closed when
``now >= open + bar_duration``. Predictions prefer the latest *closed* candle
so incomplete (in-progress) bars are not used unless explicitly labeled.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import pandas as pd

from app.prediction.config import BAR_DURATION_MINUTES, STALE_AFTER_MINUTES

MarketStatus = Literal["LIVE", "STALE", "ERROR"]
PredictionFreshness = Literal["CURRENT", "STALE"]


def bar_duration(timeframe: str) -> timedelta:
    minutes = BAR_DURATION_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unknown timeframe for bar duration: {timeframe}")
    return timedelta(minutes=minutes)


def stale_threshold(timeframe: str) -> timedelta:
    minutes = STALE_AFTER_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unknown timeframe for stale threshold: {timeframe}")
    return timedelta(minutes=minutes)


def _as_utc(ts: datetime | pd.Timestamp | str) -> datetime:
    if isinstance(ts, str):
        t = pd.Timestamp(ts)
    elif isinstance(ts, pd.Timestamp):
        t = ts
    else:
        t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def candle_close_time(open_ts: datetime | pd.Timestamp | str, timeframe: str) -> datetime:
    return _as_utc(open_ts) + bar_duration(timeframe)


def is_candle_closed(
    open_ts: datetime | pd.Timestamp | str,
    timeframe: str,
    *,
    now: datetime | None = None,
) -> bool:
    now_utc = _as_utc(now or datetime.now(timezone.utc))
    return now_utc >= candle_close_time(open_ts, timeframe)


def select_closed_candles(
    ohlcv: pd.DataFrame,
    timeframe: str,
    *,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return OHLCV with incomplete trailing candle removed when present.

    Meta describes whether an in-progress candle was dropped.
    """
    if ohlcv is None or ohlcv.empty:
        raise ValueError("No candles available")

    frame = ohlcv.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    now_utc = _as_utc(now or datetime.now(timezone.utc))

    last_open = frame.iloc[-1]["timestamp"]
    dropped_incomplete = False
    incomplete_open: str | None = None
    if not is_candle_closed(last_open, timeframe, now=now_utc):
        if len(frame) < 2:
            raise ValueError("Only an incomplete candle is available; wait for close")
        incomplete_open = pd.Timestamp(last_open).isoformat()
        frame = frame.iloc[:-1].reset_index(drop=True)
        dropped_incomplete = True

    closed_open = frame.iloc[-1]["timestamp"]
    closed_close = candle_close_time(closed_open, timeframe)
    age = now_utc - closed_close
    threshold = stale_threshold(timeframe)
    market_status: MarketStatus = "LIVE" if age <= threshold else "STALE"
    prediction_status: PredictionFreshness = "CURRENT" if market_status == "LIVE" else "STALE"

    meta = {
        "dropped_incomplete_candle": dropped_incomplete,
        "incomplete_candle_open": incomplete_open,
        "prediction_candle_open": pd.Timestamp(closed_open).isoformat(),
        "prediction_candle_close": closed_close.isoformat(),
        "data_timestamp": now_utc.isoformat(),
        "last_market_update": closed_close.isoformat(),
        "market_status": market_status,
        "prediction_status": prediction_status,
        "age_seconds": int(age.total_seconds()),
        "stale_after_seconds": int(threshold.total_seconds()),
        "candle_source": "closed",
    }
    return frame, meta


def freshness_from_candle(
    candle_open: datetime | pd.Timestamp | str,
    timeframe: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compute LIVE/STALE metadata for an already-chosen closed candle."""
    now_utc = _as_utc(now or datetime.now(timezone.utc))
    close_t = candle_close_time(candle_open, timeframe)
    age = now_utc - close_t
    threshold = stale_threshold(timeframe)
    market_status: MarketStatus = "LIVE" if age <= threshold else "STALE"
    return {
        "market_status": market_status,
        "prediction_status": "CURRENT" if market_status == "LIVE" else "STALE",
        "prediction_candle_open": _as_utc(candle_open).isoformat(),
        "prediction_candle_close": close_t.isoformat(),
        "data_timestamp": now_utc.isoformat(),
        "last_market_update": close_t.isoformat(),
        "age_seconds": int(age.total_seconds()),
        "stale_after_seconds": int(threshold.total_seconds()),
        "candle_source": "closed",
    }
