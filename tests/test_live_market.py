"""Phase 14 — live market freshness, closed candles, refresh behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.prediction.engine import PredictionError, build_prediction_from_ohlcv, clear_prediction_cache
from app.prediction.freshness import (
    candle_close_time,
    is_candle_closed,
    select_closed_candles,
)
from app.prediction.history import refresh_and_store, save_prediction_snapshot
from app.prediction import storage


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(storage, "HISTORY_PATH", path)
    monkeypatch.setattr(storage, "PREDICTIONS_DIR", tmp_path)
    storage.clear_all_for_tests()
    clear_prediction_cache()
    yield
    storage.clear_all_for_tests()
    clear_prediction_cache()


def _ohlcv(n: int = 260, start: str = "2024-01-01", freq: str = "4h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(np.ones(n) * 0.05)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )


def test_closed_candle_detection_drops_incomplete() -> None:
    df = _ohlcv(250, start="2026-09-01", freq="4h")
    # Last open = 2026-09-01 + 249*4h. Force now inside last candle.
    last_open = pd.Timestamp(df.iloc[-1]["timestamp"])
    now = last_open + timedelta(hours=1)  # incomplete 4h candle
    closed, meta = select_closed_candles(df, "4h", now=now)
    assert len(closed) == len(df) - 1
    assert meta["dropped_incomplete_candle"] is True
    assert meta["candle_source"] == "closed"
    assert is_candle_closed(closed.iloc[-1]["timestamp"], "4h", now=now)


def test_stale_detection() -> None:
    open_ts = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    close_t = candle_close_time(open_ts, "15m")
    frame = pd.DataFrame(
        {
            "timestamp": [open_ts - timedelta(minutes=15), open_ts],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    live_now = close_t + timedelta(minutes=5)
    _, meta_live = select_closed_candles(frame, "15m", now=live_now)
    assert meta_live["market_status"] == "LIVE"
    assert meta_live["prediction_status"] == "CURRENT"

    stale_now = close_t + timedelta(minutes=25)
    _, meta_stale = select_closed_candles(frame, "15m", now=stale_now)
    assert meta_stale["market_status"] == "STALE"
    assert meta_stale["prediction_status"] == "STALE"


def test_prediction_ignores_future_incomplete_and_look_ahead() -> None:
    df = _ohlcv(260)
    n = 220
    base = df.iloc[: n + 1].copy()
    # Append a fake "live incomplete" future-looking row that must not affect prediction
    last = base.iloc[-1]
    incomplete = last.copy()
    incomplete["timestamp"] = pd.Timestamp(last["timestamp"]) + timedelta(hours=4)
    incomplete["close"] = 99999.0
    incomplete["high"] = 100000.0
    with_incomplete = pd.concat([base, pd.DataFrame([incomplete])], ignore_index=True)

    # now = just after last closed open of base (inside incomplete candle)
    now = pd.Timestamp(incomplete["timestamp"]) + timedelta(hours=1)
    a = build_prediction_from_ohlcv(base, symbol="BTC-USDT", timeframe="4h", now=now.to_pydatetime())
    b = build_prediction_from_ohlcv(
        with_incomplete, symbol="BTC-USDT", timeframe="4h", now=now.to_pydatetime()
    )
    assert a.prediction == b.prediction
    assert a.signal_strength == b.signal_strength
    assert a.current_price == b.current_price
    assert a.live is not None
    assert a.live.candle_source == "closed"


def test_duplicate_refresh_same_candle(monkeypatch) -> None:
    df = _ohlcv(260)
    now = pd.Timestamp(df.iloc[-1]["timestamp"]) + timedelta(hours=5)

    def fake_fetch(symbol, timeframe, *, client=None):
        return df.copy()

    monkeypatch.setattr("app.prediction.history.predict", lambda *a, **k: build_prediction_from_ohlcv(
        df, symbol="BTC-USDT", timeframe="4h", now=now.to_pydatetime()
    ))
    # Also patch evaluate_pending to no-op network
    monkeypatch.setattr(
        "app.prediction.history.evaluate_pending",
        lambda **kwargs: {"checked": 0, "newly_evaluated": 0, "still_pending": 0, "skipped_neutral": 0},
    )

    first = refresh_and_store("BTC-USDT", "4h")
    second = refresh_and_store("BTC-USDT", "4h")
    assert first["snapshot_created"] is True
    assert second["snapshot_created"] is False
    assert first["snapshot"]["id"] == second["snapshot"]["id"]
    assert storage.count_records() == 1


def test_okx_failure_refresh() -> None:
    from app.api.main import app

    client = TestClient(app)

    with patch("app.prediction.history.predict", side_effect=RuntimeError("OKX down")):
        r = client.post("/api/predictions/refresh?symbol=BTC-USDT&timeframe=4h")
        assert r.status_code == 502


def test_invalid_symbol_timeframe_refresh() -> None:
    from app.api.main import app

    client = TestClient(app)
    bad = client.post("/api/predictions/refresh?symbol=DOGE-USDT&timeframe=4h")
    assert bad.status_code == 400
    bad_tf = client.post("/api/predictions/refresh?symbol=BTC-USDT&timeframe=1m")
    assert bad_tf.status_code == 400


def test_persist_snapshot_fields() -> None:
    df = _ohlcv(260)
    now = pd.Timestamp(df.iloc[-1]["timestamp"]) + timedelta(hours=5)
    pred = build_prediction_from_ohlcv(df, symbol="ETH-USDT", timeframe="4h", now=now.to_pydatetime())
    snap = save_prediction_snapshot(pred)
    assert snap["engine_version"]
    assert snap["candle_timestamp"] == pred.timestamp
    assert snap["evaluated"] is False
    again = save_prediction_snapshot(pred)
    assert again["id"] == snap["id"]
