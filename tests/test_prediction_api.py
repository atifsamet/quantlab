"""Phase 12 prediction engine tests (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.prediction import (
    PredictionError,
    build_prediction_from_ohlcv,
    validate_symbol,
    validate_timeframe,
)
from app.prediction.engine import clear_prediction_cache


def _ohlcv(n: int = 250, start: str = "2024-01-01", freq: str = "4h") -> pd.DataFrame:
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


def test_validate_symbol() -> None:
    assert validate_symbol("btc-usdt") == "BTC-USDT"
    with pytest.raises(PredictionError):
        validate_symbol("DOGE-USDT")


def test_validate_timeframe() -> None:
    assert validate_timeframe("4H") == "4h"
    with pytest.raises(PredictionError):
        validate_timeframe("1m")


def test_prediction_from_ohlcv_deterministic() -> None:
    df = _ohlcv()
    a = build_prediction_from_ohlcv(df, symbol="BTC-USDT", timeframe="4h")
    b = build_prediction_from_ohlcv(df, symbol="BTC-USDT", timeframe="4h")
    assert a.prediction == b.prediction
    assert a.signal_strength == b.signal_strength
    assert a.ml_probability is None
    assert "NOT" in a.signal_strength_label.upper() or "not" in a.signal_strength_label
    assert a.research_conclusion
    assert 0 <= a.signal_strength <= 100


def test_insufficient_candles() -> None:
    with pytest.raises(PredictionError):
        build_prediction_from_ohlcv(_ohlcv(50), symbol="BTC-USDT", timeframe="4h")


def test_api_health_and_invalid_symbol() -> None:
    from app.api.main import app

    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["live_trading_enabled"] is False
    assert body["mode"] == "RESEARCH MODE"

    bad = client.get("/api/prediction/DOGE-USDT/4h")
    assert bad.status_code == 400

    bad_tf = client.get("/api/prediction/BTC-USDT/1m")
    assert bad_tf.status_code == 400


def test_cors_allows_production_frontend_origins() -> None:
    from app.api.main import ALLOWED_CORS_ORIGINS, app

    assert "https://quantlab-three.vercel.app" in ALLOWED_CORS_ORIGINS
    assert "https://quantlabapp.com" in ALLOWED_CORS_ORIGINS
    assert "*" not in ALLOWED_CORS_ORIGINS

    client = TestClient(app)
    for origin in ("https://quantlab-three.vercel.app", "https://quantlabapp.com"):
        preflight = client.options(
            "/api/health",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )
        assert preflight.status_code in (200, 204)
        assert preflight.headers.get("access-control-allow-origin") == origin

        get_health = client.get("/api/health", headers={"Origin": origin})
        assert get_health.status_code == 200
        assert get_health.headers.get("access-control-allow-origin") == origin


def test_api_research_and_artifacts() -> None:
    from app.api.main import app

    client = TestClient(app)
    research = client.get("/api/research")
    assert research.status_code == 200
    data = research.json()
    assert data["conclusion"]
    assert len(data["timeline"]) >= 11
    assert data["live_trading_enabled"] is False

    strategies = client.get("/api/strategies")
    assert strategies.status_code == 200

    mc = client.get("/api/monte-carlo")
    assert mc.status_code == 200

    analytics = client.get("/api/analytics")
    assert analytics.status_code == 200


def test_clear_cache() -> None:
    clear_prediction_cache()
