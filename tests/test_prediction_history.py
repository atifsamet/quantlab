"""Phase 13 — prediction history, evaluation, no look-ahead tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.prediction.backfill import build_prediction_at_index, snapshot_from_result
from app.prediction.evaluate import evaluate_directional, future_slice_after_candle
from app.prediction.engine import PredictionError, validate_symbol, validate_timeframe
from app.prediction import storage
from app.prediction.stats import compute_stats


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(storage, "HISTORY_PATH", path)
    monkeypatch.setattr(storage, "PREDICTIONS_DIR", tmp_path)
    storage.clear_all_for_tests()
    yield
    storage.clear_all_for_tests()


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


def test_no_lookahead_prediction_stable_when_future_mutated() -> None:
    df = _ohlcv(260)
    n = 220
    a = build_prediction_at_index(df, n, symbol="BTC-USDT", timeframe="4h")
    mutated = df.copy()
    mutated.loc[n + 1 :, "close"] = 999_999.0
    mutated.loc[n + 1 :, "high"] = 1_000_000.0
    mutated.loc[n + 1 :, "low"] = 1.0
    b = build_prediction_at_index(mutated, n, symbol="BTC-USDT", timeframe="4h")
    assert a.prediction == b.prediction
    assert a.signal_strength == b.signal_strength
    assert a.current_price == b.current_price
    assert a.timestamp == b.timestamp
    assert a.levels.stop_loss == b.levels.stop_loss
    assert a.levels.take_profit == b.levels.take_profit


def test_duplicate_prediction_prevention() -> None:
    df = _ohlcv()
    pred = build_prediction_at_index(df, 220, symbol="BTC-USDT", timeframe="4h")
    snap = snapshot_from_result(pred)
    first = storage.append_record(snap)
    second = storage.append_record(snapshot_from_result(pred))
    assert first["id"] == second["id"]
    assert storage.count_records() == 1


def test_long_win_and_short_loss() -> None:
    # Future: rises to TP
    future = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=6, freq="4h", tz="UTC"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [100.5, 101.5, 102.5, 110, 110, 110],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5, 104.5],
            "close": [100, 101, 102, 109, 109, 109],
        }
    )
    long_win = evaluate_directional(
        side="LONG",
        entry_price=100.0,
        stop_loss=97.0,
        take_profit=106.0,
        future_ohlcv=future,
        horizon=6,
    )
    assert long_win["evaluated"] is True
    assert long_win["outcome"] == "WIN"

    short_loss = evaluate_directional(
        side="SHORT",
        entry_price=100.0,
        stop_loss=103.0,
        take_profit=94.0,
        future_ohlcv=future,
        horizon=6,
    )
    assert short_loss["outcome"] == "LOSS"


def test_same_candle_stop_loss_first() -> None:
    future = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=3, freq="4h", tz="UTC"),
            "open": [100, 100, 100],
            "high": [110, 100, 100],  # TP and SL both hit on bar 1 for LONG
            "low": [90, 100, 100],
            "close": [100, 100, 100],
        }
    )
    result = evaluate_directional(
        side="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=105.0,
        future_ohlcv=future,
        horizon=3,
    )
    assert result["outcome"] == "LOSS"
    assert result["bars_to_outcome"] == 1


def test_timeout_and_incomplete_horizon() -> None:
    future = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=6, freq="4h", tz="UTC"),
            "open": [100] * 6,
            "high": [100.5] * 6,
            "low": [99.5] * 6,
            "close": [100.2] * 6,
        }
    )
    timeout = evaluate_directional(
        side="LONG",
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        future_ohlcv=future,
        horizon=6,
    )
    assert timeout["outcome"] == "TIMEOUT"

    pending = evaluate_directional(
        side="LONG",
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        future_ohlcv=future.iloc[:2],
        horizon=6,
    )
    assert pending["evaluated"] is False
    assert pending["outcome"] == "PENDING"


def test_neutral_skipped() -> None:
    future = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=6, freq="4h", tz="UTC"),
            "open": [100] * 6,
            "high": [101] * 6,
            "low": [99] * 6,
            "close": [100] * 6,
        }
    )
    result = evaluate_directional(
        side="NEUTRAL",
        entry_price=100.0,
        stop_loss=None,
        take_profit=None,
        future_ohlcv=future,
        horizon=6,
    )
    assert result["outcome"] == "SKIPPED"


def test_future_slice_excludes_prediction_candle() -> None:
    df = _ohlcv(30, freq="1h")
    ts = df.iloc[10]["timestamp"]
    fut = future_slice_after_candle(df, ts)
    assert fut.iloc[0]["timestamp"] > pd.Timestamp(ts)


def test_invalid_symbol_timeframe() -> None:
    with pytest.raises(PredictionError):
        validate_symbol("DOGE-USDT")
    with pytest.raises(PredictionError):
        validate_timeframe("1m")


def test_stats_and_api_history_endpoints() -> None:
    df = _ohlcv()
    pred = build_prediction_at_index(df, 220, symbol="ETH-USDT", timeframe="4h")
    snap = snapshot_from_result(pred)
    snap["prediction"] = "LONG"
    snap["stop_loss"] = 50.0
    snap["take_profit"] = 200.0
    snap["entry_price"] = 100.0
    stored = storage.append_record(snap)
    # Evaluate with synthetic future via storage update
    storage.update_evaluation(
        stored["id"],
        {
            "outcome": "WIN",
            "return_pct": 2.0,
            "mfe_pct": 3.0,
            "mae_pct": -0.5,
            "bars_to_outcome": 2,
        },
    )
    stats = compute_stats()
    assert stats["overall"]["total_predictions"] == 1
    assert stats["overall"]["wins"] == 1

    from app.api.main import app

    client = TestClient(app)
    hist = client.get("/api/predictions/history")
    assert hist.status_code == 200
    assert hist.json()["count"] >= 1

    st = client.get("/api/predictions/stats")
    assert st.status_code == 200
    assert "by_signal_strength" in st.json()

    detail = client.get(f"/api/predictions/{stored['id']}")
    assert detail.status_code == 200
    assert detail.json()["record"]["id"] == stored["id"]

    bad = client.get("/api/predictions/does-not-exist")
    assert bad.status_code == 404

    perf = client.get("/api/predictions/performance/ETH-USDT/4h")
    assert perf.status_code == 200
