"""Offline unit tests for OKX historical downloader (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.data.okx_historical import (
    HistoricalDownloadError,
    OkxHistoricalDownloader,
    filter_date_range,
    normalize_timeframe,
    parse_utc_timestamp,
    rows_to_dataframe,
    save_ohlcv_csv,
    validate_ohlcv_frame,
)


def _candle(ts_ms: int, price: float = 100.0) -> list[str]:
    return [
        str(ts_ms),
        str(price),
        str(price + 1),
        str(price - 1),
        str(price),
        "1.5",
        "0",
        "0",
        "1",
    ]


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def test_normalize_timeframe() -> None:
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("15m") == "15m"
    assert normalize_timeframe("4h") == "4h"
    with pytest.raises(HistoricalDownloadError):
        normalize_timeframe("2h")


def test_parse_utc_timestamp() -> None:
    ts = parse_utc_timestamp("2025-01-01")
    assert str(ts.tzinfo) == "UTC"
    assert ts.hour == 0


def test_rows_to_dataframe_and_validate() -> None:
    rows = [_candle(1_700_000_000_000), _candle(1_700_000_060_000, 101.0)]
    frame = rows_to_dataframe(rows)
    clean = validate_ohlcv_frame(frame)
    assert list(clean.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert clean["timestamp"].is_monotonic_increasing


def test_duplicate_removal_and_sort() -> None:
    rows = [
        _candle(1_700_000_060_000, 101.0),
        _candle(1_700_000_000_000, 100.0),
        _candle(1_700_000_000_000, 100.5),  # duplicate ts
    ]
    frame = rows_to_dataframe(rows)
    clean = validate_ohlcv_frame(frame)
    assert len(clean) == 2
    assert clean.iloc[0]["close"] == pytest.approx(100.0)


def test_ohlcv_validation_rejects_bad_high_low() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 100.0,
                "high": 90.0,
                "low": 95.0,
                "close": 96.0,
                "volume": 1.0,
            }
        ]
    )
    with pytest.raises(HistoricalDownloadError, match="invalid"):
        validate_ohlcv_frame(frame)


def test_date_filtering() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.1, 2.1, 3.1],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    filtered = filter_date_range(
        frame,
        start="2024-01-01T01:00:00Z",
        end="2024-01-01T02:00:00Z",
    )
    assert len(filtered) == 2
    assert filtered.iloc[0]["close"] == pytest.approx(2.1)


def test_csv_write_and_load(tmp_path: Path) -> None:
    frame = validate_ohlcv_frame(
        rows_to_dataframe([_candle(1_700_000_000_000), _candle(1_700_000_060_000)])
    )
    path = tmp_path / "BTC-USDT_1h.csv"
    save_ohlcv_csv(frame, path)
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert len(loaded) == 2


def test_malformed_api_response() -> None:
    def http_get(path: str, params: dict) -> _FakeResponse:
        return _FakeResponse({"code": "0", "data": [{"bad": True}]})

    downloader = OkxHistoricalDownloader(http_get=http_get, request_pause_seconds=0)
    with pytest.raises(HistoricalDownloadError, match="Malformed"):
        downloader.download(
            "BTC-USDT",
            "1h",
            start="2024-01-01",
            end="2024-01-02",
            warmup_bars=0,
            save=False,
        )


def test_empty_api_response() -> None:
    def http_get(path: str, params: dict) -> _FakeResponse:
        return _FakeResponse({"code": "0", "msg": "", "data": []})

    downloader = OkxHistoricalDownloader(http_get=http_get, request_pause_seconds=0)
    with pytest.raises(HistoricalDownloadError, match="no candles"):
        downloader.download(
            "BTC-USDT",
            "1h",
            start="2024-01-01",
            end="2024-01-02",
            warmup_bars=0,
            save=False,
        )


def test_retry_on_transient_then_success() -> None:
    calls = {"n": 0}

    def http_get(path: str, params: dict) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse({"code": "50011", "msg": "rate", "data": []})
        # One page covering the range
        end = int(pd.Timestamp("2024-01-01 03:00:00+00:00").timestamp() * 1000)
        start = int(pd.Timestamp("2024-01-01 00:00:00+00:00").timestamp() * 1000)
        data = [_candle(ts) for ts in range(end, start - 1, -3_600_000)]
        return _FakeResponse({"code": "0", "data": data})

    sleeps: list[float] = []
    downloader = OkxHistoricalDownloader(
        http_get=http_get,
        request_pause_seconds=0,
        sleep_fn=sleeps.append,
        max_retries=3,
    )
    result = downloader.download(
        "BTC-USDT",
        "1h",
        start="2024-01-01",
        end="2024-01-01 03:00:00+00:00",
        warmup_bars=0,
        save=False,
    )
    assert calls["n"] >= 2
    assert sleeps  # backoff used
    assert result.candle_count >= 1


def test_pagination_collects_multiple_pages_and_stops() -> None:
    # Build 250 hourly candles ending at 2024-01-10
    end = pd.Timestamp("2024-01-10 00:00:00+00:00")
    all_ts = [int((end - pd.Timedelta(hours=i)).timestamp() * 1000) for i in range(250)]
    # newest first as OKX returns
    pages_served = {"n": 0}

    def http_get(path: str, params: dict) -> _FakeResponse:
        pages_served["n"] += 1
        after = int(params["after"])
        older = [ts for ts in all_ts if ts < after]
        older_sorted = sorted(older, reverse=True)[: int(params["limit"])]
        data = [_candle(ts, price=100 + (ts % 7)) for ts in older_sorted]
        return _FakeResponse({"code": "0", "data": data})

    downloader = OkxHistoricalDownloader(
        http_get=http_get,
        limit=100,
        request_pause_seconds=0,
        sleep_fn=lambda _s: None,
        max_pages=20,
    )
    result = downloader.download(
        "BTC-USDT",
        "1h",
        start="2024-01-01",
        end="2024-01-10",
        warmup_bars=0,
        save=False,
    )
    assert pages_served["n"] >= 2
    assert result.candle_count == len(result.frame)
    assert result.frame["timestamp"].is_monotonic_increasing
    # No infinite loop
    assert pages_served["n"] < 20


def test_max_pages_guard() -> None:
    def http_get(path: str, params: dict) -> _FakeResponse:
        # Always return the same "new" page that never reaches start
        after = int(params.get("after", 10_000_000))
        ts = after - 1
        return _FakeResponse({"code": "0", "data": [_candle(ts)]})

    downloader = OkxHistoricalDownloader(
        http_get=http_get,
        request_pause_seconds=0,
        sleep_fn=lambda _s: None,
        max_pages=3,
    )
    with pytest.raises(HistoricalDownloadError, match="max_pages"):
        downloader.download(
            "BTC-USDT",
            "1h",
            start="2020-01-01",
            end="2024-01-01",
            warmup_bars=0,
            save=False,
        )


def test_httpx_timeout_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def http_get(path: str, params: dict):
        calls["n"] += 1
        raise httpx.TimeoutException("timeout")

    downloader = OkxHistoricalDownloader(
        http_get=http_get,
        request_pause_seconds=0,
        sleep_fn=lambda _s: None,
        max_retries=2,
    )
    with pytest.raises(HistoricalDownloadError, match="retries"):
        downloader.download(
            "BTC-USDT",
            "1h",
            start="2024-01-01",
            end="2024-01-02",
            warmup_bars=0,
            save=False,
        )
    assert calls["n"] == 2
