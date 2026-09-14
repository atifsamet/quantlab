"""Unit tests for the OKX public market-data client."""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

from app.config import Settings
from app.exchange.okx_client import (
    OkxAPIError,
    OkxClient,
    OkxRateLimitError,
    OkxResponseError,
)


def _settings() -> Settings:
    return Settings(
        okx_base_url="https://www.okx.com",
        okx_timeout_seconds=5.0,
        trading_enabled=False,
        live_trading_enabled=False,
    )


def _sample_candle_row(ts: str = "1700000000000") -> list[str]:
    return [
        ts,
        "42000.1",
        "42100.5",
        "41900.0",
        "42050.2",
        "12.34",
        "518000.0",
        "518000.0",
        "1",
    ]


def test_get_candles_parses_successful_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=1m&limit=2",
        json={
            "code": "0",
            "msg": "",
            "data": [
                _sample_candle_row("1700000060000"),
                _sample_candle_row("1700000000000"),
            ],
        },
    )

    with OkxClient(settings=_settings()) as client:
        frame = client.get_candles("BTC-USDT", bar="1m", limit=2)

    assert list(frame.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert len(frame) == 2
    assert frame.iloc[0]["timestamp"] < frame.iloc[1]["timestamp"]
    assert frame.iloc[0]["open"] == pytest.approx(42000.1)
    assert frame.iloc[0]["volume"] == pytest.approx(12.34)
    assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])


def test_get_candles_empty_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=1m&limit=100",
        json={"code": "0", "msg": "", "data": []},
    )

    with OkxClient(settings=_settings()) as client:
        frame = client.get_candles("BTC-USDT", bar="1m", limit=100)

    assert frame.empty
    assert list(frame.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_get_candles_api_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://www.okx.com/api/v5/market/candles?instId=BAD&bar=1m&limit=100",
        json={"code": "51001", "msg": "Instrument ID does not exist.", "data": []},
    )

    with OkxClient(settings=_settings()) as client:
        with pytest.raises(OkxAPIError) as exc_info:
            client.get_candles("BAD", bar="1m", limit=100)

    assert exc_info.value.code == "51001"


def test_get_candles_malformed_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=1m&limit=100",
        json={"code": "0", "msg": "", "data": [{"not": "a candle row"}]},
    )

    with OkxClient(settings=_settings()) as client:
        with pytest.raises(OkxResponseError):
            client.get_candles("BTC-USDT", bar="1m", limit=100)


def test_get_candles_rate_limit(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=1m&limit=100",
        status_code=429,
        json={"code": "50011", "msg": "Rate limit reached", "data": []},
    )

    with OkxClient(settings=_settings()) as client:
        with pytest.raises(OkxRateLimitError):
            client.get_candles("BTC-USDT", bar="1m", limit=100)


def test_live_trading_flag_blocks_client() -> None:
    settings = Settings(live_trading_enabled=True)
    with pytest.raises(RuntimeError):
        OkxClient(settings=settings, client=httpx.Client())
