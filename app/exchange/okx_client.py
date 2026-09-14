"""OKX public REST client for market data (Phase 1 — no trading)."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from app.config import Settings, get_settings
from app.market.candles import CANDLE_COLUMNS, candles_to_dataframe
from app.utils.logger import get_logger

logger = get_logger(__name__)

CANDLES_PATH = "/api/v5/market/candles"


class OkxError(Exception):
    """Base error for OKX client failures."""


class OkxHTTPError(OkxError):
    """Raised when the HTTP transport fails or returns a non-success status."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OkxAPIError(OkxError):
    """Raised when OKX returns a business-level error (code != '0')."""

    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"OKX API error {code}: {msg}")
        self.code = code
        self.msg = msg


class OkxRateLimitError(OkxHTTPError):
    """Raised when OKX rate-limits the request (HTTP 429 or code 50011)."""


class OkxResponseError(OkxError):
    """Raised when the response body is missing or malformed."""


class OkxClient:
    """
    Public-only OKX REST client.

    Does not place orders, withdraw funds, or sign private requests.
    Credentials are accepted in Settings for future phases but are unused here.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.assert_trading_disabled()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.settings.okx_base_url.rstrip("/"),
            timeout=self.settings.okx_timeout_seconds,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        """Close the underlying HTTP client if this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OkxClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_candles(
        self,
        inst_id: str,
        bar: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch candlestick data for an instrument.

        Uses OKX public endpoint GET /api/v5/market/candles.
        No API credentials are required.

        Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
        sorted ascending by timestamp.
        """
        if not inst_id:
            raise ValueError("inst_id is required")
        if limit < 1 or limit > 300:
            raise ValueError("limit must be between 1 and 300")

        params = {
            "instId": inst_id,
            "bar": bar,
            "limit": str(limit),
        }
        logger.info(
            "Requesting candles inst_id=%s bar=%s limit=%s",
            inst_id,
            bar,
            limit,
        )

        payload = self._get_json(CANDLES_PATH, params=params)
        rows = self._extract_candle_rows(payload)
        frame = candles_to_dataframe(rows)
        logger.info("Received %s candles for %s", len(frame), inst_id)
        return frame

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            response = self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise OkxHTTPError(f"Request timed out calling {path}") from exc
        except httpx.RequestError as exc:
            raise OkxHTTPError(f"HTTP request failed for {path}: {exc}") from exc

        if response.status_code == 429:
            raise OkxRateLimitError(
                "OKX rate limit reached (HTTP 429). Slow down and retry later.",
                status_code=429,
            )

        if response.status_code >= 400:
            raise OkxHTTPError(
                f"HTTP {response.status_code} from OKX for {path}",
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OkxResponseError("OKX response was not valid JSON") from exc

        if not isinstance(payload, dict):
            raise OkxResponseError("OKX response JSON root must be an object")

        code = str(payload.get("code", ""))
        msg = str(payload.get("msg", ""))

        if code == "50011":
            raise OkxRateLimitError(
                f"OKX rate limit reached (API code 50011): {msg}",
                status_code=response.status_code,
            )

        if code != "0":
            raise OkxAPIError(code=code or "unknown", msg=msg or "unknown error")

        return payload

    @staticmethod
    def _extract_candle_rows(payload: dict[str, Any]) -> list[list[Any]]:
        data = payload.get("data")
        if data is None:
            raise OkxResponseError("OKX response missing 'data' field")
        if not isinstance(data, list):
            raise OkxResponseError("OKX 'data' field must be a list")
        if not data:
            return []

        rows: list[list[Any]] = []
        for index, item in enumerate(data):
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                raise OkxResponseError(
                    f"Malformed candle at index {index}: expected list with "
                    f"at least 6 fields, got {type(item).__name__}"
                )
            rows.append(list(item))
        return rows


def empty_candles_frame() -> pd.DataFrame:
    """Return an empty OHLCV frame with the standard columns."""
    return pd.DataFrame(columns=CANDLE_COLUMNS)
