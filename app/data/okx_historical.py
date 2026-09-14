"""
OKX public historical OHLCV downloader (Phase 6).

Uses GET /api/v5/market/history-candles only — no credentials, no trading.

Pagination
----------
OKX returns at most ``limit`` candles per request (max 100 for history-candles),
newest-first. We walk **backward** in time with the ``after`` cursor set to the
oldest timestamp seen so far, until we pass the requested start (including
warm-up) or the API stops returning new rows.

Timestamp convention
--------------------
All timestamps are UTC. CSV values are written as ISO-8601 with offset,
e.g. ``2025-01-01T00:00:00+00:00``.

Warm-up
-------
When ``warmup_bars`` > 0 (default = EMA 200 warm-up), candles are fetched from
``start - warmup_bars * bar_duration`` so indicators are valid at ``start``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd

from app.indicators import WARMUP_BARS
from app.market.candles import CANDLE_COLUMNS
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

HISTORY_CANDLES_PATH = "/api/v5/market/history-candles"
DEFAULT_BASE_URL = "https://www.okx.com"
DEFAULT_LIMIT = 100
DEFAULT_MAX_PAGES = 5000
DEFAULT_REQUEST_PAUSE_SECONDS = 0.12
DEFAULT_MAX_RETRIES = 4

# User-facing timeframe → OKX bar parameter
SUPPORTED_TIMEFRAMES: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
}

BAR_DURATION: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


class HistoricalDownloadError(RuntimeError):
    """Raised when historical download or validation fails."""


def normalize_timeframe(timeframe: str) -> str:
    key = timeframe.strip().lower()
    if key not in SUPPORTED_TIMEFRAMES:
        raise HistoricalDownloadError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Supported: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return key


def bar_duration(timeframe: str) -> timedelta:
    return BAR_DURATION[normalize_timeframe(timeframe)]


def parse_utc_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Parse a timestamp and normalize to UTC pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def validate_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate/normalize OHLCV and return a clean ascending frame.

    Drops duplicate timestamps (keeps first). Raises on empty/invalid OHLC.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise HistoricalDownloadError("Input must be a DataFrame")
    missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing:
        raise HistoricalDownloadError(f"Missing columns: {missing}")

    frame = df.loc[:, list(CANDLE_COLUMNS)].copy()
    if frame.empty:
        raise HistoricalDownloadError("No candles to validate")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise HistoricalDownloadError("One or more timestamps are invalid")

    before = len(frame)
    frame = frame.drop_duplicates(subset=["timestamp"], keep="first")
    dropped = before - len(frame)
    if dropped:
        logger.info("Removed %s duplicate timestamps", dropped)

    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise HistoricalDownloadError(f"Invalid numeric values in '{col}'")

    bad_hl = frame["high"] < frame["low"]
    bad_high = (frame["high"] < frame["open"]) | (frame["high"] < frame["close"])
    bad_low = (frame["low"] > frame["open"]) | (frame["low"] > frame["close"])
    bad_vol = frame["volume"] < 0
    invalid = bad_hl | bad_high | bad_low | bad_vol
    if invalid.any():
        n_bad = int(invalid.sum())
        raise HistoricalDownloadError(f"Found {n_bad} invalid OHLC/volume rows")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if not frame["timestamp"].is_monotonic_increasing:
        raise HistoricalDownloadError("Timestamps are not chronological after sort")
    return frame


def filter_date_range(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Inclusive filter on UTC timestamps."""
    out = df
    if start is not None:
        out = out[out["timestamp"] >= parse_utc_timestamp(start)]
    if end is not None:
        out = out[out["timestamp"] <= parse_utc_timestamp(end)]
    return out.reset_index(drop=True)


def save_ohlcv_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write validated OHLCV to CSV (UTC ISO-8601 timestamps)."""
    clean = validate_ohlcv_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export = clean.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Normalize +0000 → +00:00 for readability
    export["timestamp"] = export["timestamp"].str.replace(
        r"(\d{2})(\d{2})$",
        r"\1:\2",
        regex=True,
    )
    export.to_csv(path, index=False)
    logger.info("Wrote %s candles to %s", len(export), path)
    return path


def default_csv_path(
    symbol: str,
    timeframe: str,
    directory: str | Path = "data/historical",
) -> Path:
    safe_symbol = symbol.replace("/", "-").upper()
    tf = normalize_timeframe(timeframe)
    return Path(directory) / f"{safe_symbol}_{tf}.csv"


def rows_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    """Convert OKX candle arrays to an OHLCV DataFrame (unsorted OK)."""
    records = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise HistoricalDownloadError("Malformed candle row from API")
        records.append(
            {
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    if not records:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    return pd.DataFrame.from_records(records, columns=CANDLE_COLUMNS)


@dataclass(slots=True)
class DownloadResult:
    frame: pd.DataFrame
    path: Path | None
    symbol: str
    timeframe: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    fetch_start: pd.Timestamp
    candle_count: int


class OkxHistoricalDownloader:
    """
    Paginated public historical candle downloader.

    Injectable ``http_get`` is used for offline unit tests.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        limit: int = DEFAULT_LIMIT,
        max_pages: int = DEFAULT_MAX_PAGES,
        request_pause_seconds: float = DEFAULT_REQUEST_PAUSE_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep_fn: Callable[[float], None] = time.sleep,
        http_get: Callable[..., httpx.Response] | None = None,
    ) -> None:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100 for history-candles")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.limit = limit
        self.max_pages = max_pages
        self.request_pause_seconds = request_pause_seconds
        self.max_retries = max_retries
        self._sleep = sleep_fn
        self._http_get = http_get
        self._client: httpx.Client | None = None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OkxHistoricalDownloader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def download(
        self,
        symbol: str,
        timeframe: str,
        start: str | datetime | pd.Timestamp,
        end: str | datetime | pd.Timestamp,
        *,
        warmup_bars: int = WARMUP_BARS,
        output_path: str | Path | None = None,
        save: bool = True,
    ) -> DownloadResult:
        """
        Download OHLCV for ``[fetch_start, end]`` where
        ``fetch_start = start - warmup_bars * bar_size``.
        """
        tf = normalize_timeframe(timeframe)
        okx_bar = SUPPORTED_TIMEFRAMES[tf]
        start_ts = parse_utc_timestamp(start)
        end_ts = parse_utc_timestamp(end)
        if end_ts < start_ts:
            raise HistoricalDownloadError("end must be >= start")
        if warmup_bars < 0:
            raise HistoricalDownloadError("warmup_bars must be >= 0")

        fetch_start = start_ts - bar_duration(tf) * warmup_bars
        logger.info(
            "Downloading %s %s from %s to %s (fetch_start=%s, warmup_bars=%s)",
            symbol,
            tf,
            start_ts,
            end_ts,
            fetch_start,
            warmup_bars,
        )

        raw_rows = self._paginate(
            symbol=symbol,
            okx_bar=okx_bar,
            fetch_start_ms=int(fetch_start.timestamp() * 1000),
            end_ms=int(end_ts.timestamp() * 1000),
        )
        frame = rows_to_dataframe(raw_rows)
        if frame.empty:
            raise HistoricalDownloadError("API returned no candles for the range")

        frame = validate_ohlcv_frame(frame)
        frame = filter_date_range(frame, start=fetch_start, end=end_ts)
        if frame.empty:
            raise HistoricalDownloadError("No candles left after date filtering")

        path: Path | None = None
        if save:
            path = Path(output_path) if output_path else default_csv_path(symbol, tf)
            path = save_ohlcv_csv(frame, path)

        return DownloadResult(
            frame=frame,
            path=path,
            symbol=symbol.upper(),
            timeframe=tf,
            requested_start=start_ts,
            requested_end=end_ts,
            fetch_start=fetch_start,
            candle_count=len(frame),
        )

    def _paginate(
        self,
        *,
        symbol: str,
        okx_bar: str,
        fetch_start_ms: int,
        end_ms: int,
    ) -> list[list[Any]]:
        """
        Walk backward from ``end_ms`` using ``after`` until past ``fetch_start_ms``.

        Safety: stops after ``max_pages``, empty pages, or when no older cursor moves.
        """
        collected: list[list[Any]] = []
        seen_ts: set[int] = set()
        cursor_after: int | None = end_ms + 1  # exclusive upper bound helper
        pages = 0

        while pages < self.max_pages:
            pages += 1
            params: dict[str, str] = {
                "instId": symbol,
                "bar": okx_bar,
                "limit": str(self.limit),
            }
            if cursor_after is not None:
                params["after"] = str(cursor_after)

            payload = self._get_json(HISTORY_CANDLES_PATH, params)
            data = payload.get("data")
            if data is None:
                raise HistoricalDownloadError("Response missing 'data'")
            if not isinstance(data, list):
                raise HistoricalDownloadError("'data' must be a list")
            if not data:
                logger.info("Empty page at page=%s — stopping pagination", pages)
                break

            new_on_page = 0
            oldest_ms: int | None = None
            for item in data:
                if not isinstance(item, (list, tuple)) or len(item) < 6:
                    raise HistoricalDownloadError("Malformed candle in API response")
                ts_ms = int(item[0])
                if ts_ms in seen_ts:
                    continue
                # Keep candles in [fetch_start, end]
                if ts_ms > end_ms:
                    continue
                if ts_ms < fetch_start_ms:
                    # Still track oldest for cursor, but skip storing older-than-needed
                    oldest_ms = ts_ms if oldest_ms is None else min(oldest_ms, ts_ms)
                    continue
                seen_ts.add(ts_ms)
                collected.append(list(item))
                new_on_page += 1
                oldest_ms = ts_ms if oldest_ms is None else min(oldest_ms, ts_ms)

            page_oldest = min(int(item[0]) for item in data)
            if page_oldest < fetch_start_ms:
                logger.info(
                    "Reached fetch_start on page=%s (oldest=%s)",
                    pages,
                    page_oldest,
                )
                break

            if new_on_page == 0 and page_oldest >= fetch_start_ms:
                # No progress / only duplicates
                if cursor_after is not None and page_oldest >= cursor_after:
                    logger.info("Pagination cursor did not advance — stopping")
                    break

            next_cursor = page_oldest
            if cursor_after is not None and next_cursor >= cursor_after:
                logger.info("No older candles returned — stopping")
                break
            cursor_after = next_cursor

            if self.request_pause_seconds > 0:
                self._sleep(self.request_pause_seconds)

        if pages >= self.max_pages:
            raise HistoricalDownloadError(
                f"Pagination exceeded max_pages={self.max_pages} "
                "(possible infinite loop guard)"
            )

        logger.info("Pagination complete: pages=%s candles=%s", pages, len(collected))
        return collected

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._request(path, params)
            except httpx.TimeoutException as exc:
                last_error = exc
                self._backoff(attempt)
                continue
            except httpx.RequestError as exc:
                last_error = exc
                self._backoff(attempt)
                continue

            if response.status_code == 429:
                last_error = HistoricalDownloadError("HTTP 429 rate limited")
                self._backoff(attempt)
                continue

            if response.status_code >= 500:
                last_error = HistoricalDownloadError(
                    f"HTTP {response.status_code} from OKX"
                )
                self._backoff(attempt)
                continue

            if response.status_code >= 400:
                raise HistoricalDownloadError(
                    f"HTTP {response.status_code} from OKX for {path}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise HistoricalDownloadError("Response was not valid JSON") from exc

            if not isinstance(payload, dict):
                raise HistoricalDownloadError("JSON root must be an object")

            code = str(payload.get("code", ""))
            if code == "50011":
                last_error = HistoricalDownloadError("OKX rate limit code 50011")
                self._backoff(attempt)
                continue
            if code != "0":
                raise HistoricalDownloadError(
                    f"OKX API error {code}: {payload.get('msg', '')}"
                )
            return payload

        raise HistoricalDownloadError(
            f"Failed after {self.max_retries} retries: {last_error}"
        )

    def _request(self, path: str, params: dict[str, str]) -> httpx.Response:
        if self._http_get is not None:
            return self._http_get(path, params=params)

        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._client.get(path, params=params)

    def _backoff(self, attempt: int) -> None:
        delay = min(2.0 ** attempt * 0.25, 8.0)
        logger.warning("Transient failure — retry %s sleeping %.2fs", attempt, delay)
        self._sleep(delay)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download OKX public historical OHLCV candles to CSV (no trading)."
        )
    )
    parser.add_argument("--symbol", default="BTC-USDT", help="Instrument ID")
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=sorted(SUPPORTED_TIMEFRAMES),
        help="Candle timeframe",
    )
    parser.add_argument("--start", required=True, help="Evaluation start (UTC)")
    parser.add_argument("--end", required=True, help="Evaluation end (UTC)")
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=WARMUP_BARS,
        help=f"Extra bars before start for indicators (default {WARMUP_BARS})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default data/historical/SYMBOL_TF.csv)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    with OkxHistoricalDownloader() as downloader:
        result = downloader.download(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            warmup_bars=args.warmup_bars,
            output_path=args.output,
            save=True,
        )

    print(f"Symbol        : {result.symbol}")
    print(f"Timeframe     : {result.timeframe}")
    print(f"Eval start    : {result.requested_start}")
    print(f"Eval end      : {result.requested_end}")
    print(f"Fetch start   : {result.fetch_start} (includes warm-up)")
    print(f"Candles saved : {result.candle_count}")
    print(f"CSV path      : {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
