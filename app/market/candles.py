"""Candle / OHLCV helpers for market data."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def candles_to_dataframe(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """
    Convert OKX candle rows into a typed OHLCV DataFrame.

    OKX candle array layout:
    [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    """
    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    records: list[dict[str, Any]] = []
    for row in rows:
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

    frame = pd.DataFrame.from_records(records, columns=CANDLE_COLUMNS)
    return frame.sort_values("timestamp").reset_index(drop=True)
