"""Prepare historical CSV data for a warm-up-aware backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.backtest.engine import load_ohlcv_csv
from app.data.okx_historical import filter_date_range, parse_utc_timestamp
from app.indicators import WARMUP_BARS, add_indicators
from app.strategy import generate_signals


def prepare_backtest_frame(
    csv_path: str | Path,
    *,
    eval_start: str | pd.Timestamp | None = None,
    eval_end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Load CSV, compute indicators/signals on the **full** series (warm-up),
    then return only the evaluation window with indicators already populated.

    This prevents treating the first EMA-200 bars of the evaluation period as
    if the indicator were already warmed up.
    """
    raw = load_ohlcv_csv(csv_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    if eval_start is None:
        start = raw["timestamp"].iloc[0]
    else:
        start = parse_utc_timestamp(eval_start)
    if eval_end is None:
        end = raw["timestamp"].iloc[-1]
    else:
        end = parse_utc_timestamp(eval_end)

    if len(raw) < WARMUP_BARS:
        # Still allow run, but indicators will be largely NaN → HOLD.
        pass

    with_indicators = add_indicators(raw)
    signaled = generate_signals(with_indicators)
    evaluation = filter_date_range(signaled, start=start, end=end)
    if evaluation.empty:
        raise ValueError("Evaluation window contains no candles")
    return evaluation.reset_index(drop=True), start, end
