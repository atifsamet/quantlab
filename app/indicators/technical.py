"""
Deterministic technical indicators over OHLCV candle DataFrames.

Warm-up periods (bars needed before the first non-NaN value):

* EMA n            → n bars (seeded with SMA of the first n closes)
* RSI 14           → 15 bars (1 seed close + 14 Wilder periods)
* MACD line        → 26 bars (slow EMA warm-up)
* MACD signal/hist → 34 bars (26 + 9 - 1)
* ATR 14           → 15 bars (1 prior close + 14 Wilder periods)
* Volume SMA 20    → 20 bars

The longest warm-up in this module is EMA 200 → WARMUP_BARS = 200.
Until enough history exists, values are NaN — never fabricated.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

INDICATOR_COLUMNS = [
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "volume_sma_20",
]

# Longest warm-up among configured indicators (EMA 200).
WARMUP_BARS = 200

REQUIRED_OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


class IndicatorDataError(ValueError):
    """Raised when candle data is invalid for indicator calculation."""


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average with SMA seed.

    The first non-NaN value appears at index ``span - 1`` and equals the
    simple mean of the first ``span`` observations. Subsequent values use
    the standard multiplier ``2 / (span + 1)``.
    """
    if span < 1:
        raise ValueError("span must be >= 1")

    values = series.astype(float).to_numpy(copy=True)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < span:
        return pd.Series(out, index=series.index, dtype=float)

    alpha = 2.0 / (span + 1.0)
    out[span - 1] = values[:span].mean()
    for i in range(span, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index, dtype=float)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder smoothing.

    First average gain/loss uses a simple mean over ``period`` changes;
    subsequent values use Wilder's recursive smoothing.
    """
    if period < 1:
        raise ValueError("period must be >= 1")

    close_arr = close.astype(float).to_numpy(copy=True)
    n = len(close_arr)
    out = np.full(n, np.nan, dtype=float)
    if n <= period:
        return pd.Series(out, index=close.index, dtype=float)

    deltas = np.diff(close_arr)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    return pd.Series(out, index=close.index, dtype=float)


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    if not (1 <= fast < slow):
        raise ValueError("require 1 <= fast < slow")
    if signal < 1:
        raise ValueError("signal must be >= 1")

    macd_line = ema(close, fast) - ema(close, slow)
    # Signal EMA is computed on the MACD line; leading NaNs stay NaN until
    # there are `signal` contiguous MACD values after the slow EMA warm-up.
    signal_line = _ema_on_series_with_leading_nans(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _ema_on_series_with_leading_nans(series: pd.Series, span: int) -> pd.Series:
    """EMA that ignores leading NaNs, then applies the standard SMA-seeded EMA."""
    values = series.astype(float).to_numpy(copy=True)
    out = np.full(len(values), np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) < span:
        return pd.Series(out, index=series.index, dtype=float)

    start = int(valid_idx[0])
    window = values[start : start + span]
    if np.isnan(window).any():
        # Gap inside the warm-up window — refuse to invent values.
        return pd.Series(out, index=series.index, dtype=float)

    alpha = 2.0 / (span + 1.0)
    seed_pos = start + span - 1
    out[seed_pos] = window.mean()
    for i in range(seed_pos + 1, len(values)):
        if np.isnan(values[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
            continue
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index, dtype=float)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range for ATR."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    if period < 1:
        raise ValueError("period must be >= 1")

    tr = true_range(high.astype(float), low.astype(float), close.astype(float))
    tr_arr = tr.to_numpy(dtype=float)
    out = np.full(len(tr_arr), np.nan, dtype=float)

    if len(tr_arr) <= period:
        return pd.Series(out, index=close.index, dtype=float)

    # First ATR is the mean of the first `period` true ranges after the
    # initial bar (which has no previous close).
    out[period] = np.mean(tr_arr[1 : period + 1])
    for i in range(period + 1, len(tr_arr)):
        out[i] = (out[i - 1] * (period - 1) + tr_arr[i]) / period
    return pd.Series(out, index=close.index, dtype=float)


def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return volume.astype(float).rolling(window=window, min_periods=window).mean()


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise IndicatorDataError(f"Missing required columns: {missing}")


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize an OHLCV frame before indicator calculation.

    * Requires OHLCV columns
    * Rejects empty frames
    * Rejects duplicate timestamps
    * Rejects non-finite / missing OHLC/volume values
    * Sorts by timestamp ascending
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise IndicatorDataError("Input must be a pandas DataFrame")

    _require_columns(df, REQUIRED_OHLCV)

    if df.empty:
        raise IndicatorDataError("Candle DataFrame is empty")

    frame = df.loc[:, list(REQUIRED_OHLCV)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    if frame["timestamp"].isna().any():
        raise IndicatorDataError("One or more timestamps are invalid / missing")

    if frame["timestamp"].duplicated().any():
        raise IndicatorDataError("Duplicate timestamps are not allowed")

    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise IndicatorDataError(
                f"Column '{col}' contains missing or non-numeric values"
            )
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise IndicatorDataError(f"Column '{col}' contains non-finite values")

    if (frame["high"] < frame["low"]).any():
        raise IndicatorDataError("Found rows where high < low")

    return frame.sort_values("timestamp").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``df`` with Phase 2 indicator columns appended.

    Original OHLCV columns are preserved. Indicator cells remain NaN until
    their warm-up period is satisfied (see module docstring / WARMUP_BARS).
    """
    prepared = prepare_ohlcv(df)
    out = prepared.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["ema_200"] = ema(close, 200)
    out["rsi_14"] = rsi(close, 14)

    macd_line, signal_line, histogram = macd(close, 12, 26, 9)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = histogram

    out["atr_14"] = atr(high, low, close, 14)
    out["volume_sma_20"] = volume_sma(volume, 20)
    return out
