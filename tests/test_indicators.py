"""Unit tests for Phase 2 technical indicators (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.indicators.technical import (
    INDICATOR_COLUMNS,
    WARMUP_BARS,
    IndicatorDataError,
    add_indicators,
    atr,
    ema,
    macd,
    rsi,
    volume_sma,
)
from app.market.candles import CANDLE_COLUMNS


def _ohlcv_from_closes(closes: list[float], start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    rows = []
    for i, close in enumerate(closes):
        # Synthetic bars: high/low bracket the close; volume = i + 1
        rows.append(
            {
                "timestamp": pd.to_datetime(start_ms + i * 60_000, unit="ms", utc=True),
                "open": float(close),
                "high": float(close) + 1.0,
                "low": float(close) - 1.0,
                "close": float(close),
                "volume": float(i + 1),
            }
        )
    return pd.DataFrame(rows, columns=CANDLE_COLUMNS)


def test_ema_sma_seed_and_recursion() -> None:
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(closes, span=3)

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx((1 + 2 + 3) / 3)
    alpha = 2.0 / (3 + 1)
    expected_3 = alpha * 4.0 + (1 - alpha) * result.iloc[2]
    expected_4 = alpha * 5.0 + (1 - alpha) * expected_3
    assert result.iloc[3] == pytest.approx(expected_3)
    assert result.iloc[4] == pytest.approx(expected_4)


def test_ema_insufficient_data_all_nan() -> None:
    result = ema(pd.Series([1.0, 2.0]), span=5)
    assert result.isna().all()


def test_rsi_known_values() -> None:
    # Strictly rising prices → RSI approaches 100 after warm-up.
    closes = pd.Series([float(i) for i in range(1, 30)])
    result = rsi(closes, period=14)
    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(100.0)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_flat_market() -> None:
    closes = pd.Series([10.0] * 20)
    result = rsi(closes, period=14)
    assert result.iloc[14] == pytest.approx(50.0)


def test_macd_structure_and_warmup() -> None:
    closes = pd.Series(np.linspace(100, 120, 60))
    macd_line, signal, hist = macd(closes, fast=12, slow=26, signal=9)

    assert macd_line.iloc[:25].isna().all()
    assert not np.isnan(macd_line.iloc[25])
    # Signal needs 9 MACD points starting at index 25 → first at index 33
    assert signal.iloc[:33].isna().all()
    assert not np.isnan(signal.iloc[33])
    assert hist.iloc[33] == pytest.approx(macd_line.iloc[33] - signal.iloc[33])


def test_atr_known_simple_case() -> None:
    # Constant 2-point range, flat closes after first bar → TR mostly 2.
    n = 20
    high = pd.Series([12.0] * n)
    low = pd.Series([10.0] * n)
    close = pd.Series([11.0] * n)
    result = atr(high, low, close, period=14)

    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(2.0)
    assert result.iloc[-1] == pytest.approx(2.0)


def test_volume_sma() -> None:
    volume = pd.Series([float(i) for i in range(1, 26)])
    result = volume_sma(volume, window=20)
    assert result.iloc[:19].isna().all()
    assert result.iloc[19] == pytest.approx(sum(range(1, 21)) / 20)
    assert result.iloc[20] == pytest.approx(sum(range(2, 22)) / 20)


def test_add_indicators_output_columns() -> None:
    df = _ohlcv_from_closes([100 + i * 0.1 for i in range(220)])
    out = add_indicators(df)

    for col in CANDLE_COLUMNS:
        assert col in out.columns
    for col in INDICATOR_COLUMNS:
        assert col in out.columns

    assert list(out.columns[:6]) == CANDLE_COLUMNS
    # EMA 200 should be populated by bar 200 (index 199)
    assert out["ema_200"].iloc[:199].isna().all()
    assert not np.isnan(out["ema_200"].iloc[199])
    assert not np.isnan(out["rsi_14"].iloc[-1])
    assert not np.isnan(out["macd"].iloc[-1])
    assert not np.isnan(out["atr_14"].iloc[-1])
    assert not np.isnan(out["volume_sma_20"].iloc[-1])


def test_insufficient_data_leaves_long_ema_nan() -> None:
    df = _ohlcv_from_closes([100.0 + i for i in range(50)])
    out = add_indicators(df)
    assert out["ema_50"].notna().any()
    assert out["ema_200"].isna().all()
    assert WARMUP_BARS == 200


def test_unsorted_timestamps_are_sorted() -> None:
    df = _ohlcv_from_closes([10.0, 11.0, 12.0, 13.0, 14.0] + [15.0] * 30)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = add_indicators(shuffled)
    assert out["timestamp"].is_monotonic_increasing
    assert out["close"].iloc[0] == pytest.approx(10.0)


def test_duplicate_timestamps_raise() -> None:
    df = _ohlcv_from_closes([10.0, 11.0, 12.0])
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]
    with pytest.raises(IndicatorDataError, match="Duplicate"):
        add_indicators(df)


def test_missing_values_raise() -> None:
    df = _ohlcv_from_closes([10.0, 11.0, 12.0])
    df.loc[1, "close"] = np.nan
    with pytest.raises(IndicatorDataError, match="close"):
        add_indicators(df)


def test_invalid_non_finite_raise() -> None:
    df = _ohlcv_from_closes([10.0, 11.0, 12.0])
    df.loc[1, "volume"] = np.inf
    with pytest.raises(IndicatorDataError, match="volume"):
        add_indicators(df)


def test_empty_dataframe_raises() -> None:
    df = pd.DataFrame(columns=CANDLE_COLUMNS)
    with pytest.raises(IndicatorDataError, match="empty"):
        add_indicators(df)


def test_high_below_low_raises() -> None:
    df = _ohlcv_from_closes([10.0, 11.0, 12.0])
    df.loc[1, "high"] = 1.0
    df.loc[1, "low"] = 5.0
    with pytest.raises(IndicatorDataError, match="high < low"):
        add_indicators(df)
