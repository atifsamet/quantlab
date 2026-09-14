"""Unit tests for the Phase 3 baseline strategy (no network)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.strategy.baseline import (
    REASON_HOLD,
    REASON_LONG,
    REASON_MISSING,
    REASON_SHORT,
    BaselineStrategy,
    Signal,
    generate_signal,
    generate_signals,
)


def _row(**overrides: float | str | None) -> dict[str, float | str | None]:
    base: dict[str, float | str | None] = {
        "ema_20": 110.0,
        "ema_50": 105.0,
        "ema_200": 100.0,
        "rsi_14": 60.0,
        "macd": 1.5,
        "macd_signal": 1.0,
        "macd_hist": 0.5,
    }
    base.update(overrides)
    return base


def _short_row(**overrides: float | str | None) -> dict[str, float | str | None]:
    base: dict[str, float | str | None] = {
        "ema_20": 90.0,
        "ema_50": 95.0,
        "ema_200": 100.0,
        "rsi_14": 40.0,
        "macd": -1.5,
        "macd_signal": -1.0,
        "macd_hist": -0.5,
    }
    base.update(overrides)
    return base


@pytest.fixture
def strategy() -> BaselineStrategy:
    return BaselineStrategy()


def test_valid_long_setup(strategy: BaselineStrategy) -> None:
    result = strategy.generate_signal(_row())
    assert result.signal == Signal.LONG
    assert result.reason == REASON_LONG


def test_valid_short_setup(strategy: BaselineStrategy) -> None:
    result = strategy.generate_signal(_short_row())
    assert result.signal == Signal.SHORT
    assert result.reason == REASON_SHORT


def test_bullish_trend_rsi_outside_range_holds(strategy: BaselineStrategy) -> None:
    high = strategy.generate_signal(_row(rsi_14=75.0))
    assert high.signal == Signal.HOLD
    assert high.reason == REASON_HOLD

    low = strategy.generate_signal(_row(rsi_14=45.0))
    assert low.signal == Signal.HOLD
    assert low.reason == REASON_HOLD


def test_bullish_trend_macd_missing_confirmation_holds(
    strategy: BaselineStrategy,
) -> None:
    no_cross = strategy.generate_signal(_row(macd=0.5, macd_signal=1.0, macd_hist=-0.5))
    assert no_cross.signal == Signal.HOLD

    hist_zero = strategy.generate_signal(_row(macd_hist=0.0))
    assert hist_zero.signal == Signal.HOLD


def test_bearish_trend_rsi_outside_range_holds(strategy: BaselineStrategy) -> None:
    low = strategy.generate_signal(_short_row(rsi_14=25.0))
    assert low.signal == Signal.HOLD

    high = strategy.generate_signal(_short_row(rsi_14=55.0))
    assert high.signal == Signal.HOLD


def test_bearish_trend_macd_missing_confirmation_holds(
    strategy: BaselineStrategy,
) -> None:
    no_cross = strategy.generate_signal(
        _short_row(macd=-0.5, macd_signal=-1.0, macd_hist=0.5)
    )
    assert no_cross.signal == Signal.HOLD

    hist_zero = strategy.generate_signal(_short_row(macd_hist=0.0))
    assert hist_zero.signal == Signal.HOLD


def test_missing_indicator_holds(strategy: BaselineStrategy) -> None:
    incomplete = {
        "ema_20": 110.0,
        "ema_50": 105.0,
        "ema_200": 100.0,
        "rsi_14": 60.0,
    }
    result = strategy.generate_signal(incomplete)
    assert result.signal == Signal.HOLD
    assert result.reason == REASON_MISSING


def test_nan_indicator_holds(strategy: BaselineStrategy) -> None:
    result = strategy.generate_signal(_row(rsi_14=float("nan")))
    assert result.signal == Signal.HOLD
    assert result.reason == REASON_MISSING

    series = pd.Series(_row(macd=np.nan))
    result_series = strategy.generate_signal(series)
    assert result_series.signal == Signal.HOLD


def test_invalid_numeric_value_holds(strategy: BaselineStrategy) -> None:
    inf_result = strategy.generate_signal(_row(ema_20=math.inf))
    assert inf_result.signal == Signal.HOLD
    assert inf_result.reason == REASON_MISSING

    bad_result = strategy.generate_signal(_row(macd_signal="not-a-number"))
    assert bad_result.signal == Signal.HOLD
    assert bad_result.reason == REASON_MISSING


def test_module_level_generate_signal() -> None:
    result = generate_signal(_row())
    assert result.signal == Signal.LONG


def test_generate_signals_dataframe(strategy: BaselineStrategy) -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _short_row(),
            _row(rsi_14=80.0),
            {**_row(), "rsi_14": np.nan},
        ]
    )
    out = strategy.generate_signals(frame)

    assert list(out["signal"]) == [
        Signal.LONG.value,
        Signal.SHORT.value,
        Signal.HOLD.value,
        Signal.HOLD.value,
    ]
    assert out.loc[0, "signal_reason"] == REASON_LONG
    assert out.loc[1, "signal_reason"] == REASON_SHORT
    assert out.loc[2, "signal_reason"] == REASON_HOLD
    assert out.loc[3, "signal_reason"] == REASON_MISSING
    assert "ema_20" in out.columns
    assert "signal" in out.columns
    assert "signal_reason" in out.columns


def test_module_level_generate_signals() -> None:
    frame = pd.DataFrame([_row(), _short_row()])
    out = generate_signals(frame)
    assert list(out["signal"]) == [Signal.LONG.value, Signal.SHORT.value]


def test_rsi_boundaries_inclusive(strategy: BaselineStrategy) -> None:
    assert strategy.generate_signal(_row(rsi_14=50.0)).signal == Signal.LONG
    assert strategy.generate_signal(_row(rsi_14=70.0)).signal == Signal.LONG
    assert strategy.generate_signal(_short_row(rsi_14=30.0)).signal == Signal.SHORT
    assert strategy.generate_signal(_short_row(rsi_14=50.0)).signal == Signal.SHORT
