"""Deterministic strategy variants for Phase 8/9 research."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from app.research.params import ResearchParams, validate_research_params
from app.strategy.baseline import (
    BaselineStrategy,
    Signal,
    StrategySignal,
    _as_finite_float,
    _has_key,
)


def _get(row: Mapping[str, Any] | pd.Series, *keys: str) -> float | None:
    for key in keys:
        if _has_key(row, key):
            return _as_finite_float(row[key])
    return None


def _extract(row: Mapping[str, Any] | pd.Series, keys: tuple[str, ...]) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for key in keys:
        parsed = _get(row, key)
        if parsed is None:
            return None
        values[key] = parsed
    return values


def _assign_signals(
    df: pd.DataFrame, long_mask: pd.Series, short_mask: pd.Series, reason: str
) -> pd.DataFrame:
    out = df.copy()
    signals = np.full(len(out), Signal.HOLD.value, dtype=object)
    reasons = np.full(len(out), "no confirmed setup", dtype=object)
    long_arr = long_mask.fillna(False).to_numpy()
    short_arr = short_mask.fillna(False).to_numpy()
    both = long_arr & short_arr
    long_arr = long_arr & ~both
    short_arr = short_arr & ~both
    signals[long_arr] = Signal.LONG.value
    signals[short_arr] = Signal.SHORT.value
    reasons[long_arr] = reason
    reasons[short_arr] = reason
    out["signal"] = signals
    out["signal_reason"] = reasons
    return out


class _ParamStrategyBase:
    """Shared helpers for parametric research strategies."""

    name: str = "base"

    def __init__(self, params: ResearchParams) -> None:
        validate_research_params(params)
        self.params = params

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        signals: list[str] = []
        reasons: list[str] = []
        for _, row in out.iterrows():
            result = self.generate_signal(row)
            signals.append(result.signal.value)
            reasons.append(result.reason)
        out["signal"] = signals
        out["signal_reason"] = reasons
        return out


class BaselineResearchStrategy(_ParamStrategyBase):
    """Exact Phase 3 baseline (ignores param EMA/RSI overrides for signal rules)."""

    name = "baseline"

    def __init__(self, params: ResearchParams | None = None) -> None:
        super().__init__(params or ResearchParams(variant="baseline"))
        self._inner = BaselineStrategy()

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        return self._inner.generate_signal(row)


class TrendMomentumStrategy(_ParamStrategyBase):
    """EMA stack + looser RSI + MACD histogram sign."""

    name = "trend_momentum"

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        p = self.params
        ind = _extract(row, ("ema_fast", "ema_med", "ema_slow", "rsi", "macd_hist"))
        if ind is None:
            return StrategySignal(Signal.HOLD, "missing or invalid indicators")
        bullish = ind["ema_fast"] > ind["ema_med"] > ind["ema_slow"]
        bearish = ind["ema_fast"] < ind["ema_med"] < ind["ema_slow"]
        if bullish and p.rsi_long_low <= ind["rsi"] <= p.rsi_long_high and ind["macd_hist"] > 0:
            return StrategySignal(Signal.LONG, "trend + loose momentum")
        if bearish and p.rsi_short_low <= ind["rsi"] <= p.rsi_short_high and ind["macd_hist"] < 0:
            return StrategySignal(Signal.SHORT, "trend + loose momentum")
        return StrategySignal(Signal.HOLD, "no confirmed setup")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        bullish = (df["ema_fast"] > df["ema_med"]) & (df["ema_med"] > df["ema_slow"])
        bearish = (df["ema_fast"] < df["ema_med"]) & (df["ema_med"] < df["ema_slow"])
        long_m = bullish & df["rsi"].between(p.rsi_long_low, p.rsi_long_high) & (df["macd_hist"] > 0)
        short_m = (
            bearish & df["rsi"].between(p.rsi_short_low, p.rsi_short_high) & (df["macd_hist"] < 0)
        )
        return _assign_signals(df, long_m, short_m, "trend + loose momentum")


class TrendFollowingStrategy(TrendMomentumStrategy):
    """Phase 9 family: EMA trend confirmation with momentum confirmation."""

    name = "trend_following"


class EmaCrossRsiStrategy(_ParamStrategyBase):
    """EMA fast/med relationship + slow-trend filter + RSI confirmation."""

    name = "ema_cross_rsi"

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        p = self.params
        ind = _extract(row, ("ema_fast", "ema_med", "ema_slow", "rsi"))
        if ind is None:
            return StrategySignal(Signal.HOLD, "missing or invalid indicators")
        up_trend = ind["ema_med"] > ind["ema_slow"] and ind["ema_fast"] > ind["ema_med"]
        down_trend = ind["ema_med"] < ind["ema_slow"] and ind["ema_fast"] < ind["ema_med"]
        if up_trend and p.rsi_long_low <= ind["rsi"] <= p.rsi_long_high:
            return StrategySignal(Signal.LONG, "ema cross + rsi")
        if down_trend and p.rsi_short_low <= ind["rsi"] <= p.rsi_short_high:
            return StrategySignal(Signal.SHORT, "ema cross + rsi")
        return StrategySignal(Signal.HOLD, "no confirmed setup")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        up = (df["ema_med"] > df["ema_slow"]) & (df["ema_fast"] > df["ema_med"])
        down = (df["ema_med"] < df["ema_slow"]) & (df["ema_fast"] < df["ema_med"])
        long_m = up & df["rsi"].between(p.rsi_long_low, p.rsi_long_high)
        short_m = down & df["rsi"].between(p.rsi_short_low, p.rsi_short_high)
        return _assign_signals(df, long_m, short_m, "ema cross + rsi")


class EmaMacdStrategy(_ParamStrategyBase):
    """EMA stack + MACD line vs signal (no RSI filter)."""

    name = "ema_macd"

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        ind = _extract(row, ("ema_fast", "ema_med", "ema_slow", "macd", "macd_signal", "macd_hist"))
        if ind is None:
            return StrategySignal(Signal.HOLD, "missing or invalid indicators")
        bullish = ind["ema_fast"] > ind["ema_med"] > ind["ema_slow"]
        bearish = ind["ema_fast"] < ind["ema_med"] < ind["ema_slow"]
        if bullish and ind["macd"] > ind["macd_signal"] and ind["macd_hist"] > 0:
            return StrategySignal(Signal.LONG, "ema + macd")
        if bearish and ind["macd"] < ind["macd_signal"] and ind["macd_hist"] < 0:
            return StrategySignal(Signal.SHORT, "ema + macd")
        return StrategySignal(Signal.HOLD, "no confirmed setup")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        bullish = (df["ema_fast"] > df["ema_med"]) & (df["ema_med"] > df["ema_slow"])
        bearish = (df["ema_fast"] < df["ema_med"]) & (df["ema_med"] < df["ema_slow"])
        long_m = bullish & (df["macd"] > df["macd_signal"]) & (df["macd_hist"] > 0)
        short_m = bearish & (df["macd"] < df["macd_signal"]) & (df["macd_hist"] < 0)
        return _assign_signals(df, long_m, short_m, "ema + macd")


class BreakoutMomentumStrategy(_ParamStrategyBase):
    """Volatility-aware breakout with ATR buffer and EMA slow filter."""

    name = "breakout_momentum"

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        close = _get(row, "close")
        atr_v = _get(row, "atr_14")
        prior_high = _get(row, "prior_high")
        prior_low = _get(row, "prior_low")
        ema_slow = _get(row, "ema_slow")
        if None in (close, atr_v, prior_high, prior_low, ema_slow):
            return StrategySignal(Signal.HOLD, "missing or invalid indicators")
        assert close is not None and atr_v is not None
        assert prior_high is not None and prior_low is not None and ema_slow is not None
        buffer = atr_v * 0.25
        if close > prior_high + buffer and close > ema_slow:
            return StrategySignal(Signal.LONG, "breakout momentum")
        if close < prior_low - buffer and close < ema_slow:
            return StrategySignal(Signal.SHORT, "breakout momentum")
        return StrategySignal(Signal.HOLD, "no confirmed setup")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        buffer = df["atr_14"] * 0.25
        long_m = (df["close"] > df["prior_high"] + buffer) & (df["close"] > df["ema_slow"])
        short_m = (df["close"] < df["prior_low"] - buffer) & (df["close"] < df["ema_slow"])
        return _assign_signals(df, long_m, short_m, "breakout momentum")


class BreakoutStrategy(BreakoutMomentumStrategy):
    """Phase 9 family alias for breakout."""

    name = "breakout"


class MomentumStrategy(_ParamStrategyBase):
    """Momentum without requiring the full baseline AND stack."""

    name = "momentum"

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        p = self.params
        ind = _extract(row, ("ema_med", "rsi", "macd_hist", "close"))
        if ind is None:
            return StrategySignal(Signal.HOLD, "missing or invalid indicators")
        if (
            ind["close"] > ind["ema_med"]
            and p.rsi_long_low <= ind["rsi"] <= p.rsi_long_high
            and ind["macd_hist"] > 0
        ):
            return StrategySignal(Signal.LONG, "momentum")
        if (
            ind["close"] < ind["ema_med"]
            and p.rsi_short_low <= ind["rsi"] <= p.rsi_short_high
            and ind["macd_hist"] < 0
        ):
            return StrategySignal(Signal.SHORT, "momentum")
        return StrategySignal(Signal.HOLD, "no confirmed setup")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        long_m = (
            (df["close"] > df["ema_med"])
            & df["rsi"].between(p.rsi_long_low, p.rsi_long_high)
            & (df["macd_hist"] > 0)
        )
        short_m = (
            (df["close"] < df["ema_med"])
            & df["rsi"].between(p.rsi_short_low, p.rsi_short_high)
            & (df["macd_hist"] < 0)
        )
        return _assign_signals(df, long_m, short_m, "momentum")


class MeanReversionStrategy(_ParamStrategyBase):
    """Extreme RSI + EMA deviation + ATR volatility filter."""

    name = "mean_reversion"

    def generate_signal(self, row: Mapping[str, Any] | pd.Series) -> StrategySignal:
        p = self.params
        ind = _extract(row, ("ema_med", "rsi", "atr_14", "close"))
        if ind is None:
            return StrategySignal(Signal.HOLD, "missing or invalid indicators")
        atr_pct = ind["atr_14"] / ind["close"] if ind["close"] else 0.0
        if atr_pct < p.min_atr_pct:
            return StrategySignal(Signal.HOLD, "volatility too low")
        lower = ind["ema_med"] * (1.0 - p.mean_dev_pct)
        upper = ind["ema_med"] * (1.0 + p.mean_dev_pct)
        if ind["rsi"] <= p.rsi_oversold and ind["close"] <= lower:
            return StrategySignal(Signal.LONG, "mean reversion long")
        if ind["rsi"] >= p.rsi_overbought and ind["close"] >= upper:
            return StrategySignal(Signal.SHORT, "mean reversion short")
        return StrategySignal(Signal.HOLD, "no confirmed setup")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        atr_pct = df["atr_14"] / df["close"]
        vol_ok = atr_pct >= p.min_atr_pct
        lower = df["ema_med"] * (1.0 - p.mean_dev_pct)
        upper = df["ema_med"] * (1.0 + p.mean_dev_pct)
        long_m = vol_ok & (df["rsi"] <= p.rsi_oversold) & (df["close"] <= lower)
        short_m = vol_ok & (df["rsi"] >= p.rsi_overbought) & (df["close"] >= upper)
        return _assign_signals(df, long_m, short_m, "mean reversion")


STRATEGY_FACTORIES: dict[str, Callable[[ResearchParams], Any]] = {
    "baseline": BaselineResearchStrategy,
    "trend_momentum": TrendMomentumStrategy,
    "ema_cross_rsi": EmaCrossRsiStrategy,
    "ema_macd": EmaMacdStrategy,
    "breakout_momentum": BreakoutMomentumStrategy,
    "trend_following": TrendFollowingStrategy,
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def build_strategy(params: ResearchParams) -> Any:
    validate_research_params(params)
    factory = STRATEGY_FACTORIES.get(params.variant)
    if factory is None:
        raise ValueError(
            f"Unknown strategy variant '{params.variant}'. "
            f"Known: {sorted(STRATEGY_FACTORIES)}"
        )
    return factory(params)
