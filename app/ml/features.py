"""Feature extraction for Phase 10 ML signal filter (no future leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.technical import volume_sma
from app.research.indicators_flex import add_research_indicators
from app.research.params import ResearchParams

# Columns used as model inputs (excluding metadata / labels).
FEATURE_COLUMNS: tuple[str, ...] = (
    "ema_fast",
    "ema_med",
    "ema_slow",
    "ema_fast_med_ratio",
    "ema_med_slow_ratio",
    "ema_fast_slow_ratio",
    "close_ema_fast_dist",
    "close_ema_med_dist",
    "close_ema_slow_dist",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_norm",
    "atr_14",
    "atr_pct",
    "volume_sma_20",
    "volume_ratio",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "vol_12",
    "dist_prior_high",
    "dist_prior_low",
    "breakout_strength",
    "trend_strength",
    "body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "side_sign",
)


def enrich_feature_frame(df: pd.DataFrame, params: ResearchParams | None = None) -> pd.DataFrame:
    """
    Add research indicators plus ML features derived only from past/present bars.
    """
    params = params or ResearchParams(variant="breakout")
    out = add_research_indicators(df, params)
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    volume = out["volume"]

    eps = 1e-12
    out["ema_fast_med_ratio"] = out["ema_fast"] / (out["ema_med"] + eps)
    out["ema_med_slow_ratio"] = out["ema_med"] / (out["ema_slow"] + eps)
    out["ema_fast_slow_ratio"] = out["ema_fast"] / (out["ema_slow"] + eps)
    out["close_ema_fast_dist"] = (close - out["ema_fast"]) / (close + eps)
    out["close_ema_med_dist"] = (close - out["ema_med"]) / (close + eps)
    out["close_ema_slow_dist"] = (close - out["ema_slow"]) / (close + eps)

    out["atr_pct"] = out["atr_14"] / (close + eps)
    out["macd_hist_norm"] = out["macd_hist"] / (close + eps)

    if "volume_sma_20" not in out.columns:
        out["volume_sma_20"] = volume_sma(volume, 20)
    out["volume_ratio"] = volume / (out["volume_sma_20"] + eps)

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)
    out["vol_12"] = close.pct_change(1).rolling(12).std()

    out["dist_prior_high"] = (close - out["prior_high"]) / (close + eps)
    out["dist_prior_low"] = (close - out["prior_low"]) / (close + eps)
    # Breakout strength vs ATR (positive when beyond prior extreme).
    out["breakout_strength"] = np.where(
        close > out["prior_high"],
        (close - out["prior_high"]) / (out["atr_14"] + eps),
        np.where(
            close < out["prior_low"],
            (out["prior_low"] - close) / (out["atr_14"] + eps),
            0.0,
        ),
    )
    out["trend_strength"] = (out["ema_fast"] - out["ema_slow"]) / (out["atr_14"] + eps)

    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    out["body_pct"] = body / (close + eps)
    out["range_pct"] = (high - low) / (close + eps)
    out["upper_wick_pct"] = (high - np.maximum(open_, close)) / (close + eps)
    out["lower_wick_pct"] = (np.minimum(open_, close) - low) / (close + eps)

    return out


def extract_feature_row(row: pd.Series, *, side: str) -> dict[str, float]:
    """Build a feature dict for one signal row (side encoded as +/-1)."""
    feats: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col == "side_sign":
            feats[col] = 1.0 if str(side).upper() == "LONG" else -1.0
            continue
        val = row[col]
        feats[col] = float(val) if pd.notna(val) else float("nan")
    return feats


def feature_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature matrix; drops rows with any NaN feature."""
    missing = [c for c in FEATURE_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return rows.loc[:, list(FEATURE_COLUMNS)].astype(float)
