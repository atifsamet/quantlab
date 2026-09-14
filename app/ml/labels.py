"""Outcome labels for candidate strategy signals (TP-before-SL)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml import LABEL_DOC
from app.risk.manager import compute_stop_loss, compute_take_profit


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """
    Risk-compatible labeling.

    See ``app.ml.LABEL_DOC`` for full documentation.
    """

    atr_multiplier: float = 1.5
    risk_reward_ratio: float = 2.0
    horizon: int = 24  # max bars after entry to resolve TP/SL
    atr_column: str = "atr_14"


def label_doc() -> str:
    return LABEL_DOC


def resolve_tp_sl_outcome(
    *,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    highs: np.ndarray,
    lows: np.ndarray,
) -> int:
    """
    Return 1 if TP before SL, else 0.

    Same-candle ambiguity -> SL first (matches backtester).
    """
    side_u = side.upper()
    for high, low in zip(highs, lows, strict=True):
        if side_u == "LONG":
            sl_hit = low <= stop_loss
            tp_hit = high >= take_profit
        else:
            sl_hit = high >= stop_loss
            tp_hit = low <= take_profit
        if sl_hit and tp_hit:
            return 0
        if sl_hit:
            return 0
        if tp_hit:
            return 1
    return 0  # timeout = unsuccessful


def label_signal_at_index(
    df: pd.DataFrame,
    signal_idx: int,
    side: str,
    cfg: LabelConfig,
) -> int | None:
    """
    Label one signal at ``signal_idx`` using future bars after entry.

    Returns None when entry/horizon cannot be formed (end of data / invalid ATR).
    """
    if signal_idx < 0 or signal_idx >= len(df) - 1:
        return None
    entry_idx = signal_idx + 1
    atr_raw = df.iloc[signal_idx][cfg.atr_column]
    if pd.isna(atr_raw) or float(atr_raw) <= 0:
        return None
    entry_price = float(df.iloc[entry_idx]["open"])
    if entry_price <= 0:
        return None

    stop = compute_stop_loss(side, entry_price, float(atr_raw), cfg.atr_multiplier)
    take = compute_take_profit(side, entry_price, stop, cfg.risk_reward_ratio)

    end_idx = min(len(df), entry_idx + cfg.horizon)
    if end_idx <= entry_idx:
        return None
    window = df.iloc[entry_idx:end_idx]
    return resolve_tp_sl_outcome(
        side=side,
        entry_price=entry_price,
        stop_loss=stop,
        take_profit=take,
        highs=window["high"].to_numpy(dtype=float),
        lows=window["low"].to_numpy(dtype=float),
    )


def label_signals(
    df: pd.DataFrame,
    *,
    signal_column: str = "signal",
    cfg: LabelConfig | None = None,
) -> pd.Series:
    """
    Return a Series of labels aligned to ``df`` index.

    Non-signal rows and unresolvable outcomes are NaN.
    """
    cfg = cfg or LabelConfig()
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    for i in range(len(df)):
        side = str(df.iloc[i][signal_column]).upper()
        if side not in ("LONG", "SHORT"):
            continue
        lab = label_signal_at_index(df, i, side, cfg)
        if lab is not None:
            labels.iloc[i] = float(lab)
    return labels
