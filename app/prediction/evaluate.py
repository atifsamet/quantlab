"""
Causal prediction evaluation (Phase 13).

For a prediction at candle N:
  - Prediction used only candles <= N (enforced by the generator).
  - Evaluation uses candles N+1 .. N+horizon only.

WIN / LOSS: take-profit vs stop-loss touch on future bars.
Same-candle conflict: STOP LOSS FIRST (project convention).
Neither reached within horizon: TIMEOUT.
NEUTRAL without levels: SKIPPED (not counted as win/loss).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.prediction.config import EVAL_HORIZONS


def evaluation_horizon(timeframe: str) -> int:
    return int(EVAL_HORIZONS.get(timeframe, 6))


def _pct(distance: float, entry: float) -> float | None:
    if entry == 0:
        return None
    return round(100.0 * distance / entry, 6)


def evaluate_directional(
    *,
    side: str,
    entry_price: float,
    stop_loss: float | None,
    take_profit: float | None,
    future_ohlcv: pd.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    """
    Evaluate LONG/SHORT against future candles (must start at N+1).

    ``future_ohlcv`` columns: high, low, close.
    """
    side_u = side.upper()
    now = datetime.now(timezone.utc).isoformat()

    if side_u == "NEUTRAL" or stop_loss is None or take_profit is None:
        return {
            "evaluated": True,
            "outcome": "SKIPPED",
            "evaluation_timestamp": now,
            "evaluation_horizon": horizon,
            "bars_to_outcome": None,
            "exit_price": None,
            "return_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "directional_correct": None,
            "evaluation_note": "NEUTRAL or missing SL/TP — not scored as WIN/LOSS",
        }

    if future_ohlcv is None or len(future_ohlcv) == 0:
        return {
            "evaluated": False,
            "outcome": "PENDING",
            "evaluation_timestamp": None,
            "evaluation_horizon": horizon,
            "bars_to_outcome": None,
            "exit_price": None,
            "return_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "directional_correct": None,
            "evaluation_note": "Insufficient future candles",
        }

    window = future_ohlcv.iloc[:horizon]
    if len(window) < horizon:
        return {
            "evaluated": False,
            "outcome": "PENDING",
            "evaluation_timestamp": None,
            "evaluation_horizon": horizon,
            "bars_to_outcome": None,
            "exit_price": None,
            "return_pct": None,
            "mfe_pct": None,
            "mae_pct": None,
            "directional_correct": None,
            "evaluation_note": f"Need {horizon} future bars, have {len(window)}",
        }

    outcome = "TIMEOUT"
    bars_to: int | None = None
    exit_price: float | None = None
    best_fav = 0.0
    worst_adv = 0.0

    for i, row in enumerate(window.itertuples(index=False), start=1):
        high = float(row.high)
        low = float(row.low)

        if side_u == "LONG":
            best_fav = max(best_fav, high - entry_price)
            worst_adv = min(worst_adv, low - entry_price)
            hit_sl = low <= float(stop_loss)
            hit_tp = high >= float(take_profit)
        else:
            best_fav = max(best_fav, entry_price - low)
            worst_adv = min(worst_adv, entry_price - high)
            hit_sl = high >= float(stop_loss)
            hit_tp = low <= float(take_profit)

        # Same-candle: STOP LOSS FIRST
        if hit_sl:
            outcome = "LOSS"
            bars_to = i
            exit_price = float(stop_loss)
            break
        if hit_tp:
            outcome = "WIN"
            bars_to = i
            exit_price = float(take_profit)
            break

    last_close = float(window.iloc[-1]["close"])
    if outcome == "TIMEOUT":
        exit_price = last_close
        bars_to = horizon
        if side_u == "LONG":
            ret = _pct(last_close - entry_price, entry_price)
            directional = last_close > entry_price
        else:
            ret = _pct(entry_price - last_close, entry_price)
            directional = last_close < entry_price
    else:
        if side_u == "LONG":
            ret = _pct(float(exit_price) - entry_price, entry_price)
        else:
            ret = _pct(entry_price - float(exit_price), entry_price)
        directional = outcome == "WIN"

    return {
        "evaluated": True,
        "outcome": outcome,
        "evaluation_timestamp": now,
        "evaluation_horizon": horizon,
        "bars_to_outcome": bars_to,
        "exit_price": exit_price,
        "return_pct": ret,
        "mfe_pct": _pct(best_fav, entry_price),
        "mae_pct": _pct(worst_adv, entry_price),
        "directional_correct": bool(directional),
        "evaluation_note": (
            "TP before SL"
            if outcome == "WIN"
            else "SL before TP (SL-first on same candle)"
            if outcome == "LOSS"
            else "Neither SL nor TP within horizon"
        ),
    }


def future_slice_after_candle(
    ohlcv: pd.DataFrame,
    candle_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return candles strictly after the prediction candle timestamp."""
    ts = pd.Timestamp(candle_timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    frame = ohlcv.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[frame["timestamp"] > ts].reset_index(drop=True)
