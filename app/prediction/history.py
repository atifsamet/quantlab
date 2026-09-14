"""
Snapshot creation, refresh, and automatic evaluation (Phase 13).

No orders. No look-ahead when generating predictions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.prediction.config import ENGINE_VERSION, MARKET_DATA_DISCLAIMER, RESEARCH_CONCLUSION
from app.prediction.engine import (
    PredictionError,
    build_prediction_from_ohlcv,
    fetch_candles,
    predict,
    validate_symbol,
    validate_timeframe,
)
from app.prediction.evaluate import evaluate_directional, evaluation_horizon, future_slice_after_candle
from app.prediction.models import PredictionResult
from app.prediction import storage

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_DIR = ROOT / "data" / "historical"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def result_to_snapshot(result: PredictionResult) -> dict[str, Any]:
    """Convert a live PredictionResult into a persistable snapshot (unevaluated)."""
    levels = result.levels
    entry_price = result.current_price
    return {
        "id": "",  # assigned on append
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "prediction_timestamp": _utc_now_iso(),
        "candle_timestamp": result.timestamp,
        "current_price": result.current_price,
        "prediction": result.prediction,
        "signal_strength": result.signal_strength,
        "entry_price": entry_price,
        "entry_low": levels.entry_low,
        "entry_high": levels.entry_high,
        "stop_loss": levels.stop_loss,
        "take_profit": levels.take_profit,
        "risk_reward": levels.risk_reward,
        "trend": result.trend,
        "strategy_signal": result.strategy_signal,
        "indicators": result.indicators.model_dump(),
        "reasons": result.reasons.model_dump(),
        "engine_version": ENGINE_VERSION,
        "evaluated": False,
        "outcome": None,
        "evaluation_timestamp": None,
        "evaluation_horizon": evaluation_horizon(result.timeframe),
        "bars_to_outcome": None,
        "exit_price": None,
        "return_pct": None,
        "mfe_pct": None,
        "mae_pct": None,
        "directional_correct": None,
        "evaluation_note": None,
    }


def save_prediction_snapshot(result: PredictionResult) -> dict[str, Any]:
    """Persist snapshot; dedupe on symbol+timeframe+candle_timestamp."""
    snap = result_to_snapshot(result)
    return storage.append_record(snap)


def load_historical_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    symbol = validate_symbol(symbol)
    timeframe = validate_timeframe(timeframe)
    path = HISTORICAL_DIR / f"{symbol}_{timeframe}.csv"
    if not path.is_file():
        raise PredictionError(f"Missing historical CSV: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def evaluate_record_with_ohlcv(
    record: dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    flush: bool = True,
) -> dict[str, Any] | None:
    if record.get("evaluated"):
        return record
    future = future_slice_after_candle(ohlcv, record["candle_timestamp"])
    horizon = int(record.get("evaluation_horizon") or evaluation_horizon(record["timeframe"]))
    fields = evaluate_directional(
        side=str(record["prediction"]),
        entry_price=float(record["entry_price"]),
        stop_loss=record.get("stop_loss"),
        take_profit=record.get("take_profit"),
        future_ohlcv=future,
        horizon=horizon,
    )
    if not fields.get("evaluated"):
        return record  # still pending
    updated = storage.update_evaluation(record["id"], fields, flush=flush)
    return updated


def evaluate_pending(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    prefer_disk: bool = True,
    fetch_live: bool = False,
    client=None,
) -> dict[str, Any]:
    """
    Evaluate pending snapshots that now have enough future candles.
    Safe to run repeatedly; already-evaluated records are skipped.
    """
    pending = storage.list_records(
        symbol=symbol,
        timeframe=timeframe,
        evaluated=False,
        limit=10_000,
    )
    evaluated_n = 0
    still_pending = 0
    skipped = 0
    cache: dict[tuple[str, str], pd.DataFrame] = {}

    for rec in pending:
        key = (rec["symbol"], rec["timeframe"])
        if key not in cache:
            path = HISTORICAL_DIR / f"{key[0]}_{key[1]}.csv"
            frame: pd.DataFrame | None = None
            if prefer_disk and path.is_file():
                frame = load_historical_csv(key[0], key[1])
            if fetch_live:
                try:
                    live = fetch_candles(key[0], key[1], client=client)
                    if frame is None:
                        frame = live
                    else:
                        frame = (
                            pd.concat([frame, live], ignore_index=True)
                            .drop_duplicates(subset=["timestamp"])
                            .sort_values("timestamp")
                            .reset_index(drop=True)
                        )
                except Exception:
                    pass
            if frame is None or frame.empty:
                still_pending += 1
                continue
            cache[key] = frame

        before = rec.get("evaluated")
        updated = evaluate_record_with_ohlcv(rec, cache[key], flush=False)
        if updated and updated.get("evaluated") and not before:
            if updated.get("outcome") == "SKIPPED":
                skipped += 1
            evaluated_n += 1
        elif not updated or not updated.get("evaluated"):
            still_pending += 1

    if evaluated_n:
        storage.flush_store()

    return {
        "checked": len(pending),
        "newly_evaluated": evaluated_n,
        "still_pending": still_pending,
        "skipped_neutral": skipped,
    }


def refresh_and_store(symbol: str, timeframe: str, *, client=None) -> dict[str, Any]:
    """
    Fetch latest public data, generate closed-candle prediction, store snapshot,
    run pending eval. Does NOT place trades.
    """
    from app.prediction import stats as pred_stats

    result = predict(symbol, timeframe, use_cache=False, client=client)
    before = storage.find_duplicate(result.symbol, result.timeframe, result.timestamp)
    snapshot = save_prediction_snapshot(result)
    created_new = before is None

    eval_summary = evaluate_pending(
        symbol=symbol,
        timeframe=timeframe,
        prefer_disk=True,
        fetch_live=True,
        client=client,
    )
    hist = pred_stats.compute_stats(symbol=symbol, timeframe=timeframe)
    overall = hist.get("overall", {})
    pending_rows = storage.list_records(
        symbol=symbol,
        timeframe=timeframe,
        evaluated=False,
        limit=20,
    )
    return {
        "prediction": result.model_dump(),
        "snapshot": snapshot,
        "snapshot_created": created_new,
        "evaluation": eval_summary,
        "historical_context": {
            "label": "HISTORICAL PERFORMANCE",
            "note": (
                "Past snapshot outcomes for this asset/timeframe. "
                "Not a probability of future return."
            ),
            "sample": overall.get("scored_n"),
            "total_predictions": overall.get("total_predictions"),
            "win_rate": overall.get("win_rate"),
            "loss_rate": overall.get("loss_rate"),
            "timeout_rate": overall.get("timeout_rate"),
            "insufficient_sample": overall.get("insufficient_sample"),
        },
        "pending_predictions": pending_rows,
        "research_conclusion": RESEARCH_CONCLUSION,
        "disclaimers": {
            "prediction": result.disclaimer,
            "market_data": MARKET_DATA_DISCLAIMER,
            "research": (
                f"{RESEARCH_CONCLUSION} — historical testing has not demonstrated "
                "a durable predictive edge."
            ),
        },
    }


def prediction_detail(prediction_id: str) -> dict[str, Any] | None:
    rec = storage.get_by_id(prediction_id)
    if rec is None:
        return None
    # Attach future path for chart (evaluation only)
    chart_future: list[dict[str, Any]] = []
    chart_history: list[dict[str, Any]] = []
    try:
        ohlcv = load_historical_csv(rec["symbol"], rec["timeframe"])
    except PredictionError:
        try:
            ohlcv = fetch_candles(rec["symbol"], rec["timeframe"])
        except Exception:
            ohlcv = None
    if ohlcv is not None and not ohlcv.empty:
        ts = pd.Timestamp(rec["candle_timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        hist = ohlcv.loc[ohlcv["timestamp"] <= ts].tail(40)
        fut = future_slice_after_candle(ohlcv, ts).head(int(rec.get("evaluation_horizon") or 12) + 2)
        for _, row in hist.iterrows():
            chart_history.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "close": float(row["close"]),
                    "phase": "available_at_prediction",
                }
            )
        for _, row in fut.iterrows():
            chart_future.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "close": float(row["close"]),
                    "phase": "evaluation_only",
                }
            )
    return {
        "record": rec,
        "chart_history": chart_history,
        "chart_future": chart_future,
        "note": (
            "chart_future is used ONLY for post-hoc evaluation visualization. "
            "It was not available when the prediction was generated."
        ),
    }
