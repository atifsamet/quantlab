"""
Historical prediction backfill (causal, no look-ahead).

Usage:
  python -m app.prediction.backfill

For each sample index i, the prediction uses only candles <= i.
Future candles i+1.. are used later for evaluation only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.indicators import WARMUP_BARS, add_indicators
from app.prediction.config import (
    BACKFILL_STRIDE,
    BACKFILL_WINDOWS_DAYS,
    ENGINE_VERSION,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
)
from app.prediction.engine import (
    PredictionError,
    _levels,
    _score_conditions,
    _trend_label,
    build_prediction_from_ohlcv,
    validate_symbol,
    validate_timeframe,
)
from app.prediction.evaluate import evaluation_horizon
from app.prediction.explanation import build_explanation
from app.prediction.history import evaluate_record_with_ohlcv, load_historical_csv
from app.prediction.models import IndicatorSnapshot, PredictionResult
from app.prediction import storage
from app.research.indicators_flex import add_research_indicators
from app.research.params import ResearchParams
from app.research.variants import build_strategy


def _row_prediction(
    with_ind: pd.DataFrame,
    signaled: pd.DataFrame,
    index: int,
    *,
    symbol: str,
    timeframe: str,
) -> PredictionResult:
    """Build prediction from prepared frames at index (causal if frames were built causally)."""
    last = with_ind.iloc[index]
    last_sig = signaled.iloc[index]
    close = float(last["close"])
    ts = pd.Timestamp(last["timestamp"])
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    def f(col: str) -> float | None:
        if col not in last.index or pd.isna(last[col]):
            return None
        return float(last[col])

    indicators = IndicatorSnapshot(
        ema_20=f("ema_20"),
        ema_50=f("ema_50"),
        ema_200=f("ema_200"),
        rsi_14=f("rsi_14"),
        macd=f("macd"),
        macd_signal=f("macd_signal"),
        macd_hist=f("macd_hist"),
        atr_14=f("atr_14"),
        atr_pct=(float(last["atr_14"]) / close) if f("atr_14") else None,
        volume=f("volume"),
        volume_sma_20=f("volume_sma_20"),
        volume_ratio=(
            float(last["volume"]) / float(last["volume_sma_20"])
            if f("volume") and f("volume_sma_20") and float(last["volume_sma_20"]) > 0
            else None
        ),
    )
    strategy_signal = str(last_sig["signal"]).upper()
    trend = _trend_label(close, indicators.ema_50, indicators.ema_200)
    prediction, strength = _score_conditions(
        trend=trend,
        strategy_signal=strategy_signal,
        rsi=indicators.rsi_14,
        macd_hist=indicators.macd_hist,
        volume_ratio=indicators.volume_ratio,
        ema20=indicators.ema_20,
        ema50=indicators.ema_50,
        ema200=indicators.ema_200,
        close=close,
    )
    levels = _levels(side=prediction, price=close, atr=indicators.atr_14)
    reasons = build_explanation(
        prediction=prediction,
        strategy_signal=strategy_signal,
        indicators=indicators,
        trend=trend,
    )
    return PredictionResult(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts.isoformat(),
        current_price=close,
        prediction=prediction,
        signal_strength=strength,
        ml_probability=None,
        trend=trend,
        strategy_signal=strategy_signal,
        levels=levels,
        indicators=indicators,
        reasons=reasons,
        warnings=[],
        research_conclusion="NO ROBUST EDGE DETECTED",
        disclaimer="Historical backfill snapshot — analytical only.",
    )


def build_prediction_at_index(
    ohlcv: pd.DataFrame,
    index: int,
    *,
    symbol: str,
    timeframe: str,
) -> PredictionResult:
    """
    Strictly causal prediction at candle index.

    Uses ONLY ohlcv.iloc[: index + 1]. Mutating future rows must not change the result.
    """
    if index < 0 or index >= len(ohlcv):
        raise PredictionError(f"Index {index} out of range for OHLCV length {len(ohlcv)}")
    sliced = ohlcv.iloc[: index + 1].copy().reset_index(drop=True)
    return build_prediction_from_ohlcv(sliced, symbol=symbol, timeframe=timeframe)


def snapshot_from_result(result: PredictionResult, *, prediction_timestamp: str | None = None) -> dict[str, Any]:
    levels = result.levels
    return {
        "id": "",
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "prediction_timestamp": prediction_timestamp or result.timestamp,
        "candle_timestamp": result.timestamp,
        "current_price": result.current_price,
        "prediction": result.prediction,
        "signal_strength": result.signal_strength,
        "entry_price": result.current_price,
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


def backfill_symbol_timeframe(
    symbol: str,
    timeframe: str,
    *,
    evaluate: bool = True,
) -> dict[str, Any]:
    symbol = validate_symbol(symbol)
    timeframe = validate_timeframe(timeframe)
    print(f"  loading {symbol} {timeframe}...", flush=True)
    ohlcv = load_historical_csv(symbol, timeframe)

    days = BACKFILL_WINDOWS_DAYS[timeframe]
    stride = BACKFILL_STRIDE[timeframe]
    end_ts = pd.Timestamp(ohlcv["timestamp"].iloc[-1])
    start_ts = end_ts - timedelta(days=days)

    print(f"  computing indicators ({len(ohlcv)} bars)...", flush=True)
    with_ind = add_indicators(ohlcv)
    params = ResearchParams(variant="breakout", breakout_lookback=20)
    research = add_research_indicators(ohlcv, params)
    signaled = build_strategy(params).generate_signals(research)

    horizon = evaluation_horizon(timeframe)
    max_i = len(ohlcv) - 1 - horizon
    min_i = WARMUP_BARS

    indices = []
    for i in range(min_i, max_i + 1):
        ts = pd.Timestamp(ohlcv.iloc[i]["timestamp"])
        if ts < start_ts:
            continue
        if (i - min_i) % stride != 0:
            continue
        indices.append(i)

    print(f"  generating {len(indices)} snapshots (stride={stride})...", flush=True)
    batch: list[dict[str, Any]] = []
    for i in indices:
        result = _row_prediction(with_ind, signaled, i, symbol=symbol, timeframe=timeframe)
        batch.append(snapshot_from_result(result, prediction_timestamp=result.timestamp))

    counts = storage.bulk_append(batch)
    created = counts["created"]
    skipped_dup = counts["duplicates"]

    evaluated = 0
    if evaluate:
        print(f"  evaluating...", flush=True)
        # Re-load records for this symbol/tf that are pending
        pending = storage.list_records(symbol=symbol, timeframe=timeframe, evaluated=False, limit=50_000)
        for rec in pending:
            updated = evaluate_record_with_ohlcv(rec, ohlcv, flush=False)
            if updated and updated.get("evaluated"):
                evaluated += 1
        storage.flush_store()

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candidates": len(indices),
        "created": created,
        "duplicates_skipped": skipped_dup,
        "evaluated": evaluated,
        "window_days": days,
        "stride": stride,
    }


def run_backfill(
    *,
    symbols: tuple[str, ...] | None = None,
    timeframes: tuple[str, ...] | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    symbols = symbols or SUPPORTED_SYMBOLS
    timeframes = timeframes or SUPPORTED_TIMEFRAMES
    storage.ensure_store()
    results = []
    for symbol in symbols:
        for tf in timeframes:
            print(f"Backfilling {symbol} {tf}...")
            results.append(backfill_symbol_timeframe(symbol, tf, evaluate=evaluate))
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "total_created": sum(r["created"] for r in results),
        "total_evaluated": sum(r["evaluated"] for r in results),
        "store_count": storage.count_records(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantLab prediction history backfill (no trading)")
    parser.add_argument("--symbol", action="append", help="Limit to symbol (repeatable)")
    parser.add_argument("--timeframe", action="append", help="Limit to timeframe (repeatable)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation pass")
    args = parser.parse_args()
    summary = run_backfill(
        symbols=tuple(args.symbol) if args.symbol else None,
        timeframes=tuple(args.timeframe) if args.timeframe else None,
        evaluate=not args.no_eval,
    )
    print(summary)


if __name__ == "__main__":
    main()
