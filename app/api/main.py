"""
QuantLab FastAPI application — read-only research & prediction API.

No order endpoints. LIVE_TRADING_ENABLED must remain false.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.artifacts import (
    load_analytics,
    load_equity,
    load_monte_carlo,
    load_phase9_summary,
    load_strategies_table,
    load_trades,
    research_timeline,
)
from app.api.schemas import HealthResponse, OverviewCard, OverviewResponse
from app.config import get_settings
from app.prediction import (
    DISCLAIMER,
    RESEARCH_CONCLUSION,
    PredictionError,
    PredictionResult,
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    market_snapshot,
    predict,
)
from app.prediction.config import MARKET_DATA_DISCLAIMER
from app.prediction.history import (
    evaluate_pending,
    prediction_detail,
    refresh_and_store,
    save_prediction_snapshot,
)
from app.prediction import stats as pred_stats
from app.prediction import storage as pred_storage
from app.utils.logger import setup_logging

setup_logging()
settings = get_settings()
settings.assert_trading_disabled()

app = FastAPI(
    title="QuantLab API",
    description="Quantitative crypto market analysis and prediction (research only).",
    version="0.15.0",
)

app.add_middleware(
    CORSMiddleware,
    # Next.js may bind 3000, or 3001+ if 3000 is already taken.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings.assert_trading_disabled()
    return HealthResponse(live_trading_enabled=settings.live_trading_enabled)


@app.get("/api/market/{symbol}/{timeframe}")
def get_market(symbol: str, timeframe: str) -> dict[str, Any]:
    try:
        return market_snapshot(symbol, timeframe)
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # network / OKX
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}") from exc


@app.get("/api/prediction/{symbol}/{timeframe}", response_model=PredictionResult)
def get_prediction(
    symbol: str,
    timeframe: str,
    refresh: bool = Query(False, description="Bypass short cache"),
    persist: bool = Query(False, description="Also store a history snapshot"),
) -> PredictionResult:
    try:
        result = predict(symbol, timeframe, use_cache=not refresh)
        if persist or refresh:
            save_prediction_snapshot(result)
            evaluate_pending(symbol=symbol, timeframe=timeframe, prefer_disk=True)
        return result
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Prediction unavailable: {exc}") from exc


@app.get("/api/predictions/history")
def predictions_history(
    symbol: str | None = None,
    timeframe: str | None = None,
    prediction: str | None = None,
    outcome: str | None = None,
    evaluated: bool | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    rows = pred_storage.list_records(
        symbol=symbol,
        timeframe=timeframe,
        prediction=prediction,
        outcome=outcome,
        evaluated=evaluated,
        limit=limit,
    )
    return {"count": len(rows), "records": rows}


@app.get("/api/predictions/stats")
def predictions_stats(
    symbol: str | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    return pred_stats.compute_stats(symbol=symbol, timeframe=timeframe)


@app.get("/api/predictions/performance/{symbol}/{timeframe}")
def predictions_performance(symbol: str, timeframe: str) -> dict[str, Any]:
    try:
        return pred_stats.compute_stats(symbol=symbol, timeframe=timeframe)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predictions/latest")
def predictions_latest(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    rows = pred_storage.list_records(limit=limit)
    return {"count": len(rows), "records": rows}


@app.get("/api/predictions/{prediction_id}")
def predictions_by_id(prediction_id: str) -> dict[str, Any]:
    detail = prediction_detail(prediction_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return detail


@app.post("/api/predictions/refresh")
def predictions_refresh(
    symbol: str = Query(...),
    timeframe: str = Query(...),
) -> dict[str, Any]:
    """Generate and store a NEW prediction snapshot. Does not execute trades."""
    try:
        return refresh_and_store(symbol, timeframe)
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Refresh unavailable: {exc}") from exc


@app.post("/api/predictions/evaluate")
def predictions_evaluate() -> dict[str, Any]:
    """Re-check pending predictions against available future candles."""
    return evaluate_pending(prefer_disk=True)


@app.get("/api/overview", response_model=OverviewResponse)
def overview(timeframe: str = Query("4h")) -> OverviewResponse:
    cards: list[OverviewCard] = []
    for symbol in SUPPORTED_SYMBOLS:
        try:
            pred = predict(symbol, timeframe, use_cache=True)
            live = pred.live
            cards.append(
                OverviewCard(
                    symbol=pred.symbol,
                    timeframe=pred.timeframe,
                    current_price=pred.current_price,
                    prediction=pred.prediction,
                    signal_strength=pred.signal_strength,
                    trend=pred.trend,
                    timestamp=pred.timestamp,
                    market_status=None if live is None else live.market_status,
                    prediction_status=None if live is None else live.prediction_status,
                    last_market_update=None if live is None else live.last_market_update,
                )
            )
        except Exception as exc:
            cards.append(
                OverviewCard(
                    symbol=symbol,
                    timeframe=timeframe,
                    market_status="ERROR",
                    error=str(exc),
                )
            )
    analytics = load_analytics()
    mc = load_monte_carlo()
    hist = pred_stats.compute_stats()
    overall = hist.get("overall", {})
    return OverviewResponse(
        research_conclusion=RESEARCH_CONCLUSION,
        disclaimer=DISCLAIMER,
        market_data_disclaimer=MARKET_DATA_DISCLAIMER,
        cards=cards,
        analytics_verdict=analytics.get("verdict"),
        monte_carlo_available=bool(mc.get("available")),
        prediction_performance={
            "label": "Historical prediction performance",
            "win_rate": overall.get("win_rate"),
            "evaluated": overall.get("evaluated_predictions"),
            "pending": overall.get("pending_predictions"),
            "long_win_rate": (overall.get("long") or {}).get("win_rate"),
            "short_win_rate": (overall.get("short") or {}).get("win_rate"),
            "insufficient_sample": overall.get("insufficient_sample"),
            "total": overall.get("total_predictions"),
        },
    )


@app.get("/api/strategies")
def strategies() -> dict[str, Any]:
    rows = load_strategies_table()
    p9 = load_phase9_summary()
    return {
        "available": bool(rows),
        "rows": rows,
        "phase9_best": None if not p9 else p9.get("best_family"),
        "phase9_note": None if not p9 else p9.get("note"),
        "supported_symbols": list(SUPPORTED_SYMBOLS),
        "supported_timeframes": list(SUPPORTED_TIMEFRAMES),
    }


@app.get("/api/analytics")
def analytics() -> dict[str, Any]:
    return load_analytics()


@app.get("/api/monte-carlo")
def monte_carlo() -> dict[str, Any]:
    return load_monte_carlo()


@app.get("/api/research")
def research() -> dict[str, Any]:
    return {
        "conclusion": RESEARCH_CONCLUSION,
        "disclaimer": DISCLAIMER,
        "timeline": research_timeline(),
        "live_trading_enabled": False,
        "mode": "RESEARCH MODE",
    }


@app.get("/api/equity/{symbol}/{timeframe}")
def equity(symbol: str, timeframe: str) -> dict[str, Any]:
    return load_equity(symbol, timeframe)


@app.get("/api/trades/{symbol}/{timeframe}")
def trades(symbol: str, timeframe: str) -> dict[str, Any]:
    return load_trades(symbol, timeframe)


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "symbols": list(SUPPORTED_SYMBOLS),
        "timeframes": list(SUPPORTED_TIMEFRAMES),
        "live_trading_enabled": False,
        "mode": "RESEARCH MODE",
    }
