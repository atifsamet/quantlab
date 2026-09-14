"""Load Phase 8–11 research artifacts from disk (never fabricate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results"


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    # Replace NaN with None so JSON responses stay valid.
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return records


def load_phase9_summary() -> dict[str, Any] | None:
    return _read_json(RESULTS / "phase9" / "phase9_summary.json")  # type: ignore[return-value]


def load_phase10_summary() -> dict[str, Any] | None:
    return _read_json(RESULTS / "phase10" / "phase10_summary.json")  # type: ignore[return-value]


def load_phase11_report() -> dict[str, Any] | None:
    return _read_json(RESULTS / "phase11" / "phase11_report.json")  # type: ignore[return-value]


def load_strategies_table() -> list[dict[str, Any]]:
    path = RESULTS / "phase9" / "cells_summary.csv"
    return _read_csv(path)


def load_monte_carlo() -> dict[str, Any]:
    report = load_phase11_report() or {}
    summary_rows = _read_csv(RESULTS / "phase11" / "monte_carlo_summary.csv")
    dd_rows = _read_csv(RESULTS / "phase11" / "drawdown_distribution.csv")
    return {
        "available": bool(summary_rows or report.get("monte_carlo_pooled")),
        "summary": summary_rows,
        "drawdown_distribution": dd_rows,
        "pooled": report.get("monte_carlo_pooled"),
        "btc": report.get("monte_carlo_btc"),
        "random_null": report.get("random_null"),
        "verdict": report.get("verdict"),
        "notes": report.get("notes", []),
    }


def load_analytics() -> dict[str, Any]:
    p11 = load_phase11_report() or {}
    return {
        "available": bool(p11),
        "verdict": p11.get("verdict"),
        "answers": p11.get("answers", {}),
        "per_asset": p11.get("per_asset", []),
        "benchmarks": _read_csv(RESULTS / "phase11" / "benchmark_summary.csv"),
        "bootstrap": _read_csv(RESULTS / "phase11" / "bootstrap_summary.csv"),
        "bootstrap_pooled": p11.get("bootstrap_pooled", []),
        "cost_sensitivity": _read_csv(RESULTS / "phase11" / "cost_sensitivity.csv"),
        "statistical": _read_csv(RESULTS / "phase11" / "statistical_summary.csv"),
        "eval_window": p11.get("eval_window"),
        "pooled_mean_return_pct": p11.get("pooled_mean_return_pct"),
        "buy_and_hold_mean_return": p11.get("buy_and_hold_mean_return"),
        "notes": p11.get("notes", []),
    }


def load_equity(symbol: str, timeframe: str) -> dict[str, Any]:
    """Prefer existing exported equity CSVs; empty if missing."""
    sym = symbol.upper().replace("/", "-")
    tf = timeframe.lower()
    candidates = [
        RESULTS / f"{sym}_{tf}_test_phase5_equity.csv",
        RESULTS / f"{sym}_{tf}_test_equity.csv",
        RESULTS / f"{sym}_{tf}_equity.csv",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            rows = _read_csv(path)
            return {"available": True, "path": str(path), "points": rows}
    return {"available": False, "path": None, "points": []}


def load_trades(symbol: str, timeframe: str) -> dict[str, Any]:
    sym = symbol.upper().replace("/", "-")
    tf = timeframe.lower()
    candidates = [
        RESULTS / f"{sym}_{tf}_test_phase5_trades.csv",
        RESULTS / f"{sym}_{tf}_test_trades.csv",
        RESULTS / f"{sym}_{tf}_trades.csv",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return {"available": True, "path": str(path), "trades": _read_csv(path)}
    return {"available": False, "path": None, "trades": []}


def research_timeline() -> list[dict[str, Any]]:
    p9 = load_phase9_summary() or {}
    p10 = load_phase10_summary() or {}
    p11 = load_phase11_report() or {}
    best = p9.get("best_family") or {}
    return [
        {"phase": 1, "title": "Market Data", "result": "Public OKX candles", "conclusion": "Foundation ready"},
        {"phase": 2, "title": "Technical Indicators", "result": "EMA/RSI/MACD/ATR", "conclusion": "Warm-up aware"},
        {"phase": 3, "title": "Baseline Strategy", "result": "Trend+momentum AND rules", "conclusion": "Unvalidated educational baseline"},
        {"phase": 4, "title": "Backtesting", "result": "Look-ahead-safe simulator", "conclusion": "Fees/slippage modeled"},
        {"phase": 5, "title": "Risk Management", "result": "ATR stops, sizing, loss caps", "conclusion": "Drawdown reduced vs full capital"},
        {"phase": 6, "title": "Historical Data", "result": "Multi-year public OHLCV", "conclusion": "Real-data measurement only"},
        {"phase": 7, "title": "Robust Validation", "result": "Train/test + walk-forward", "conclusion": "Baseline not profitable"},
        {"phase": 8, "title": "Strategy Research", "result": "Small-grid OOS research", "conclusion": "No durable improvement"},
        {
            "phase": 9,
            "title": "Multi-Asset Research",
            "result": f"Best val family: {best.get('variant', 'n/a')}@{best.get('timeframe', 'n/a')}",
            "conclusion": "No robust strategy found",
        },
        {
            "phase": 10,
            "title": "ML Signal Filter",
            "result": p10.get("verdict", "Evidence insufficient"),
            "conclusion": "ML AUC < 0.5; no durable filter edge",
        },
        {
            "phase": 11,
            "title": "Statistical Validation",
            "result": p11.get("verdict", "No evidence of an edge"),
            "conclusion": "Not distinguishable from random; do not paper trade",
        },
        {
            "phase": 12,
            "title": "QuantLab Dashboard",
            "result": "Read-only prediction UI + FastAPI",
            "conclusion": "Analytical signals only; live trading disabled",
        },
        {
            "phase": 13,
            "title": "Prediction History & Evaluation",
            "result": "Stored snapshots + causal WIN/LOSS/TIMEOUT scoring",
            "conclusion": "Measure whether predictions were useful historically",
        },
        {
            "phase": 14,
            "title": "Live Market Intelligence",
            "result": "Closed-candle live refresh + freshness status",
            "conclusion": "Read-only live analysis; research conclusion unchanged",
        },
        {
            "phase": 15,
            "title": "Portfolio Release & Polish",
            "result": "UX, documentation, deployment readiness",
            "conclusion": "Product presentation without changing research findings",
        },
    ]
