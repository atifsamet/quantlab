"""Phase 8 research framework tests (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.indicators import WARMUP_BARS, add_indicators
from app.research.diagnostics import diagnose_baseline_filters
from app.research.indicators_flex import add_research_indicators
from app.research.params import ResearchParams, validate_research_params
from app.research.search import (
    RobustnessConfig,
    build_small_search_grid,
    evaluate_candidate,
    passes_robustness,
    prepare_candidate_history,
    rank_candidates,
    score_candidate,
)
from app.research.variants import (
    BreakoutMomentumStrategy,
    EmaCrossRsiStrategy,
    EmaMacdStrategy,
    TrendMomentumStrategy,
    build_strategy,
)
from app.strategy.baseline import BaselineStrategy, Signal


def _ohlcv(n: int = 350) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for i in range(n):
        ts = start + pd.Timedelta(hours=i)
        price = 100.0 + 0.08 * i + 3.0 * np.sin(i / 11.0)
        rows.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price + 1.5,
                "low": price - 1.5,
                "close": price + 0.3 * np.cos(i / 9.0),
                "volume": 5.0 + i % 7,
            }
        )
    return pd.DataFrame(rows)


def test_param_validation() -> None:
    validate_research_params(ResearchParams())
    with pytest.raises(ValueError):
        validate_research_params(ResearchParams(ema_fast=50, ema_med=40, ema_slow=200))
    with pytest.raises(ValueError):
        validate_research_params(ResearchParams(rsi_long_low=80, rsi_long_high=70))


def test_build_strategy_variants() -> None:
    for name in (
        "baseline",
        "trend_momentum",
        "ema_cross_rsi",
        "ema_macd",
        "breakout_momentum",
    ):
        strat = build_strategy(ResearchParams(variant=name))
        assert strat is not None
    with pytest.raises(ValueError):
        build_strategy(ResearchParams(variant="nope"))


def test_variant_signals_deterministic() -> None:
    df = add_research_indicators(_ohlcv(), ResearchParams(variant="trend_momentum"))
    strat = TrendMomentumStrategy(ResearchParams(variant="trend_momentum"))
    a = strat.generate_signals(df)
    b = strat.generate_signals(df)
    assert list(a["signal"]) == list(b["signal"])
    assert set(a["signal"]).issubset({"LONG", "SHORT", "HOLD"})


def test_baseline_research_matches_phase3() -> None:
    raw = _ohlcv()
    framed = add_indicators(raw)
    phase3 = BaselineStrategy().generate_signals(framed)
    research = prepare_candidate_history(raw, ResearchParams(variant="baseline"))
    assert list(phase3["signal"]) == list(research["signal"])


def test_no_lookahead_breakout_uses_shifted_highs() -> None:
    df = add_research_indicators(_ohlcv(80), ResearchParams(breakout_lookback=5))
    # prior_high at i must equal max(high[i-5:i]) not including i
    for i in range(10, 30):
        expected = float(df["high"].iloc[i - 5 : i].max())
        assert df["prior_high"].iloc[i] == pytest.approx(expected)


def test_diagnostics_runs() -> None:
    framed = add_indicators(_ohlcv())
    report = diagnose_baseline_filters(framed)
    assert report.rows == len(framed)
    assert report.hold_pct + report.long_pct + report.short_pct == pytest.approx(100.0)
    assert report.ready_rows <= report.rows


def test_small_grid_is_bounded() -> None:
    grid = build_small_search_grid()
    assert 5 <= len(grid) <= 40
    assert any(p.variant == "baseline" for p in grid)


def test_rank_and_score_ordering() -> None:
    raw = _ohlcv(400)
    # Use short windows with enough warm-up in synthetic data
    start = raw["timestamp"].iloc[WARMUP_BARS]
    mid = raw["timestamp"].iloc[WARMUP_BARS + 40]
    end = raw["timestamp"].iloc[WARMUP_BARS + 80]
    c1 = evaluate_candidate(
        raw,
        ResearchParams(variant="ema_macd"),
        train_start=start,
        train_end=mid,
        val_start=mid,
        val_end=end,
        robustness=RobustnessConfig(min_trades=1, max_drawdown_pct=100),
    )
    c2 = evaluate_candidate(
        raw,
        ResearchParams(variant="baseline"),
        train_start=start,
        train_end=mid,
        val_start=mid,
        val_end=end,
        robustness=RobustnessConfig(min_trades=1, max_drawdown_pct=100),
    )
    ranked = rank_candidates([c1, c2])
    assert len(ranked) == 2
    # Reproducibility
    c1b = evaluate_candidate(
        raw,
        ResearchParams(variant="ema_macd"),
        train_start=start,
        train_end=mid,
        val_start=mid,
        val_end=end,
        robustness=RobustnessConfig(min_trades=1, max_drawdown_pct=100),
    )
    assert c1.train.metrics.total_return_pct == pytest.approx(
        c1b.train.metrics.total_return_pct
    )


def test_train_val_separation(tmp_path: Path) -> None:
    raw = _ohlcv(450)
    train_start = raw["timestamp"].iloc[WARMUP_BARS]
    train_end = raw["timestamp"].iloc[WARMUP_BARS + 50]
    val_start = raw["timestamp"].iloc[WARMUP_BARS + 51]
    val_end = raw["timestamp"].iloc[WARMUP_BARS + 100]
    cand = evaluate_candidate(
        raw,
        ResearchParams(variant="ema_cross_rsi"),
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        robustness=RobustnessConfig(min_trades=0, max_drawdown_pct=100),
    )
    assert cand.train.frame["timestamp"].max() <= train_end
    assert cand.validation.frame["timestamp"].min() >= val_start
    assert cand.final_test is None  # not run during evaluate_candidate


def test_passes_robustness_min_trades() -> None:
    raw = _ohlcv(400)
    start = raw["timestamp"].iloc[WARMUP_BARS]
    mid = raw["timestamp"].iloc[WARMUP_BARS + 30]
    end = raw["timestamp"].iloc[WARMUP_BARS + 60]
    cand = evaluate_candidate(
        raw,
        ResearchParams(variant="baseline"),
        train_start=start,
        train_end=mid,
        val_start=mid,
        val_end=end,
        robustness=RobustnessConfig(min_trades=1000),
    )
    assert cand.rejected is True
    assert "too few trades" in cand.reject_reason


def test_long_short_metrics_present() -> None:
    raw = _ohlcv(400)
    hist = prepare_candidate_history(raw, ResearchParams(variant="breakout_momentum"))
    assert "signal" in hist.columns
    # Strategy produces finite signals only from current/prior columns
    strat = BreakoutMomentumStrategy(ResearchParams())
    row = hist.iloc[-1]
    sig = strat.generate_signal(row)
    assert sig.signal in {Signal.LONG, Signal.SHORT, Signal.HOLD}
