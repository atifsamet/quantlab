"""
Phase 9 multi-asset / multi-timeframe strategy research.

Selection uses TRAIN + VALIDATION only. FINAL TEST is evaluated once after
selection. No live trading, no private APIs, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.engine import BacktestConfig, load_ohlcv_csv
from app.backtest.robust import PeriodRun, generate_walk_forward_windows, run_period
from app.indicators import WARMUP_BARS, add_indicators
from app.research.costs import DEFAULT_COST_SCENARIOS, CostScenario, has_funding_column
from app.research.indicators_flex import RESEARCH_REQUIRED_COLUMNS, add_research_indicators
from app.research.params import ResearchParams, validate_research_params
from app.research.portfolio import PortfolioConfig, PortfolioResult, run_portfolio_backtest
from app.research.search import RobustnessConfig, passes_robustness, score_candidate
from app.research.universe import (
    DEFAULT_SYMBOLS,
    DEFAULT_TIMEFRAMES,
    PHASE9_FAMILIES,
    csv_path_for,
    discover_available,
)
from app.research.variants import build_strategy
from app.risk.config import RiskConfig
from app.strategy.baseline import REQUIRED_INDICATORS


@dataclass(slots=True)
class SignalFrequency:
    rows: int
    long_count: int
    short_count: int
    hold_count: int
    long_pct: float
    short_pct: float
    hold_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "long_count": self.long_count,
            "short_count": self.short_count,
            "hold_count": self.hold_count,
            "long_pct": self.long_pct,
            "short_pct": self.short_pct,
            "hold_pct": self.hold_pct,
        }


@dataclass(slots=True)
class CellResult:
    """One strategy x symbol x timeframe evaluation on train/val/(optional test)."""

    params: ResearchParams
    symbol: str
    timeframe: str
    train: PeriodRun
    validation: PeriodRun
    signal_freq: SignalFrequency
    rejected: bool
    reject_reason: str
    score: float
    final_test: PeriodRun | None = None
    walk_forward_tests: list[PeriodRun] = field(default_factory=list)
    cost_sensitivity: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "label": self.params.label,
            "variant": self.params.variant,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "score": self.score,
            "signal_long_pct": self.signal_freq.long_pct,
            "signal_short_pct": self.signal_freq.short_pct,
            "signal_hold_pct": self.signal_freq.hold_pct,
            "train_return": self.train.metrics.total_return_pct,
            "train_trades": self.train.metrics.total_trades,
            "train_pf": self.train.metrics.profit_factor,
            "train_dd": self.train.metrics.max_drawdown_pct,
            "val_return": self.validation.metrics.total_return_pct,
            "val_trades": self.validation.metrics.total_trades,
            "val_pf": self.validation.metrics.profit_factor,
            "val_dd": self.validation.metrics.max_drawdown_pct,
            "val_win_rate": self.validation.metrics.win_rate,
            "val_expectancy": self.validation.metrics.expectancy_per_trade,
            "val_time_in_market": self.validation.metrics.time_in_market_pct,
            "val_long_trades": self.validation.metrics.long.trade_count,
            "val_short_trades": self.validation.metrics.short.trade_count,
            "final_return": (
                None if self.final_test is None else self.final_test.metrics.total_return_pct
            ),
            "final_trades": (
                None if self.final_test is None else self.final_test.metrics.total_trades
            ),
            "final_dd": (
                None if self.final_test is None else self.final_test.metrics.max_drawdown_pct
            ),
        }


@dataclass(slots=True)
class FamilyAggregate:
    variant: str
    timeframe: str
    cells: list[CellResult]
    mean_val_return: float
    median_val_return: float
    assets_positive_val: int
    assets_total: int
    mean_val_trades: float
    cross_asset_score: float
    rejected: bool
    reject_reason: str
    final_test_mean_return: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "timeframe": self.timeframe,
            "mean_val_return": self.mean_val_return,
            "median_val_return": self.median_val_return,
            "assets_positive_val": self.assets_positive_val,
            "assets_total": self.assets_total,
            "mean_val_trades": self.mean_val_trades,
            "cross_asset_score": self.cross_asset_score,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "final_test_mean_return": self.final_test_mean_return,
        }


def build_phase9_candidates() -> list[ResearchParams]:
    """Small fixed family set — not a re-optimization of the Phase 8 winner."""
    return [
        ResearchParams(variant="baseline"),
        ResearchParams(
            variant="trend_following",
            # Intentionally wider than Phase-3 RSI bands so this family differs
            # from baseline (MACD hist sign alone is equivalent to line>signal).
            rsi_long_low=45.0,
            rsi_long_high=75.0,
            rsi_short_low=25.0,
            rsi_short_high=55.0,
        ),
        ResearchParams(variant="momentum"),
        ResearchParams(variant="breakout", breakout_lookback=20),
        ResearchParams(
            variant="mean_reversion",
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            mean_dev_pct=0.015,
            min_atr_pct=0.002,
        ),
    ]


def measure_signal_frequency(history: pd.DataFrame) -> SignalFrequency:
    if "signal" not in history.columns or history.empty:
        return SignalFrequency(0, 0, 0, 0, 0.0, 0.0, 100.0)
    sig = history["signal"].astype(str).str.upper()
    n = len(sig)
    long_c = int((sig == "LONG").sum())
    short_c = int((sig == "SHORT").sum())
    hold_c = n - long_c - short_c
    return SignalFrequency(
        rows=n,
        long_count=long_c,
        short_count=short_c,
        hold_count=hold_c,
        long_pct=100.0 * long_c / n,
        short_pct=100.0 * short_c / n,
        hold_pct=100.0 * hold_c / n,
    )


def prepare_history(ohlcv: pd.DataFrame, params: ResearchParams) -> pd.DataFrame:
    validate_research_params(params)
    if params.variant == "baseline":
        framed = add_indicators(ohlcv)
        # Vectorized Phase-3 rules (identical logic, much faster on 15m).
        long_m = (
            (framed["ema_20"] > framed["ema_50"])
            & (framed["ema_50"] > framed["ema_200"])
            & framed["rsi_14"].between(50.0, 70.0)
            & (framed["macd"] > framed["macd_signal"])
            & (framed["macd_hist"] > 0)
        )
        short_m = (
            (framed["ema_20"] < framed["ema_50"])
            & (framed["ema_50"] < framed["ema_200"])
            & framed["rsi_14"].between(30.0, 50.0)
            & (framed["macd"] < framed["macd_signal"])
            & (framed["macd_hist"] < 0)
        )
        from app.research.variants import _assign_signals

        return _assign_signals(framed, long_m, short_m, "baseline vectorized")
    framed = add_research_indicators(ohlcv, params)
    return build_strategy(params).generate_signals(framed)


def _required_columns(params: ResearchParams) -> tuple[str, ...]:
    return REQUIRED_INDICATORS if params.variant == "baseline" else RESEARCH_REQUIRED_COLUMNS


def evaluate_cell(
    ohlcv: pd.DataFrame,
    params: ResearchParams,
    *,
    symbol: str,
    timeframe: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    use_risk: bool = True,
    robustness: RobustnessConfig | None = None,
    backtest_config: BacktestConfig | None = None,
) -> CellResult:
    robustness = robustness or RobustnessConfig(min_trades=4)
    history = prepare_history(ohlcv, params)
    freq = measure_signal_frequency(history)
    required = _required_columns(params)
    risk_cfg = RiskConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
    )
    cfg = backtest_config or BacktestConfig()
    train = run_period(
        history,
        label=f"train:{params.variant}:{symbol}:{timeframe}",
        start=train_start,
        end=train_end,
        instrument=symbol,
        timeframe=timeframe,
        use_risk=use_risk,
        risk_config=risk_cfg,
        required_columns=required,
        config=cfg,
    )
    validation = run_period(
        history,
        label=f"val:{params.variant}:{symbol}:{timeframe}",
        start=val_start,
        end=val_end,
        instrument=symbol,
        timeframe=timeframe,
        use_risk=use_risk,
        risk_config=risk_cfg,
        required_columns=required,
        config=cfg,
    )

    rejected = False
    reasons: list[str] = []
    # Soft signal-frequency notes; hard reject only extreme cases
    trade_signal_pct = freq.long_pct + freq.short_pct
    if trade_signal_pct < 0.5:
        rejected = True
        reasons.append(f"near-zero signal frequency ({trade_signal_pct:.2f}%)")
    if freq.hold_pct < 5.0:
        rejected = True
        reasons.append(f"unrealistic exposure (hold {freq.hold_pct:.1f}%)")

    ok_t, reason_t = passes_robustness(train, robustness)
    ok_v, reason_v = passes_robustness(validation, robustness)
    if not ok_t:
        rejected = True
        reasons.append(f"train: {reason_t}")
    if not ok_v:
        rejected = True
        reasons.append(f"validation: {reason_v}")
    if validation.metrics.total_return_pct < -20:
        rejected = True
        reasons.append("validation collapsed (< -20%)")

    score = float("-inf") if rejected else score_candidate(train, validation)
    return CellResult(
        params=params,
        symbol=symbol,
        timeframe=timeframe,
        train=train,
        validation=validation,
        signal_freq=freq,
        rejected=rejected,
        reject_reason="; ".join(reasons),
        score=score,
    )


def aggregate_family(
    cells: list[CellResult],
    *,
    min_assets: int = 2,
) -> FamilyAggregate:
    if not cells:
        raise ValueError("no cells")
    variant = cells[0].params.variant
    timeframe = cells[0].timeframe
    accepted = [c for c in cells if not c.rejected]
    val_returns = [c.validation.metrics.total_return_pct for c in cells]
    mean_ret = float(pd.Series(val_returns).mean()) if val_returns else 0.0
    median_ret = float(pd.Series(val_returns).median()) if val_returns else 0.0
    pos = sum(1 for r in val_returns if r > 0)
    mean_trades = float(
        pd.Series([c.validation.metrics.total_trades for c in cells]).mean()
    )
    # Prefer strategies that work on multiple assets with decent trades
    cross = (
        mean_ret
        + 2.0 * median_ret
        + 5.0 * (pos / max(len(cells), 1))
        + min(mean_trades, 30) / 30.0 * 5.0
        - 0.5 * float(pd.Series([c.validation.metrics.max_drawdown_pct for c in cells]).mean())
    )
    rejected = False
    reason = ""
    if len(accepted) < min_assets and len(cells) >= min_assets:
        rejected = True
        reason = f"accepted on fewer than {min_assets} assets"
    if pos == 0 and mean_ret < 0:
        # Not automatically rejected — may still be best-of-bad; flag in reason
        reason = (reason + "; ").lstrip("; ") + "no positive validation asset"
    if rejected:
        cross = float("-inf")
    return FamilyAggregate(
        variant=variant,
        timeframe=timeframe,
        cells=cells,
        mean_val_return=mean_ret,
        median_val_return=median_ret,
        assets_positive_val=pos,
        assets_total=len(cells),
        mean_val_trades=mean_trades,
        cross_asset_score=cross,
        rejected=rejected,
        reject_reason=reason,
    )


def run_phase9_pipeline(
    *,
    data_dir: str | Path = "data/historical",
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
    train_start: str = "2024-09-01",
    train_end: str = "2025-03-01",
    val_start: str = "2025-03-01",
    val_end: str = "2025-09-01",
    test_start: str = "2025-09-01",
    test_end: str = "2026-09-01",
    use_risk: bool = True,
    robustness: RobustnessConfig | None = None,
    top_k_families: int = 3,
    portfolio_timeframe: str = "1h",
) -> dict[str, Any]:
    """
    Full Phase 9 research pass.

    Strategy family selection uses train+validation across assets.
    Final test runs once for selected family aggregates.
    """
    robustness = robustness or RobustnessConfig(min_trades=4)
    available = discover_available(data_dir, symbols, timeframes)
    if not available:
        raise FileNotFoundError(
            f"No historical CSVs found under {data_dir} for {symbols} x {timeframes}"
        )

    # Load OHLCV cache
    ohlcv_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol, tf, path in available:
        frame = load_ohlcv_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        ohlcv_cache[(symbol, tf)] = frame

    candidates = build_phase9_candidates()
    cells: list[CellResult] = []
    for params in candidates:
        for symbol, tf, _path in available:
            cells.append(
                evaluate_cell(
                    ohlcv_cache[(symbol, tf)],
                    params,
                    symbol=symbol,
                    timeframe=tf,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    use_risk=use_risk,
                    robustness=robustness,
                )
            )

    # Aggregate by (variant, timeframe)
    aggregates: list[FamilyAggregate] = []
    for variant in PHASE9_FAMILIES:
        for tf in timeframes:
            group = [c for c in cells if c.params.variant == variant and c.timeframe == tf]
            if not group:
                continue
            aggregates.append(aggregate_family(group))

    ranked_families = sorted(
        aggregates,
        key=lambda a: (not a.rejected, a.cross_asset_score),
        reverse=True,
    )
    selected = [a for a in ranked_families if not a.rejected][:top_k_families]
    if not selected and ranked_families:
        # Do not force a winner — keep empty selected and report best rejected
        selected = []

    # Final untouched test for ALL cells in selected families (reporting).
    # Family selection already used train+validation only.
    for fam in selected:
        finals: list[float] = []
        for cell in fam.cells:
            history = prepare_history(ohlcv_cache[(cell.symbol, cell.timeframe)], cell.params)
            risk_cfg = RiskConfig(
                atr_multiplier=cell.params.atr_stop_multiplier,
                risk_reward_ratio=cell.params.risk_reward_ratio,
            )
            cell.final_test = run_period(
                history,
                label=f"final:{cell.params.variant}:{cell.symbol}:{cell.timeframe}",
                start=test_start,
                end=test_end,
                instrument=cell.symbol,
                timeframe=cell.timeframe,
                use_risk=use_risk,
                risk_config=risk_cfg,
                required_columns=_required_columns(cell.params),
            )
            # Cost sensitivity on final window (fee/slip only; funding if column exists)
            for scenario in DEFAULT_COST_SCENARIOS:
                if scenario.apply_funding and not has_funding_column(history):
                    cell.cost_sensitivity.append(
                        {
                            "scenario": scenario.name,
                            "skipped": True,
                            "reason": "no funding_rate column (not invented)",
                        }
                    )
                    continue
                period = run_period(
                    history,
                    label=f"cost:{scenario.name}:{cell.symbol}",
                    start=test_start,
                    end=test_end,
                    instrument=cell.symbol,
                    timeframe=cell.timeframe,
                    use_risk=use_risk,
                    risk_config=risk_cfg,
                    required_columns=_required_columns(cell.params),
                    config=scenario.to_backtest_config(),
                )
                cell.cost_sensitivity.append(
                    {
                        "scenario": scenario.name,
                        "skipped": False,
                        "return_pct": period.metrics.total_return_pct,
                        "trades": period.metrics.total_trades,
                        "max_dd": period.metrics.max_drawdown_pct,
                    }
                )
            finals.append(cell.final_test.metrics.total_return_pct)

            # Walk-forward on selected cells (BTC primary + others lightly)
            if cell.symbol == "BTC-USDT":
                wf_windows = generate_walk_forward_windows(
                    overall_start=train_start,
                    overall_end=test_end,
                    train_days=180,
                    test_days=90,
                    step_days=90,
                )
                for i, (_a, _b, te_s, te_e) in enumerate(wf_windows, start=1):
                    cell.walk_forward_tests.append(
                        run_period(
                            history,
                            label=f"wf{i}:{cell.params.variant}:{cell.symbol}",
                            start=te_s,
                            end=te_e,
                            instrument=cell.symbol,
                            timeframe=cell.timeframe,
                            use_risk=use_risk,
                            risk_config=risk_cfg,
                            required_columns=_required_columns(cell.params),
                            warmup_bars=WARMUP_BARS,
                        )
                    )
        fam.final_test_mean_return = float(pd.Series(finals).mean()) if finals else None

    # Portfolio-test the validation winner on its own timeframe (1x, capped risk).
    portfolio: PortfolioResult | None = None
    portfolio_meta: dict[str, Any] = {}
    port_family = selected[0] if selected else None
    if port_family is not None:
        params = port_family.cells[0].params
        frames: dict[str, pd.DataFrame] = {}
        for cell in port_family.cells:
            if cell.timeframe != port_family.timeframe:
                continue
            hist = prepare_history(ohlcv_cache[(cell.symbol, cell.timeframe)], params)
            from app.backtest.robust import slice_evaluation

            eval_frame = slice_evaluation(
                hist,
                test_start,
                test_end,
                required_columns=_required_columns(params),
            )
            frames[cell.symbol] = eval_frame
        if frames:
            portfolio = run_portfolio_backtest(
                frames,
                config=PortfolioConfig(
                    atr_multiplier=params.atr_stop_multiplier,
                    risk_reward_ratio=params.risk_reward_ratio,
                ),
            )
            portfolio_meta = {
                "variant": params.variant,
                "timeframe": port_family.timeframe,
                "symbols": sorted(frames),
                "return_pct": portfolio.result.total_return_pct,
                "trades": portfolio.result.total_trades,
                "max_dd": portfolio.result.max_drawdown_pct,
                "peak_positions": portfolio.open_position_peak,
                "symbols_traded": portfolio.symbols_traded,
                "notes": portfolio.notes,
            }

    return {
        "available": [(s, t, str(p)) for s, t, p in available],
        "cells": cells,
        "aggregates": ranked_families,
        "selected": selected,
        "best_family": selected[0] if selected else None,
        "portfolio": portfolio_meta,
        "candidates": [c.label for c in candidates],
        "note": (
            "No family forced as winner. Empty selected means no robust cross-asset edge "
            "on train/validation."
        ),
    }
