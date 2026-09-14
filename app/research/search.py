"""Controlled candidate search, ranking, and robustness filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable

import pandas as pd

from app.backtest.analysis import RobustMetrics
from app.backtest.engine import BacktestConfig, load_ohlcv_csv
from app.backtest.robust import PeriodRun, run_period
from app.indicators import WARMUP_BARS, add_indicators
from app.research.indicators_flex import RESEARCH_REQUIRED_COLUMNS, add_research_indicators
from app.research.params import ResearchParams, validate_research_params
from app.research.variants import build_strategy
from app.risk.config import RiskConfig
from app.strategy.baseline import REQUIRED_INDICATORS, generate_signals


@dataclass(slots=True)
class RobustnessConfig:
    min_trades: int = 5
    max_drawdown_pct: float = 25.0
    max_top_trade_share: float = 0.60  # reject if best trade > 60% of gross profit
    require_both_sides: bool = False
    min_side_trades: int = 0


@dataclass(slots=True)
class CandidateResult:
    params: ResearchParams
    train: PeriodRun
    validation: PeriodRun
    rejected: bool
    reject_reason: str
    score: float
    final_test: PeriodRun | None = None
    walk_forward_tests: list[PeriodRun] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "label": self.params.label,
            "variant": self.params.variant,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "score": self.score,
            "train_return": self.train.metrics.total_return_pct,
            "train_trades": self.train.metrics.total_trades,
            "train_pf": self.train.metrics.profit_factor,
            "train_dd": self.train.metrics.max_drawdown_pct,
            "val_return": self.validation.metrics.total_return_pct,
            "val_trades": self.validation.metrics.total_trades,
            "val_pf": self.validation.metrics.profit_factor,
            "val_dd": self.validation.metrics.max_drawdown_pct,
            "val_expectancy": self.validation.metrics.expectancy_per_trade,
            "final_test_return": (
                None if self.final_test is None else self.final_test.metrics.total_return_pct
            ),
            "final_test_trades": (
                None if self.final_test is None else self.final_test.metrics.total_trades
            ),
            "params": self.params.to_dict(),
        }


def build_small_search_grid() -> list[ResearchParams]:
    """
    Intentionally small predefined grid (not unrestricted brute force).

    Includes the exact baseline plus a few explainable variants/params.
    """
    candidates: list[ResearchParams] = [
        ResearchParams(variant="baseline"),
        ResearchParams(variant="trend_momentum", ema_fast=20, ema_med=50, ema_slow=200),
        ResearchParams(
            variant="trend_momentum",
            ema_fast=10,
            ema_med=40,
            ema_slow=200,
            rsi_long_low=45.0,
            rsi_long_high=70.0,
            rsi_short_low=30.0,
            rsi_short_high=55.0,
        ),
        ResearchParams(
            variant="trend_momentum",
            ema_fast=30,
            ema_med=60,
            ema_slow=200,
            rsi_long_low=50.0,
            rsi_long_high=75.0,
        ),
        ResearchParams(variant="ema_cross_rsi", ema_fast=20, ema_med=50, ema_slow=200),
        ResearchParams(variant="ema_cross_rsi", ema_fast=10, ema_med=40, ema_slow=200),
        ResearchParams(variant="ema_macd", ema_fast=20, ema_med=50, ema_slow=200),
        ResearchParams(variant="ema_macd", ema_fast=10, ema_med=50, ema_slow=200),
        ResearchParams(variant="breakout_momentum", breakout_lookback=20),
        ResearchParams(variant="breakout_momentum", breakout_lookback=15),
    ]

    # Compact risk-geometry slice on one structural variant
    for atr_x, rr in product((1.0, 1.5, 2.0), (1.5, 2.0, 2.5)):
        candidates.append(
            ResearchParams(
                variant="trend_momentum",
                ema_fast=20,
                ema_med=50,
                ema_slow=200,
                atr_stop_multiplier=atr_x,
                risk_reward_ratio=rr,
            )
        )

    unique: dict[str, ResearchParams] = {}
    for params in candidates:
        validate_research_params(params)
        unique[params.label] = params
    return list(unique.values())


def _top_trade_share(_metrics: RobustMetrics, result_trades: list) -> float:
    wins = [t.net_pnl for t in result_trades if t.net_pnl > 0]
    if not wins:
        return 0.0
    gross = sum(wins)
    if gross <= 0:
        return 0.0
    return max(wins) / gross


def passes_robustness(
    period: PeriodRun,
    cfg: RobustnessConfig,
) -> tuple[bool, str]:
    m = period.metrics
    if m.total_trades < cfg.min_trades:
        return False, f"too few trades ({m.total_trades} < {cfg.min_trades})"
    if m.max_drawdown_pct > cfg.max_drawdown_pct:
        return False, f"drawdown {m.max_drawdown_pct:.2f}% > {cfg.max_drawdown_pct:.2f}%"
    share = _top_trade_share(m, period.result.trades)
    if share > cfg.max_top_trade_share and m.total_trades >= 3:
        return False, f"top trade share {share:.2%} suggests dependency on few trades"
    if cfg.require_both_sides:
        if m.long.trade_count < cfg.min_side_trades or m.short.trade_count < cfg.min_side_trades:
            return False, "insufficient LONG/SHORT balance"
    if m.max_consecutive_losses >= 10 and m.total_trades >= 10:
        return False, "excessive consecutive losses"
    return True, ""


def score_candidate(train: PeriodRun, validation: PeriodRun) -> float:
    """
    Rank using validation quality with a train sanity term.

    Higher is better. Penalize drawdown and sparse trading.
    """
    v = validation.metrics
    t = train.metrics
    pf_v = 3.0 if v.profit_factor == float("inf") else min(v.profit_factor, 5.0)
    pf_t = 3.0 if t.profit_factor == float("inf") else min(t.profit_factor, 5.0)
    trade_term = min(v.total_trades, 40) / 40.0
    return (
        v.total_return_pct
        + 8.0 * v.expectancy_per_trade
        + 4.0 * pf_v
        + 2.0 * pf_t
        + 10.0 * trade_term
        - 1.5 * v.max_drawdown_pct
        - 0.5 * t.max_drawdown_pct
    )


def prepare_candidate_history(ohlcv: pd.DataFrame, params: ResearchParams) -> pd.DataFrame:
    """Indicators + signals for one candidate (no future bars beyond ohlcv)."""
    if params.variant == "baseline":
        framed = add_indicators(ohlcv)
        return generate_signals(framed)

    framed = add_research_indicators(ohlcv, params)
    strategy = build_strategy(params)
    return strategy.generate_signals(framed)


def evaluate_candidate(
    ohlcv: pd.DataFrame,
    params: ResearchParams,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    instrument: str = "BTC-USDT",
    timeframe: str = "1h",
    use_risk: bool = True,
    robustness: RobustnessConfig | None = None,
) -> CandidateResult:
    validate_research_params(params)
    robustness = robustness or RobustnessConfig()
    history = prepare_candidate_history(ohlcv, params)
    required = (
        REQUIRED_INDICATORS if params.variant == "baseline" else RESEARCH_REQUIRED_COLUMNS
    )
    risk_cfg = RiskConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
    )
    train = run_period(
        history,
        label=f"train:{params.variant}",
        start=train_start,
        end=train_end,
        instrument=instrument,
        timeframe=timeframe,
        use_risk=use_risk,
        risk_config=risk_cfg,
        required_columns=required,
    )
    validation = run_period(
        history,
        label=f"val:{params.variant}",
        start=val_start,
        end=val_end,
        instrument=instrument,
        timeframe=timeframe,
        use_risk=use_risk,
        risk_config=risk_cfg,
        required_columns=required,
    )

    ok_t, reason_t = passes_robustness(train, robustness)
    ok_v, reason_v = passes_robustness(validation, robustness)
    rejected = not (ok_t and ok_v)
    reject_reason = ""
    if not ok_t:
        reject_reason = f"train: {reason_t}"
    elif not ok_v:
        reject_reason = f"validation: {reason_v}"

    # Also reject clear train/val collapse patterns
    if not rejected and validation.metrics.total_return_pct < -15:
        rejected = True
        reject_reason = "validation collapsed (< -15%)"
    if not rejected and train.metrics.total_return_pct > 20 and validation.metrics.total_return_pct < -5:
        rejected = True
        reject_reason = "train/validation divergence suggests overfitting"

    score = float("-inf") if rejected else score_candidate(train, validation)
    return CandidateResult(
        params=params,
        train=train,
        validation=validation,
        rejected=rejected,
        reject_reason=reject_reason,
        score=score,
    )


def rank_candidates(candidates: Iterable[CandidateResult]) -> list[CandidateResult]:
    return sorted(
        candidates,
        key=lambda c: (not c.rejected, c.score),
        reverse=True,
    )


def run_research_pipeline(
    csv_path: str,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    instrument: str = "BTC-USDT",
    timeframe: str = "1h",
    use_risk: bool = True,
    top_k_final: int = 3,
    top_k_walkforward: int = 2,
    robustness: RobustnessConfig | None = None,
    grid: list[ResearchParams] | None = None,
) -> dict[str, Any]:
    """
    Full Phase 8 pipeline.

    Selection uses TRAIN+VALIDATION only. FINAL TEST is evaluated once for
    top_k_final survivors. Walk-forward uses top_k_walkforward.
    """
    robustness = robustness or RobustnessConfig()
    ohlcv = load_ohlcv_csv(csv_path)
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)

    grid = grid or build_small_search_grid()
    evaluated = [
        evaluate_candidate(
            ohlcv,
            params,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            instrument=instrument,
            timeframe=timeframe,
            use_risk=use_risk,
            robustness=robustness,
        )
        for params in grid
    ]
    ranked = rank_candidates(evaluated)
    accepted = [c for c in ranked if not c.rejected]

    # Final untouched test — once per selected candidate
    for candidate in accepted[:top_k_final]:
        history = prepare_candidate_history(ohlcv, candidate.params)
        required = (
            REQUIRED_INDICATORS
            if candidate.params.variant == "baseline"
            else RESEARCH_REQUIRED_COLUMNS
        )
        risk_cfg = RiskConfig(
            atr_multiplier=candidate.params.atr_stop_multiplier,
            risk_reward_ratio=candidate.params.risk_reward_ratio,
        )
        candidate.final_test = run_period(
            history,
            label=f"final_test:{candidate.params.variant}",
            start=test_start,
            end=test_end,
            instrument=instrument,
            timeframe=timeframe,
            use_risk=use_risk,
            risk_config=risk_cfg,
            required_columns=required,
        )

    # Walk-forward on strongest few (validation windows already chosen; WF is extra)
    from app.backtest.robust import generate_walk_forward_windows

    wf_windows = generate_walk_forward_windows(
        overall_start=train_start,
        overall_end=test_end,
        train_days=180,
        test_days=90,
        step_days=90,
    )
    for candidate in accepted[:top_k_walkforward]:
        history = prepare_candidate_history(ohlcv, candidate.params)
        required = (
            REQUIRED_INDICATORS
            if candidate.params.variant == "baseline"
            else RESEARCH_REQUIRED_COLUMNS
        )
        risk_cfg = RiskConfig(
            atr_multiplier=candidate.params.atr_stop_multiplier,
            risk_reward_ratio=candidate.params.risk_reward_ratio,
        )
        for i, (_tr_s, _tr_e, te_s, te_e) in enumerate(wf_windows, start=1):
            # Evaluate only the unseen test segment of each fold
            period = run_period(
                history,
                label=f"wf{i}_test:{candidate.params.variant}",
                start=te_s,
                end=te_e,
                instrument=instrument,
                timeframe=timeframe,
                use_risk=use_risk,
                risk_config=risk_cfg,
                required_columns=required,
                warmup_bars=WARMUP_BARS,
            )
            candidate.walk_forward_tests.append(period)

    return {
        "grid_size": len(grid),
        "evaluated": ranked,
        "accepted": accepted,
        "validation_winner": accepted[0] if accepted else None,
        "ohlcv_rows": len(ohlcv),
    }
