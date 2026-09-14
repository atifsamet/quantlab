"""
Robust multi-period backtesting (Phase 7).

Train/test and walk-forward evaluate the **unchanged** Phase 3 strategy on
separate windows. Indicators are always computed on a history that includes
warm-up bars *before* each evaluation start (no look-ahead into the future
beyond each period's end).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.backtest.analysis import RobustMetrics, compute_robust_metrics, format_robust_section
from app.backtest.engine import BacktestConfig, BacktestEngine, load_ohlcv_csv
from app.backtest.models import BacktestResult
from app.backtest.report import (
    BacktestReport,
    build_report,
    export_equity_curve,
    export_trades,
)
from app.data.okx_historical import filter_date_range, parse_utc_timestamp
from app.indicators import WARMUP_BARS, add_indicators
from app.risk import RiskConfig, RiskManager
from app.strategy import generate_signals
from app.strategy.baseline import REQUIRED_INDICATORS


@dataclass(slots=True)
class PeriodRun:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    result: BacktestResult
    metrics: RobustMetrics
    report: BacktestReport
    frame: pd.DataFrame


@dataclass(slots=True)
class WalkForwardFold:
    fold_index: int
    train: PeriodRun
    test: PeriodRun


def load_signaled_history(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV and compute indicators + signals on the full available history."""
    raw = load_ohlcv_csv(csv_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    with_indicators = add_indicators(raw)
    return generate_signals(with_indicators)


def assert_warmup_available(
    history: pd.DataFrame,
    eval_start: pd.Timestamp,
    *,
    warmup_bars: int = WARMUP_BARS,
) -> None:
    """Ensure enough candles exist before eval_start for EMA warm-up."""
    prior = history[history["timestamp"] < eval_start]
    if len(prior) < warmup_bars:
        raise ValueError(
            f"Insufficient warm-up before {eval_start}: "
            f"need {warmup_bars} prior candles, found {len(prior)}. "
            "Download earlier history or reduce --warmup-bars."
        )


def assert_eval_indicators_ready(
    eval_frame: pd.DataFrame,
    required_columns: tuple[str, ...] = REQUIRED_INDICATORS,
) -> None:
    """First evaluation row must have required indicator columns defined."""
    if eval_frame.empty:
        raise ValueError("Evaluation frame is empty")
    first = eval_frame.iloc[0]
    for col in required_columns:
        if col not in eval_frame.columns or pd.isna(first[col]):
            raise ValueError(
                f"Indicator '{col}' is NaN at evaluation start "
                f"{first['timestamp']} - warm-up was insufficient or sliced incorrectly"
            )


def slice_evaluation(
    history: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    warmup_bars: int = WARMUP_BARS,
    require_ready_indicators: bool = True,
    required_columns: tuple[str, ...] = REQUIRED_INDICATORS,
) -> pd.DataFrame:
    """
    Return evaluation window rows with indicators already computed on full history.

    Does not include future candles after ``end``.
    """
    start_ts = parse_utc_timestamp(start)
    end_ts = parse_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must be >= start")
    assert_warmup_available(history, start_ts, warmup_bars=warmup_bars)
    evaluation = filter_date_range(history, start=start_ts, end=end_ts)
    if evaluation.empty:
        raise ValueError(f"No candles in evaluation window {start_ts} -> {end_ts}")
    if require_ready_indicators:
        assert_eval_indicators_ready(evaluation, required_columns=required_columns)
    # Hard guarantee: no rows beyond end
    if evaluation["timestamp"].max() > end_ts:
        raise RuntimeError("Evaluation slice leaked candles after end")
    return evaluation.reset_index(drop=True)


def run_period(
    history: pd.DataFrame,
    *,
    label: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    instrument: str,
    timeframe: str,
    use_risk: bool = True,
    warmup_bars: int = WARMUP_BARS,
    config: BacktestConfig | None = None,
    risk_config: RiskConfig | None = None,
    required_columns: tuple[str, ...] = REQUIRED_INDICATORS,
) -> PeriodRun:
    """Backtest one date window on a pre-signaled history frame."""
    start_ts = parse_utc_timestamp(start)
    end_ts = parse_utc_timestamp(end)
    frame = slice_evaluation(
        history,
        start_ts,
        end_ts,
        warmup_bars=warmup_bars,
        required_columns=required_columns,
    )
    cfg = config or BacktestConfig()
    risk_manager = None
    if use_risk:
        risk_manager = RiskManager(risk_config or RiskConfig())
    engine = BacktestEngine(config=cfg, risk_manager=risk_manager)
    result = engine.run(frame, compute_indicators=False, compute_signals=False)
    metrics = compute_robust_metrics(result, period_start=start_ts, period_end=end_ts)
    report = build_report(
        result,
        instrument=instrument,
        timeframe=timeframe,
        evaluation_start=start_ts,
        evaluation_end=end_ts,
        warmup_bars=warmup_bars,
        candle_count=len(frame),
    )
    return PeriodRun(
        label=label,
        start=start_ts,
        end=end_ts,
        result=result,
        metrics=metrics,
        report=report,
        frame=frame,
    )


def generate_walk_forward_windows(
    *,
    overall_start: str | pd.Timestamp,
    overall_end: str | pd.Timestamp,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Yield (train_start, train_end, test_start, test_end) windows.

    Windows advance by ``step_days`` (default = test_days). No parameter fitting
    is performed — both segments only evaluate the fixed baseline strategy.
    """
    if train_days < 1 or test_days < 1:
        raise ValueError("train_days and test_days must be >= 1")
    step = test_days if step_days is None else step_days
    if step < 1:
        raise ValueError("step_days must be >= 1")

    start = parse_utc_timestamp(overall_start)
    end = parse_utc_timestamp(overall_end)
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    guard = 0
    while guard < 10_000:
        guard += 1
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.Timedelta(days=step)
    return windows


def run_walk_forward(
    history: pd.DataFrame,
    *,
    overall_start: str | pd.Timestamp,
    overall_end: str | pd.Timestamp,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    instrument: str,
    timeframe: str,
    use_risk: bool = True,
    warmup_bars: int = WARMUP_BARS,
) -> list[WalkForwardFold]:
    folds: list[WalkForwardFold] = []
    windows = generate_walk_forward_windows(
        overall_start=overall_start,
        overall_end=overall_end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        train = run_period(
            history,
            label=f"WF{i}-train",
            start=tr_s,
            end=tr_e,
            instrument=instrument,
            timeframe=timeframe,
            use_risk=use_risk,
            warmup_bars=warmup_bars,
        )
        test = run_period(
            history,
            label=f"WF{i}-test",
            start=te_s,
            end=te_e,
            instrument=instrument,
            timeframe=timeframe,
            use_risk=use_risk,
            warmup_bars=warmup_bars,
        )
        folds.append(WalkForwardFold(fold_index=i, train=train, test=test))
    return folds


def export_period_artifacts(
    period: PeriodRun,
    *,
    results_dir: str | Path = "data/results",
    prefix: str | None = None,
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    safe = period.report.instrument.replace("/", "-").upper()
    tf = period.report.timeframe
    tag = prefix or period.label.replace(" ", "_")
    equity_path = results_dir / f"{safe}_{tf}_{tag}_equity.csv"
    trades_path = results_dir / f"{safe}_{tf}_{tag}_trades.csv"
    summary_csv = results_dir / f"{safe}_{tf}_{tag}_summary.csv"
    summary_json = results_dir / f"{safe}_{tf}_{tag}_summary.json"

    export_equity_curve(period.result, equity_path)
    export_trades(period.result, trades_path)

    summary = {
        **period.report.to_dict(),
        "label": period.label,
        "robust": period.metrics.to_dict(),
    }
    pd.json_normalize(summary).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Monthly returns CSV
    monthly_path = results_dir / f"{safe}_{tf}_{tag}_monthly.csv"
    pd.DataFrame([asdict(m) for m in period.metrics.monthly_returns]).to_csv(
        monthly_path, index=False
    )

    return {
        "equity": equity_path,
        "trades": trades_path,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "monthly": monthly_path,
    }


def format_comparison(phase4: PeriodRun, phase5: PeriodRun) -> str:
    lines = [
        "=== Phase 4 (no risk) vs Phase 5 (risk-managed) ===",
        f"{'Metric':<28} {'Phase4':>14} {'Phase5':>14}",
        f"{'Total return %':<28} {phase4.metrics.total_return_pct:>14.4f} {phase5.metrics.total_return_pct:>14.4f}",
        f"{'Total PnL':<28} {phase4.metrics.total_pnl:>14.4f} {phase5.metrics.total_pnl:>14.4f}",
        f"{'Trades':<28} {phase4.metrics.total_trades:>14} {phase5.metrics.total_trades:>14}",
        f"{'Win rate %':<28} {phase4.metrics.win_rate:>14.2f} {phase5.metrics.win_rate:>14.2f}",
        f"{'Profit factor':<28} {phase4.metrics.profit_factor:>14.4f} {phase5.metrics.profit_factor:>14.4f}",
        f"{'Max drawdown %':<28} {phase4.metrics.max_drawdown_pct:>14.4f} {phase5.metrics.max_drawdown_pct:>14.4f}",
        f"{'Expectancy':<28} {phase4.metrics.expectancy_per_trade:>14.4f} {phase5.metrics.expectancy_per_trade:>14.4f}",
        f"{'Time in market %':<28} {phase4.metrics.time_in_market_pct:>14.2f} {phase5.metrics.time_in_market_pct:>14.2f}",
        f"{'Total fees':<28} {phase4.metrics.total_fees:>14.4f} {phase5.metrics.total_fees:>14.4f}",
    ]
    return "\n".join(lines)


def format_period_block(period: PeriodRun) -> str:
    header = (
        f"=== {period.label} ===\n"
        f"Window: {period.start} -> {period.end}\n"
        f"{period.report.format_text()}\n"
        f"{format_robust_section('Robust metrics', period.metrics)}"
    )
    return header
