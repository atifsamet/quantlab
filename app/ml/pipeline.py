"""
Phase 10 research pipeline: train models, select threshold on validation,
evaluate once on final test + walk-forward. No live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from app.backtest.engine import BacktestConfig, load_ohlcv_csv
from app.backtest.robust import PeriodRun, generate_walk_forward_windows, run_period
from app.indicators import WARMUP_BARS
from app.ml import LABEL_DOC
from app.ml.dataset import (
    build_multi_asset_dataset,
    chronological_split,
    dataset_stats,
    prepare_signaled_frame,
)
from app.ml.filter import DEFAULT_THRESHOLDS, apply_probability_filter
from app.ml.labels import LabelConfig
from app.ml.models import (
    ModelName,
    build_model,
    evaluate_classification,
    feature_importance,
    fit_model,
)
from app.research.indicators_flex import RESEARCH_REQUIRED_COLUMNS
from app.research.params import ResearchParams
from app.research.universe import DEFAULT_SYMBOLS, csv_path_for, discover_available
from app.risk.config import RiskConfig


@dataclass(slots=True)
class ThresholdCandidate:
    threshold: float
    model_name: str
    val_return: float
    val_trades: int
    val_pf: float
    val_dd: float
    score: float


@dataclass(slots=True)
class Phase10Outcome:
    params: ResearchParams
    label_cfg: LabelConfig
    dataset_stats_all: dict[str, Any]
    dataset_stats_train: dict[str, Any]
    dataset_stats_val: dict[str, Any]
    dataset_stats_test: dict[str, Any]
    classification: list[dict[str, Any]]
    threshold_grid: list[dict[str, Any]]
    selected_model: str | None
    selected_threshold: float | None
    feature_importance: list[dict[str, Any]]
    baseline_val: list[dict[str, Any]] = field(default_factory=list)
    filtered_val: list[dict[str, Any]] = field(default_factory=list)
    baseline_test: list[dict[str, Any]] = field(default_factory=list)
    filtered_test: list[dict[str, Any]] = field(default_factory=list)
    walk_forward: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "Evidence insufficient"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_definition": LABEL_DOC,
            "params": self.params.to_dict(),
            "label_cfg": {
                "atr_multiplier": self.label_cfg.atr_multiplier,
                "risk_reward_ratio": self.label_cfg.risk_reward_ratio,
                "horizon": self.label_cfg.horizon,
            },
            "dataset_stats_all": self.dataset_stats_all,
            "dataset_stats_train": self.dataset_stats_train,
            "dataset_stats_val": self.dataset_stats_val,
            "dataset_stats_test": self.dataset_stats_test,
            "classification": self.classification,
            "threshold_grid": self.threshold_grid,
            "selected_model": self.selected_model,
            "selected_threshold": self.selected_threshold,
            "feature_importance": self.feature_importance,
            "baseline_val": self.baseline_val,
            "filtered_val": self.filtered_val,
            "baseline_test": self.baseline_test,
            "filtered_test": self.filtered_test,
            "walk_forward": self.walk_forward,
            "verdict": self.verdict,
            "notes": self.notes,
        }


def _period_summary(period: PeriodRun) -> dict[str, Any]:
    m = period.metrics
    return {
        "label": period.label,
        "symbol": period.report.instrument,
        "timeframe": period.report.timeframe,
        "return_pct": m.total_return_pct,
        "trades": m.total_trades,
        "win_rate": m.win_rate,
        "profit_factor": m.profit_factor if m.profit_factor != float("inf") else None,
        "expectancy": m.expectancy_per_trade,
        "max_dd": m.max_drawdown_pct,
        "long_trades": m.long.trade_count,
        "short_trades": m.short.trade_count,
        "time_in_market": m.time_in_market_pct,
        "max_consec_losses": m.max_consecutive_losses,
        "total_fees": m.total_fees,
    }


def _backtest_symbol_window(
    framed: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    label: str,
    params: ResearchParams,
    use_risk: bool = True,
) -> PeriodRun:
    risk_cfg = RiskConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
    )
    return run_period(
        framed,
        label=label,
        start=start,
        end=end,
        instrument=symbol,
        timeframe=timeframe,
        use_risk=use_risk,
        risk_config=risk_cfg,
        required_columns=RESEARCH_REQUIRED_COLUMNS,
        config=BacktestConfig(),
        warmup_bars=WARMUP_BARS,
    )


def run_phase10_pipeline(
    *,
    data_dir: str | Path = "data/historical",
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    timeframe: str = "4h",
    train_start: str = "2024-09-01",
    train_end: str = "2025-03-01",
    val_start: str = "2025-03-01",
    val_end: str = "2025-09-01",
    test_start: str = "2025-09-01",
    test_end: str = "2026-09-01",
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    min_train_samples: int = 80,
    min_val_trades: int = 4,
    also_1h: bool = False,
) -> Phase10Outcome:
    """
    Full Phase 10 research pass.

    Model + threshold selected on TRAIN+VALIDATION backtests only.
    FINAL TEST evaluated once afterward.
    """
    params = ResearchParams(variant="breakout", breakout_lookback=20)
    label_cfg = LabelConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
        horizon=24 if timeframe == "4h" else 48,
    )
    notes: list[str] = [
        "ML is a signal filter only; risk manager still sizes/stops.",
        "No final-test tuning of model or threshold.",
        "Shared model across assets (no per-asset optimization).",
    ]

    tfs = (timeframe,) + (("1h",) if also_1h and timeframe != "1h" else ())
    available = discover_available(data_dir, symbols, tfs)
    primary = [(s, t, p) for s, t, p in available if t == timeframe]
    if not primary:
        raise FileNotFoundError(f"No CSVs for timeframe={timeframe} under {data_dir}")

    samples = build_multi_asset_dataset(primary, params, label_cfg=label_cfg, include_asset_id=False)
    train, val, test = chronological_split(
        samples,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )
    stats_all = dataset_stats(samples).to_dict()
    stats_train = dataset_stats(train).to_dict()
    stats_val = dataset_stats(val).to_dict()
    stats_test = dataset_stats(test).to_dict()

    outcome = Phase10Outcome(
        params=params,
        label_cfg=label_cfg,
        dataset_stats_all=stats_all,
        dataset_stats_train=stats_train,
        dataset_stats_val=stats_val,
        dataset_stats_test=stats_test,
        classification=[],
        threshold_grid=[],
        selected_model=None,
        selected_threshold=None,
        feature_importance=[],
        notes=notes,
    )

    if len(train) < min_train_samples:
        outcome.verdict = "Evidence insufficient"
        outcome.notes.append(
            f"Train samples {len(train)} < minimum {min_train_samples}; skipping model fit."
        )
        return outcome

    # Load framed histories once
    framed_cache: dict[str, pd.DataFrame] = {}
    for symbol, _tf, path in primary:
        ohlcv = load_ohlcv_csv(path)
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        framed_cache[symbol] = prepare_signaled_frame(ohlcv, params)

    # Baseline validation (deterministic, no ML)
    for symbol, framed in framed_cache.items():
        period = _backtest_symbol_window(
            framed,
            symbol=symbol,
            timeframe=timeframe,
            start=val_start,
            end=val_end,
            label=f"baseline_val:{symbol}",
            params=params,
        )
        outcome.baseline_val.append(_period_summary(period))

    model_names: list[ModelName] = ["logistic", "random_forest", "gradient_boosting"]
    fitted: dict[str, Pipeline] = {}
    for name in model_names:
        model = fit_model(build_model(name), train, name)
        fitted[name] = model
        outcome.classification.append(evaluate_classification(model, train, f"{name}:train").to_dict())
        outcome.classification.append(evaluate_classification(model, val, f"{name}:val").to_dict())

    # Threshold selection on validation backtests only (aggregate across assets)
    best: ThresholdCandidate | None = None
    for name, model in fitted.items():
        for thr in thresholds:
            val_periods: list[PeriodRun] = []
            for symbol, framed in framed_cache.items():
                filtered = apply_probability_filter(framed, model, threshold=thr)
                period = _backtest_symbol_window(
                    filtered,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=val_start,
                    end=val_end,
                    label=f"ml_val:{name}:{thr}:{symbol}",
                    params=params,
                )
                val_periods.append(period)
            mean_ret = float(pd.Series([p.metrics.total_return_pct for p in val_periods]).mean())
            mean_trades = float(pd.Series([p.metrics.total_trades for p in val_periods]).mean())
            mean_dd = float(pd.Series([p.metrics.max_drawdown_pct for p in val_periods]).mean())
            pfs = []
            for p in val_periods:
                pf = p.metrics.profit_factor
                pfs.append(3.0 if pf == float("inf") else float(pf))
            mean_pf = float(pd.Series(pfs).mean())
            # Composite score from mean period metrics
            score = (
                mean_ret
                + 2.0 * min(mean_pf, 5.0)
                + min(mean_trades, 30) / 30.0 * 3.0
                - 1.2 * mean_dd
            )
            if mean_trades < min_val_trades:
                score = float("-inf")
            cand = ThresholdCandidate(
                threshold=thr,
                model_name=name,
                val_return=mean_ret,
                val_trades=int(round(mean_trades)),
                val_pf=mean_pf,
                val_dd=mean_dd,
                score=score,
            )
            outcome.threshold_grid.append(
                {
                    "model": name,
                    "threshold": thr,
                    "mean_val_return": mean_ret,
                    "mean_val_trades": mean_trades,
                    "mean_val_pf": mean_pf,
                    "mean_val_dd": mean_dd,
                    "score": score,
                }
            )
            if best is None or cand.score > best.score:
                best = cand

    if best is None or best.score == float("-inf"):
        outcome.verdict = "No meaningful improvement"
        outcome.notes.append("No model/threshold met validation trade-count / score requirements.")
        return outcome

    outcome.selected_model = best.model_name
    outcome.selected_threshold = best.threshold
    selected_model = fitted[best.model_name]
    outcome.feature_importance = feature_importance(
        selected_model, best.model_name  # type: ignore[arg-type]
    ).to_dict(orient="records")

    # Filtered validation detail
    for symbol, framed in framed_cache.items():
        filtered = apply_probability_filter(
            framed, selected_model, threshold=best.threshold
        )
        period = _backtest_symbol_window(
            filtered,
            symbol=symbol,
            timeframe=timeframe,
            start=val_start,
            end=val_end,
            label=f"ml_val_selected:{symbol}",
            params=params,
        )
        outcome.filtered_val.append(_period_summary(period))

    # FINAL TEST once (baseline + filtered) — no tuning
    for symbol, framed in framed_cache.items():
        base = _backtest_symbol_window(
            framed,
            symbol=symbol,
            timeframe=timeframe,
            start=test_start,
            end=test_end,
            label=f"baseline_test:{symbol}",
            params=params,
        )
        outcome.baseline_test.append(_period_summary(base))
        filtered = apply_probability_filter(
            framed, selected_model, threshold=best.threshold
        )
        filt = _backtest_symbol_window(
            filtered,
            symbol=symbol,
            timeframe=timeframe,
            start=test_start,
            end=test_end,
            label=f"ml_test:{symbol}",
            params=params,
        )
        outcome.filtered_test.append(_period_summary(filt))

    # Walk-forward: retrain on past samples, filter next unseen window (BTC primary)
    wf_windows = generate_walk_forward_windows(
        overall_start=train_start,
        overall_end=test_end,
        train_days=180,
        test_days=90,
        step_days=90,
    )
    btc_path = csv_path_for(data_dir, "BTC-USDT", timeframe)
    if btc_path.is_file() and "BTC-USDT" in framed_cache:
        ts_all = pd.to_datetime(samples["timestamp"], utc=True)
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(wf_windows, start=1):
            train_fold = samples[(ts_all >= tr_s) & (ts_all < tr_e)].reset_index(drop=True)
            if len(train_fold) < max(40, min_train_samples // 2):
                outcome.walk_forward.append(
                    {
                        "fold": i,
                        "skipped": True,
                        "reason": f"too few train samples ({len(train_fold)})",
                    }
                )
                continue
            fold_model = fit_model(build_model(best.model_name), train_fold, best.model_name)  # type: ignore[arg-type]
            framed = framed_cache["BTC-USDT"]
            filtered = apply_probability_filter(
                framed, fold_model, threshold=best.threshold
            )
            base = _backtest_symbol_window(
                framed,
                symbol="BTC-USDT",
                timeframe=timeframe,
                start=te_s,
                end=te_e,
                label=f"wf{i}_base",
                params=params,
            )
            filt = _backtest_symbol_window(
                filtered,
                symbol="BTC-USDT",
                timeframe=timeframe,
                start=te_s,
                end=te_e,
                label=f"wf{i}_ml",
                params=params,
            )
            outcome.walk_forward.append(
                {
                    "fold": i,
                    "skipped": False,
                    "train_start": str(tr_s),
                    "train_end": str(tr_e),
                    "test_start": str(te_s),
                    "test_end": str(te_e),
                    "baseline": _period_summary(base),
                    "filtered": _period_summary(filt),
                }
            )

    # Verdict from OOS comparison (not classification metrics alone).
    # Require meaningful discrimination AND breadth; do not force positivity.
    base_rets = [r["return_pct"] for r in outcome.baseline_test]
    filt_rets = [r["return_pct"] for r in outcome.filtered_test]
    base_mean = float(pd.Series(base_rets).mean()) if base_rets else 0.0
    filt_mean = float(pd.Series(filt_rets).mean()) if filt_rets else 0.0
    improved_assets = sum(
        1
        for b, f in zip(outcome.baseline_test, outcome.filtered_test, strict=False)
        if f["return_pct"] > b["return_pct"] + 0.25 and f["trades"] >= 3
    )
    worsened_assets = sum(
        1
        for b, f in zip(outcome.baseline_test, outcome.filtered_test, strict=False)
        if f["return_pct"] < b["return_pct"] - 0.25
    )
    wf_improve = 0
    wf_total = 0
    for fold in outcome.walk_forward:
        if fold.get("skipped"):
            continue
        wf_total += 1
        if fold["filtered"]["return_pct"] > fold["baseline"]["return_pct"] + 0.25:
            wf_improve += 1

    selected_val_roc = None
    for row in outcome.classification:
        if row["model"] == f"{best.model_name}:val":
            selected_val_roc = row.get("roc_auc")
            break

    weak_classifier = selected_val_roc is None or selected_val_roc < 0.55
    if weak_classifier:
        outcome.notes.append(
            f"Validation ROC-AUC for {best.model_name} is {selected_val_roc} "
            "(< 0.55): little evidence of useful ranking beyond noise."
        )

    if (
        not weak_classifier
        and filt_mean > base_mean + 0.5
        and improved_assets >= max(3, (len(filt_rets) + 1) // 2)
        and worsened_assets == 0
        and (wf_total == 0 or wf_improve >= max(3, (wf_total + 1) // 2))
    ):
        outcome.verdict = "Robust improvement found"
        outcome.notes.append(
            "OOS mean and breadth improved vs baseline with non-trivial validation AUC, "
            "but this still does not authorize paper/live trading."
        )
    elif filt_mean <= base_mean and improved_assets <= 1:
        outcome.verdict = "No meaningful improvement"
    else:
        outcome.verdict = "Evidence insufficient"

    outcome.notes.append(
        f"Selected {best.model_name} @ threshold={best.threshold:.2f} "
        f"(val mean return={best.val_return:.2f}%, trades~{best.val_trades})."
    )
    outcome.notes.append(
        f"Final-test mean baseline={base_mean:.2f}% vs ML-filtered={filt_mean:.2f}% "
        f"(assets improved={improved_assets}/{len(filt_rets)}; worsened={worsened_assets}; "
        f"WF improved={wf_improve}/{wf_total})."
    )
    return outcome
