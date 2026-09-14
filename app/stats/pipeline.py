"""
Phase 11 research pipeline — validate breakout@4h statistically.

No parameter tuning. Final-test window used for reporting only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.backtest.engine import BacktestConfig, load_ohlcv_csv
from app.research.params import ResearchParams
from app.research.universe import DEFAULT_SYMBOLS, csv_path_for, discover_available
from app.risk.config import RiskConfig
from app.stats import MIN_TRADES_FOR_INFERENCE, N_SIMULATIONS, RANDOM_SEED
from app.stats.benchmarks import (
    assign_constant_signal,
    assign_random_entry_matched_count,
    assign_random_signals,
    buy_and_hold,
    measure_signal_stats,
    prepare_breakout_frame,
    run_signaled_backtest,
    slice_eval_frame,
)
from app.stats.bootstrap import bootstrap_trade_metrics
from app.stats.monte_carlo import run_trade_monte_carlo
from app.stats.plots import save_benchmark_bars, save_equity_curve, save_histogram
from app.stats.risk_metrics import compute_risk_adjusted
from app.stats.sensitivity import (
    DEFAULT_COST_SCENARIOS,
    cost_sensitivity,
    slippage_break_even,
)


@dataclass(slots=True)
class Phase11Outcome:
    verdict: str
    answers: dict[str, str]
    notes: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


def _result_row(name: str, symbol: str, result, risk, **extra: Any) -> dict[str, Any]:
    row = {
        "name": name,
        "symbol": symbol,
        "return_pct": float(result.total_return_pct),
        "trades": int(result.total_trades),
        "win_rate": float(result.win_rate),
        "profit_factor": (
            None
            if result.profit_factor == float("inf")
            else float(result.profit_factor)
        ),
        "expectancy": float(risk.expectancy),
        "max_dd": float(result.max_drawdown_pct),
        "sharpe": risk.sharpe,
        "sortino": risk.sortino,
        "calmar": risk.calmar,
        "avg_win": risk.average_win,
        "avg_loss": risk.average_loss,
        "payoff_ratio": risk.payoff_ratio,
        "recovery_factor": risk.recovery_factor,
        "dd_duration_bars": risk.max_drawdown_duration_bars,
        "annualized": risk.annualized,
        "n_bars": risk.n_bars,
    }
    row.update(extra)
    return row


def run_phase11_pipeline(
    *,
    data_dir: str | Path = "data/historical",
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    timeframe: str = "4h",
    # Reporting window = FINAL TEST (untouched for decisions; no tuning here)
    eval_start: str = "2025-09-01",
    eval_end: str = "2026-09-01",
    n_sims: int = N_SIMULATIONS,
    n_random_seeds: int = 200,
    seed: int = RANDOM_SEED,
    export_dir: str | Path = "data/results/phase11",
) -> Phase11Outcome:
    params = ResearchParams(variant="breakout", breakout_lookback=20)
    risk_cfg = RiskConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
    )
    base_cfg = BacktestConfig()
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)
    notes = [
        "Phase 11 validates fixed breakout@4h only — no new optimization.",
        f"Evaluation window for reporting: {eval_start} -> {eval_end} (final-test style).",
        "Monte Carlo / bootstrap are model-based uncertainty, not future guarantees.",
    ]

    available = discover_available(data_dir, symbols, (timeframe,))
    if not available:
        raise FileNotFoundError(f"No {timeframe} CSVs under {data_dir}")

    benchmark_rows: list[dict[str, Any]] = []
    statistical_rows: list[dict[str, Any]] = []
    per_asset: list[dict[str, Any]] = []
    all_strategy_trades = []
    strategy_results: dict[str, Any] = {}
    framed_evals: dict[str, pd.DataFrame] = {}

    for symbol, tf, path in available:
        ohlcv = load_ohlcv_csv(path)
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        history = prepare_breakout_frame(ohlcv, params)
        eval_frame = slice_eval_frame(history, eval_start, eval_end)
        framed_evals[symbol] = eval_frame
        sig_stats = measure_signal_stats(eval_frame)

        # 6) Strategy
        strat_res = run_signaled_backtest(
            eval_frame, use_risk=True, config=base_cfg, risk_config=risk_cfg
        )
        strat_risk = compute_risk_adjusted(strat_res, timeframe=tf)
        strategy_results[symbol] = strat_res
        all_strategy_trades.extend(strat_res.trades)
        per_asset.append(
            _result_row(
                "breakout",
                symbol,
                strat_res,
                strat_risk,
                long_signal_pct=round(sig_stats.long_pct * 100, 3),
                short_signal_pct=round(sig_stats.short_pct * 100, 3),
            )
        )
        statistical_rows.append(_result_row("breakout", symbol, strat_res, strat_risk))

        # 1) Buy and hold
        bh = buy_and_hold(
            eval_frame,
            initial_capital=base_cfg.initial_capital,
            fee_rate=base_cfg.fee_rate,
            slippage_rate=base_cfg.slippage_rate,
        )
        bh_risk = compute_risk_adjusted(bh, timeframe=tf, min_trades_for_annualization=1)
        benchmark_rows.append(_result_row("buy_and_hold", symbol, bh, bh_risk))

        # 2/3 Always LONG / SHORT
        for const in ("LONG", "SHORT"):
            framed = assign_constant_signal(eval_frame, const)
            res = run_signaled_backtest(
                framed, use_risk=True, config=base_cfg, risk_config=risk_cfg
            )
            risk = compute_risk_adjusted(res, timeframe=tf)
            benchmark_rows.append(_result_row(f"always_{const.lower()}", symbol, res, risk))

        # 4) Random matched frequency (primary seed)
        rnd = assign_random_signals(eval_frame, sig_stats, seed=seed)
        rnd_res = run_signaled_backtest(
            rnd, use_risk=True, config=base_cfg, risk_config=risk_cfg
        )
        rnd_risk = compute_risk_adjusted(rnd_res, timeframe=tf)
        benchmark_rows.append(
            _result_row("random_matched_frequency", symbol, rnd_res, rnd_risk, seed=seed)
        )

        # 5) Random matched approximate trade-signal count
        target = max(1, sig_stats.n_long + sig_stats.n_short)
        long_share = (
            sig_stats.n_long / target if target else 0.5
        )
        rnd2 = assign_random_entry_matched_count(
            eval_frame,
            target_trade_signals=target,
            long_share=long_share,
            seed=seed,
        )
        rnd2_res = run_signaled_backtest(
            rnd2, use_risk=True, config=base_cfg, risk_config=risk_cfg
        )
        rnd2_risk = compute_risk_adjusted(rnd2_res, timeframe=tf)
        benchmark_rows.append(
            _result_row("random_matched_count", symbol, rnd2_res, rnd2_risk, seed=seed)
        )

    # Pooled portfolio: equal-weight average of per-asset strategy returns + concat trades MC
    strat_rets = [r["return_pct"] for r in per_asset]
    pooled_mean_return = float(np.mean(strat_rets)) if strat_rets else 0.0

    # Random-frequency null distribution on BTC (activity-matched), many seeds
    btc_frame = framed_evals["BTC-USDT"] if "BTC-USDT" in framed_evals else next(iter(framed_evals.values()))
    btc_stats = measure_signal_stats(btc_frame)
    btc_strat = (
        strategy_results["BTC-USDT"]
        if "BTC-USDT" in strategy_results
        else next(iter(strategy_results.values()))
    )
    random_returns = []
    for s in range(n_random_seeds):
        rnd = assign_random_signals(btc_frame, btc_stats, seed=seed + 1000 + s)
        res = run_signaled_backtest(
            rnd, use_risk=True, config=base_cfg, risk_config=risk_cfg
        )
        random_returns.append(float(res.total_return_pct))
    random_returns_arr = np.asarray(random_returns, dtype=float)
    observed_btc = float(btc_strat.total_return_pct)
    pct_random_below = float(np.mean(random_returns_arr < observed_btc))
    pct_random_worse_or_eq = float(np.mean(random_returns_arr <= observed_btc))

    # Monte Carlo on pooled strategy trades + BTC alone
    mc_pooled, mc_rets, mc_dds = run_trade_monte_carlo(
        all_strategy_trades,
        initial_capital=base_cfg.initial_capital,
        n_sims=n_sims,
        seed=seed,
        observed_return_pct=pooled_mean_return,
    )
    mc_btc, mc_btc_rets, mc_btc_dds = run_trade_monte_carlo(
        btc_strat.trades,
        initial_capital=base_cfg.initial_capital,
        n_sims=n_sims,
        seed=seed + 1,
        observed_return_pct=observed_btc,
        observed_max_dd_pct=float(btc_strat.max_drawdown_pct),
    )

    # Bootstrap
    boot_pooled = bootstrap_trade_metrics(
        all_strategy_trades,
        initial_capital=base_cfg.initial_capital,
        n_boot=n_sims,
        seed=seed,
    )
    boot_btc = bootstrap_trade_metrics(
        btc_strat.trades,
        initial_capital=base_cfg.initial_capital,
        n_boot=n_sims,
        seed=seed + 2,
    )

    # Cost / slippage sensitivity on pooled? Use BTC eval frame for clarity
    cost_rows = cost_sensitivity(btc_frame, scenarios=DEFAULT_COST_SCENARIOS, params=params)
    # Also multi-asset mean under BASE costs already in per_asset
    slip = slippage_break_even(btc_frame, fee_rate=0.001, params=params)

    # Multi-asset cost: mean return under each scenario
    cost_multi: list[dict[str, Any]] = []
    for sc in DEFAULT_COST_SCENARIOS:
        rets = []
        for symbol, framed in framed_evals.items():
            rows = cost_sensitivity(framed, scenarios=(sc,), params=params)
            rets.append(rows[0].return_pct)
        cost_multi.append(
            {
                "scenario": sc.name,
                "fee_rate": sc.fee_rate,
                "slippage_rate": sc.slippage_rate,
                "mean_return_pct": float(np.mean(rets)),
                "assets": len(rets),
            }
        )

    # Exports
    pd.DataFrame(statistical_rows).to_csv(export / "statistical_summary.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(export / "benchmark_summary.csv", index=False)
    mc_df = pd.DataFrame(
        [
            {**mc_pooled.to_dict(), "scope": "pooled"},
            {**mc_btc.to_dict(), "scope": "BTC"},
        ]
    )
    mc_df.to_csv(export / "monte_carlo_summary.csv", index=False)
    boot_rows = [{**b.to_dict(), "scope": "pooled"} for b in boot_pooled] + [
        {**b.to_dict(), "scope": "BTC"} for b in boot_btc
    ]
    pd.DataFrame(boot_rows).to_csv(export / "bootstrap_summary.csv", index=False)
    pd.DataFrame([c.to_dict() for c in cost_rows]).to_csv(
        export / "cost_sensitivity_btc.csv", index=False
    )
    pd.DataFrame(cost_multi).to_csv(export / "cost_sensitivity.csv", index=False)
    rows_dd = []
    for scope, arr in (("pooled", mc_dds), ("BTC", mc_btc_dds)):
        if len(arr) == 0:
            continue
        for p in (5, 25, 50, 75, 95):
            rows_dd.append(
                {"scope": scope, "percentile": p, "max_dd": float(np.percentile(arr, p))}
            )
    pd.DataFrame(rows_dd).to_csv(export / "drawdown_distribution.csv", index=False)

    # Plots
    plots_written: list[str] = []
    if save_histogram(
        mc_rets,
        path=export / "mc_return_distribution.png",
        title="Monte Carlo returns (pooled trades)",
        xlabel="return %",
        observed=pooled_mean_return,
    ):
        plots_written.append("mc_return_distribution.png")
    if len(mc_dds) and save_histogram(
        mc_dds,
        path=export / "mc_drawdown_distribution.png",
        title="Monte Carlo max DD (pooled trades)",
        xlabel="max DD %",
        observed=None,
    ):
        plots_written.append("mc_drawdown_distribution.png")
    if save_histogram(
        random_returns_arr,
        path=export / "strategy_vs_random.png",
        title="Random matched-frequency returns (BTC)",
        xlabel="return %",
        observed=observed_btc,
    ):
        plots_written.append("strategy_vs_random.png")
    if btc_strat.equity_curve:
        if save_equity_curve(
            [p.timestamp for p in btc_strat.equity_curve],
            [p.equity for p in btc_strat.equity_curve],
            path=export / "equity_curve_btc.png",
            title="breakout@4h equity (BTC, eval window)",
        ):
            plots_written.append("equity_curve_btc.png")
    btc_symbol = "BTC-USDT" if "BTC-USDT" in framed_evals else per_asset[0]["symbol"]
    btc_bench = [r for r in benchmark_rows if r["symbol"] == btc_symbol]
    btc_bench.append(
        next(r for r in statistical_rows if r["symbol"] == btc_symbol and r["name"] == "breakout")
    )
    if save_benchmark_bars(
        [r["name"] for r in btc_bench],
        [r["return_pct"] for r in btc_bench],
        path=export / "benchmark_bars_btc.png",
        title="BTC benchmarks vs breakout (eval window)",
    ):
        plots_written.append("benchmark_bars_btc.png")

    # Verdict answers
    mean_bh = float(
        np.mean([r["return_pct"] for r in benchmark_rows if r["name"] == "buy_and_hold"])
    )
    mean_strat = pooled_mean_return
    beat_bh = mean_strat > mean_bh
    # Distinguishable from random: strategy above 95th pct of random? or percentile rank
    random_p95 = float(np.percentile(random_returns_arr, 95)) if len(random_returns_arr) else 0.0
    distinguishable = pct_random_below >= 0.95 and observed_btc > float(np.median(random_returns_arr))
    weakly_better_than_random = pct_random_below >= 0.75

    n_trades_pooled = len(all_strategy_trades)
    sample_note = (
        f"Pooled trades={n_trades_pooled}; "
        f"inference reliable only if >= {MIN_TRADES_FOR_INFERENCE}."
    )
    notes.append(sample_note)
    notes.append(mc_pooled.note)

    answers = {
        "1_distinguishable_from_random": (
            "Yes (strong)"
            if distinguishable
            else ("Weakly (exploratory)" if weakly_better_than_random else "No")
        ),
        "2_outperform_buy_and_hold": "Yes" if beat_bh else "No",
        "3_outperform_random_matched_frequency": (
            f"BTC percentile vs {n_random_seeds} random seeds: "
            f"{pct_random_below*100:.1f}% of random runs worse; "
            f"random median={float(np.median(random_returns_arr)):.2f}%, "
            f"strategy={observed_btc:.2f}%."
        ),
        "4_fee_slippage_sensitivity": (
            f"BTC break-even slippage @ fee=0.1%: {slip.get('break_even_slippage')}; "
            f"multi-asset mean returns by cost: "
            + ", ".join(f"{c['scenario']}={c['mean_return_pct']:.2f}%" for c in cost_multi)
        ),
        "5_ci_reliability": (
            "Insufficient for strong inference"
            if n_trades_pooled < MIN_TRADES_FOR_INFERENCE
            else "Exploratory / moderate"
            if n_trades_pooled < 50
            else "More adequate but still limited"
        ),
        "6_cross_asset_stability": (
            f"Returns: "
            + ", ".join(f"{r['symbol']}={r['return_pct']:.2f}%" for r in per_asset)
            + f"; mean={mean_strat:.2f}%."
        ),
        "7_substantial_drawdown_prob": (
            f"Pooled MC P(DD>5%)={mc_pooled.prob_dd_gt_5:.2%}, "
            f"P(DD>10%)={mc_pooled.prob_dd_gt_10:.2%}, "
            f"P(DD>20%)={mc_pooled.prob_dd_gt_20:.2%}, "
            f"worst={mc_pooled.worst_max_dd:.2f}% (model-based)."
        ),
        "8_justify_paper_trading": "No",
    }

    # Final categorical verdict — do not force positive
    if distinguishable and beat_bh and n_trades_pooled >= MIN_TRADES_FOR_INFERENCE:
        verdict = "Weak evidence; more research required"
    elif n_trades_pooled < MIN_TRADES_FOR_INFERENCE:
        verdict = "Insufficient sample size"
    elif not weakly_better_than_random and not beat_bh:
        verdict = "No evidence of an edge"
    else:
        verdict = "Insufficient sample size" if n_trades_pooled < 30 else "Weak evidence; more research required"

    # Given Phase 9/10 context and likely mixed results, prefer honest wording
    if not distinguishable:
        if n_trades_pooled < MIN_TRADES_FOR_INFERENCE:
            verdict = "Insufficient sample size"
        else:
            verdict = "No evidence of an edge"

    payload = {
        "verdict": verdict,
        "answers": answers,
        "notes": notes,
        "eval_window": {"start": eval_start, "end": eval_end, "timeframe": timeframe},
        "per_asset": per_asset,
        "pooled_mean_return_pct": pooled_mean_return,
        "monte_carlo_pooled": mc_pooled.to_dict(),
        "monte_carlo_btc": mc_btc.to_dict(),
        "bootstrap_pooled": [b.to_dict() for b in boot_pooled],
        "bootstrap_btc": [b.to_dict() for b in boot_btc],
        "random_null": {
            "n_seeds": n_random_seeds,
            "observed_btc_return": observed_btc,
            "random_median": float(np.median(random_returns_arr)),
            "random_p05": float(np.percentile(random_returns_arr, 5)),
            "random_p95": random_p95,
            "fraction_random_below_strategy": pct_random_below,
            "fraction_random_le_strategy": pct_random_worse_or_eq,
        },
        "cost_sensitivity_btc": [c.to_dict() for c in cost_rows],
        "cost_sensitivity_multi": cost_multi,
        "slippage_break_even_btc": slip,
        "buy_and_hold_mean_return": mean_bh,
        "plots": plots_written,
        "n_trades_pooled": n_trades_pooled,
    }
    (export / "phase11_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    return Phase11Outcome(verdict=verdict, answers=answers, notes=notes, payload=payload)
