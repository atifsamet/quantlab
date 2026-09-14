"""
Phase 8 research CLI.

Usage:
  python -m app.research --data data/historical/BTC-USDT_1h.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.backtest.engine import load_ohlcv_csv
from app.backtest.robust import format_period_block, run_period
from app.indicators import add_indicators
from app.research.diagnostics import diagnose_baseline_filters, format_diagnostic_report
from app.research.search import (
    RobustnessConfig,
    build_small_search_grid,
    run_research_pipeline,
)
from app.strategy.baseline import REQUIRED_INDICATORS, generate_signals
from app.utils.logger import setup_logging


def _print_candidate_brief(title: str, candidate) -> None:
    print(f"\n=== {title} ===")
    print(f"Label     : {candidate.params.label}")
    print(f"Rejected  : {candidate.rejected} {candidate.reject_reason}")
    print(f"Score     : {candidate.score:.4f}")
    print(
        f"Train     : return={candidate.train.metrics.total_return_pct:.4f}% "
        f"trades={candidate.train.metrics.total_trades} "
        f"PF={candidate.train.metrics.profit_factor} "
        f"DD={candidate.train.metrics.max_drawdown_pct:.4f}%"
    )
    print(
        f"Validation: return={candidate.validation.metrics.total_return_pct:.4f}% "
        f"trades={candidate.validation.metrics.total_trades} "
        f"PF={candidate.validation.metrics.profit_factor} "
        f"DD={candidate.validation.metrics.max_drawdown_pct:.4f}%"
    )
    if candidate.final_test is not None:
        ft = candidate.final_test.metrics
        print(
            f"FINAL TEST: return={ft.total_return_pct:.4f}% "
            f"trades={ft.total_trades} PF={ft.profit_factor} "
            f"DD={ft.max_drawdown_pct:.4f}% "
            f"LONG={ft.long.trade_count}/{ft.long.total_pnl:.2f} "
            f"SHORT={ft.short.trade_count}/{ft.short.total_pnl:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 strategy research (no live trading, no private APIs)."
    )
    parser.add_argument("--data", required=True, help="Historical OHLCV CSV")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h"])
    parser.add_argument("--train-start", default="2024-09-01")
    parser.add_argument("--train-end", default="2025-03-01")
    parser.add_argument("--val-start", default="2025-03-01")
    parser.add_argument("--val-end", default="2025-09-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--test-end", default="2026-09-01")
    parser.add_argument("--no-risk", action="store_true")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--top-k-final", type=int, default=3)
    parser.add_argument("--top-k-walkforward", type=int, default=2)
    parser.add_argument("--export-dir", default="data/results/research")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    # --- 8A diagnostics on original baseline over train+val+test span ---
    ohlcv = load_ohlcv_csv(args.data)
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    baseline_hist = generate_signals(add_indicators(ohlcv))
    diag = diagnose_baseline_filters(
        baseline_hist,
        start=args.train_start,
        end=args.test_end,
    )
    print(format_diagnostic_report(diag))

    # Baseline Phase-5 reference on the same FINAL TEST (for comparison only)
    baseline_test = run_period(
        baseline_hist,
        label="baseline_final_test_ref",
        start=args.test_start,
        end=args.test_end,
        instrument=args.symbol,
        timeframe=args.timeframe,
        use_risk=not args.no_risk,
        required_columns=REQUIRED_INDICATORS,
    )
    print("\n=== Phase 3/5 baseline FINAL TEST reference (unchanged strategy) ===")
    print(format_period_block(baseline_test))

    robustness = RobustnessConfig(min_trades=args.min_trades)
    print(f"\nSearch grid size: {len(build_small_search_grid())}")
    print("Selecting candidates on TRAIN+VALIDATION only; FINAL TEST untouched until selection.\n")

    outcome = run_research_pipeline(
        args.data,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        instrument=args.symbol,
        timeframe=args.timeframe,
        use_risk=not args.no_risk,
        top_k_final=args.top_k_final,
        top_k_walkforward=args.top_k_walkforward,
        robustness=robustness,
    )

    ranked = outcome["evaluated"]
    accepted = outcome["accepted"]
    winner = outcome["validation_winner"]

    print(f"\nEvaluated candidates: {outcome['grid_size']}")
    print(f"Accepted after robustness filters: {len(accepted)}")
    print("\nTop candidates by validation score:")
    for i, cand in enumerate(ranked[:8], start=1):
        status = "REJECT" if cand.rejected else "OK"
        print(
            f"  {i}. [{status}] score={cand.score:.2f} "
            f"val_ret={cand.validation.metrics.total_return_pct:.2f}% "
            f"val_trades={cand.validation.metrics.total_trades} "
            f"{cand.params.label}"
            + (f" ({cand.reject_reason})" if cand.rejected else "")
        )

    if winner is None:
        print("\nNo candidate passed robustness filters. Baseline remains the reference.")
    else:
        _print_candidate_brief("Validation winner", winner)
        print("\nWalk-forward unseen-window results (winner):")
        if not winner.walk_forward_tests:
            print("  (no walk-forward windows)")
        for period in winner.walk_forward_tests:
            print(
                f"  {period.label}: return={period.metrics.total_return_pct:.4f}% "
                f"trades={period.metrics.total_trades} "
                f"DD={period.metrics.max_drawdown_pct:.4f}%"
            )

        if winner.final_test is not None:
            print("\n=== FINAL untouched test (selected once after validation) ===")
            print(format_period_block(winner.final_test))
            improves = (
                winner.final_test.metrics.total_return_pct
                > baseline_test.metrics.total_return_pct
                and winner.final_test.metrics.max_drawdown_pct
                <= baseline_test.metrics.max_drawdown_pct + 5.0
            )
            print(
                "\nImproves vs baseline on final test (return and not much worse DD)? "
                f"{improves}"
            )
            print(
                "NOTE: Positive or better backtests still do NOT prove future profitability."
            )

    # Export summaries
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    rows = [c.to_summary_dict() for c in ranked]
    pd.DataFrame(rows).to_csv(export_dir / "candidates_summary.csv", index=False)
    (export_dir / "diagnostics.json").write_text(
        json.dumps(diag.to_dict(), indent=2), encoding="utf-8"
    )
    payload = {
        "diagnostics": diag.to_dict(),
        "baseline_final_test": baseline_test.report.to_dict(),
        "grid_size": outcome["grid_size"],
        "accepted": len(accepted),
        "validation_winner": None if winner is None else winner.to_summary_dict(),
    }
    (export_dir / "research_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote research artifacts under {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
