"""
Phase 11 CLI — statistical validation / Monte Carlo (research only).

  python -m app.stats.phase11 --data-dir data/historical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.research.universe import DEFAULT_SYMBOLS
from app.stats.pipeline import run_phase11_pipeline
from app.utils.logger import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11 statistical validation (no trading, no optimization)."
    )
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default="4h", choices=["15m", "1h", "4h"])
    parser.add_argument("--eval-start", default="2025-09-01")
    parser.add_argument("--eval-end", default="2026-09-01")
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--n-random-seeds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-dir", default="data/results/phase11")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    settings = get_settings()
    settings.assert_trading_disabled()

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    print("Phase 11 — Statistical validation (research only)")
    print("No new strategies, no ML, no parameter tuning, no paper/live trading.\n")

    outcome = run_phase11_pipeline(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframe=args.timeframe,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        n_sims=args.n_sims,
        n_random_seeds=args.n_random_seeds,
        seed=args.seed,
        export_dir=args.export_dir,
    )

    print("=== Per-asset breakout@4h (eval window) ===")
    for row in outcome.payload.get("per_asset", []):
        print(
            f"  {row['symbol']}: ret={row['return_pct']:.2f}% "
            f"trades={row['trades']} WR={row['win_rate']:.1f}% "
            f"exp={row['expectancy']:.3f} DD={row['max_dd']:.2f}% "
            f"Sharpe={row['sharpe']}"
        )
    print(f"  Pooled mean return: {outcome.payload.get('pooled_mean_return_pct'):.2f}%")

    rnd = outcome.payload.get("random_null", {})
    print("\n=== vs random matched-frequency (BTC) ===")
    print(
        f"  strategy={rnd.get('observed_btc_return'):.2f}% "
        f"random_median={rnd.get('random_median'):.2f}% "
        f"P(random < strategy)={100*float(rnd.get('fraction_random_below_strategy', 0)):.1f}%"
    )

    mc = outcome.payload.get("monte_carlo_pooled", {})
    print("\n=== Monte Carlo (pooled trades) ===")
    print(
        f"  median={mc.get('median_return'):.2f}% "
        f"p05={mc.get('p05_return'):.2f}% p95={mc.get('p95_return'):.2f}% "
        f"P(loss)={100*float(mc.get('prob_losing', 0)):.1f}% "
        f"P(DD>10%)={100*float(mc.get('prob_dd_gt_10', 0)):.1f}%"
    )

    print("\n=== Cost sensitivity (multi-asset mean) ===")
    for row in outcome.payload.get("cost_sensitivity_multi", []):
        print(
            f"  {row['scenario']}: mean_ret={row['mean_return_pct']:.2f}% "
            f"(fee={row['fee_rate']}, slip={row['slippage_rate']})"
        )
    slip = outcome.payload.get("slippage_break_even_btc", {})
    print(f"  BTC break-even slippage @ 0.1% fee: {slip.get('break_even_slippage')}")

    print("\n=== Answers ===")
    for k, v in outcome.answers.items():
        print(f"  {k}: {v}")

    print("\n=== VERDICT ===")
    print(outcome.verdict)
    for note in outcome.notes:
        print(f"  - {note}")
    print("\nPaper/live trading NOT enabled. Phase 11 stops here.")
    print(f"Wrote artifacts under {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
