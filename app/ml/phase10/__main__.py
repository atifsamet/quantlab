"""
Phase 10 CLI — ML signal-filter research (no live trading).

  python -m app.ml.phase10 --data-dir data/historical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.ml import LABEL_DOC
from app.ml.features import FEATURE_COLUMNS
from app.ml.pipeline import run_phase10_pipeline
from app.research.universe import DEFAULT_SYMBOLS
from app.utils.logger import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10 ML signal-filter research (backtest only)."
    )
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default="4h", choices=["15m", "1h", "4h"])
    parser.add_argument("--also-1h", action="store_true")
    parser.add_argument("--train-start", default="2024-09-01")
    parser.add_argument("--train-end", default="2025-03-01")
    parser.add_argument("--val-start", default="2025-03-01")
    parser.add_argument("--val-end", default="2025-09-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--test-end", default="2026-09-01")
    parser.add_argument("--export-dir", default="data/results/phase10")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    settings = get_settings()
    settings.assert_trading_disabled()

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    print("Phase 10 — ML signal filter research (no execution, no paper trading)")
    print("Selecting model/threshold on TRAIN+VALIDATION only; FINAL TEST untouched.\n")
    print("--- Label definition ---")
    print(LABEL_DOC.strip())
    print(f"\nFeatures ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}\n")

    outcome = run_phase10_pipeline(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframe=args.timeframe,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        also_1h=args.also_1h,
    )

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    payload = outcome.to_dict()
    (export_dir / "phase10_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    if outcome.feature_importance:
        pd.DataFrame(outcome.feature_importance).to_csv(
            export_dir / "feature_importance.csv", index=False
        )
    if outcome.threshold_grid:
        pd.DataFrame(outcome.threshold_grid).to_csv(
            export_dir / "threshold_grid.csv", index=False
        )
    pd.DataFrame(outcome.classification).to_csv(
        export_dir / "classification_metrics.csv", index=False
    )
    pd.DataFrame(outcome.baseline_test + outcome.filtered_test).to_csv(
        export_dir / "final_test_comparison.csv", index=False
    )

    print("=== Dataset ===")
    for name, stats in (
        ("all", outcome.dataset_stats_all),
        ("train", outcome.dataset_stats_train),
        ("val", outcome.dataset_stats_val),
        ("test", outcome.dataset_stats_test),
    ):
        print(
            f"  {name}: n={stats.get('n_rows')} "
            f"pos={stats.get('positive_pct'):.1f}% "
            f"LONG={stats.get('n_long')} SHORT={stats.get('n_short')}"
        )

    print("\n=== Classification (secondary; backtest is primary) ===")
    for row in outcome.classification:
        if ":val" in row["model"]:
            print(
                f"  {row['model']}: prec={row['precision']:.3f} "
                f"rec={row['recall']:.3f} f1={row['f1']:.3f} "
                f"roc={row['roc_auc']} pr={row['pr_auc']}"
            )

    print("\n=== Selected filter ===")
    print(f"  model={outcome.selected_model} threshold={outcome.selected_threshold}")

    print("\n=== Validation: baseline vs ML-filtered (mean context in notes) ===")
    for row in outcome.baseline_val:
        print(
            f"  BASE {row['symbol']}: ret={row['return_pct']:.2f}% "
            f"trades={row['trades']} DD={row['max_dd']:.2f}%"
        )
    for row in outcome.filtered_val:
        print(
            f"  ML   {row['symbol']}: ret={row['return_pct']:.2f}% "
            f"trades={row['trades']} DD={row['max_dd']:.2f}%"
        )

    print("\n=== FINAL untouched test ===")
    for b, f in zip(outcome.baseline_test, outcome.filtered_test, strict=False):
        print(
            f"  {b['symbol']}: baseline={b['return_pct']:.2f}%/{b['trades']}t "
            f"-> ML={f['return_pct']:.2f}%/{f['trades']}t "
            f"DD {b['max_dd']:.2f}->{f['max_dd']:.2f}"
        )

    print("\n=== Walk-forward (BTC) ===")
    for fold in outcome.walk_forward:
        if fold.get("skipped"):
            print(f"  fold {fold['fold']}: skipped ({fold.get('reason')})")
            continue
        print(
            f"  fold {fold['fold']}: base={fold['baseline']['return_pct']:.2f}% "
            f"ml={fold['filtered']['return_pct']:.2f}% "
            f"trades {fold['baseline']['trades']}->{fold['filtered']['trades']}"
        )

    if outcome.feature_importance:
        print("\n=== Top feature importance ===")
        for row in outcome.feature_importance[:10]:
            print(f"  {row['feature']}: {row['importance']:.4f} ({row['kind']})")

    print("\n=== VERDICT ===")
    print(outcome.verdict)
    for note in outcome.notes:
        print(f"  - {note}")
    print("\nPaper/live trading NOT enabled. Phase 10 stops here.")
    print(f"Wrote artifacts under {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
