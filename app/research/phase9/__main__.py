"""
Phase 9 CLI.

Examples:
  python -m app.research.phase9 --download
  python -m app.research.phase9 --data-dir data/historical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.data.okx_historical import OkxHistoricalDownloader, normalize_timeframe
from app.research.multi import run_phase9_pipeline
from app.research.universe import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, csv_path_for
from app.utils.logger import setup_logging


def _download_universe(
    *,
    symbols: list[str],
    timeframes: list[str],
    start: str,
    end: str,
    data_dir: Path,
) -> list[Path]:
    settings = get_settings()
    settings.assert_trading_disabled()
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with OkxHistoricalDownloader() as downloader:
        for symbol in symbols:
            for tf in timeframes:
                normalize_timeframe(tf)
                if tf == "1m":
                    print(f"Skipping {symbol} {tf}: Phase 9 avoids excessive 1m downloads")
                    continue
                out = csv_path_for(data_dir, symbol, tf)
                print(f"Downloading {symbol} {tf} -> {out}")
                result = downloader.download(
                    symbol=symbol,
                    timeframe=tf,
                    start=start,
                    end=end,
                    output_path=out,
                    save=True,
                )
                written.append(Path(result.path))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 9 multi-asset/timeframe research (no live trading)."
    )
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated OKX instruments",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(DEFAULT_TIMEFRAMES),
        help="Comma-separated timeframes (15m,1h,4h)",
    )
    parser.add_argument("--train-start", default="2024-09-01")
    parser.add_argument("--train-end", default="2025-03-01")
    parser.add_argument("--val-start", default="2025-03-01")
    parser.add_argument("--val-end", default="2025-09-01")
    parser.add_argument("--test-start", default="2025-09-01")
    parser.add_argument("--test-end", default="2026-09-01")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--download-start",
        default="2024-08-20",
        help="Inclusive download start (includes warm-up margin)",
    )
    parser.add_argument("--download-end", default="2026-09-01")
    parser.add_argument("--export-dir", default="data/results/phase9")
    parser.add_argument("--portfolio-timeframe", default="1h")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    timeframes = tuple(t.strip().lower() for t in args.timeframes.split(",") if t.strip())

    if args.download:
        paths = _download_universe(
            symbols=list(symbols),
            timeframes=list(timeframes),
            start=args.download_start,
            end=args.download_end,
            data_dir=Path(args.data_dir),
        )
        print(f"Downloaded {len(paths)} files")

    print("Selecting on TRAIN+VALIDATION only; FINAL TEST untouched until selection.")
    outcome = run_phase9_pipeline(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframes=timeframes,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        portfolio_timeframe=args.portfolio_timeframe,
    )

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    cell_rows = [c.to_summary_dict() for c in outcome["cells"]]
    pd.DataFrame(cell_rows).to_csv(export_dir / "cells_summary.csv", index=False)
    pd.DataFrame([a.to_dict() for a in outcome["aggregates"]]).to_csv(
        export_dir / "family_aggregates.csv", index=False
    )

    print(f"\nAvailable datasets: {len(outcome['available'])}")
    for sym, tf, path in outcome["available"]:
        print(f"  {sym} {tf}: {path}")

    print("\n=== Family aggregates (validation, cross-asset) ===")
    for i, fam in enumerate(outcome["aggregates"], start=1):
        status = "REJECT" if fam.rejected else "OK"
        print(
            f"  {i}. [{status}] {fam.variant}@{fam.timeframe} "
            f"score={fam.cross_asset_score:.2f} "
            f"mean_val={fam.mean_val_return:.2f}% "
            f"pos_assets={fam.assets_positive_val}/{fam.assets_total} "
            f"mean_trades={fam.mean_val_trades:.1f}"
            + (f" ({fam.reject_reason})" if fam.reject_reason else "")
        )

    best = outcome["best_family"]
    if best is None:
        print("\n=== CONCLUSION ===")
        print("No robust strategy found on train/validation across assets/timeframes.")
        print("Do not move to paper trading based on this evidence.")
    else:
        print("\n=== Best validation family (not claimed profitable) ===")
        print(
            f"{best.variant}@{best.timeframe} "
            f"mean_val={best.mean_val_return:.2f}% "
            f"final_mean={best.final_test_mean_return}"
        )
        print("\nPer-asset validation / final:")
        for cell in best.cells:
            final = (
                "n/a"
                if cell.final_test is None
                else (
                    f"{cell.final_test.metrics.total_return_pct:.2f}% "
                    f"trades={cell.final_test.metrics.total_trades} "
                    f"DD={cell.final_test.metrics.max_drawdown_pct:.2f}%"
                )
            )
            print(
                f"  {cell.symbol}: val={cell.validation.metrics.total_return_pct:.2f}% "
                f"tr={cell.validation.metrics.total_trades} "
                f"sig L/S/H="
                f"{cell.signal_freq.long_pct:.1f}/"
                f"{cell.signal_freq.short_pct:.1f}/"
                f"{cell.signal_freq.hold_pct:.1f} "
                f"final={final}"
            )
        if best.cells and best.cells[0].walk_forward_tests:
            print("\nWalk-forward (BTC):")
            btc = next((c for c in best.cells if c.symbol == "BTC-USDT"), None)
            if btc:
                for period in btc.walk_forward_tests:
                    print(
                        f"  {period.label}: "
                        f"{period.metrics.total_return_pct:.2f}% "
                        f"trades={period.metrics.total_trades} "
                        f"DD={period.metrics.max_drawdown_pct:.2f}%"
                    )

        # Durable edge check — require broad positive final AND non-collapsed WF
        finals = [
            c.final_test.metrics.total_return_pct
            for c in best.cells
            if c.final_test is not None
        ]
        pos_final = sum(1 for r in finals if r > 0)
        mean_final = sum(finals) / len(finals) if finals else 0.0
        btc = next((c for c in best.cells if c.symbol == "BTC-USDT"), None)
        wf_rets = (
            [p.metrics.total_return_pct for p in btc.walk_forward_tests]
            if btc and btc.walk_forward_tests
            else []
        )
        wf_pos = sum(1 for r in wf_rets if r > 0)
        print("\n=== CONCLUSION ===")
        durable = (
            len(finals) >= 3
            and pos_final >= max(3, (len(finals) + 1) // 2)
            and mean_final > 0
            and (not wf_rets or wf_pos >= max(2, (len(wf_rets) + 1) // 2))
        )
        if durable:
            print(
                "Some positive OOS signals observed, but this still does NOT prove "
                "a durable edge or future profitability."
            )
            print("Evidence is insufficient to confidently move to paper trading.")
        else:
            print("No robust strategy found.")
            print(
                "Out-of-sample results are mixed/unstable (and/or walk-forward is inconsistent). "
                "Evidence does NOT support moving to paper trading yet."
            )
        print(
            f"(final mean={mean_final:.2f}%, positive assets={pos_final}/{len(finals)}, "
            f"WF positive windows={wf_pos}/{len(wf_rets)})"
        )

    if outcome["portfolio"]:
        p = outcome["portfolio"]
        print("\n=== Portfolio (final test window, post-selection) ===")
        print(
            f"variant={p['variant']} tf={p['timeframe']} "
            f"return={p['return_pct']:.2f}% trades={p['trades']} "
            f"DD={p['max_dd']:.2f}% peak_pos={p['peak_positions']} "
            f"traded={p['symbols_traded']}"
        )
        print(f"notes: {p['notes']}")

    summary = {
        "available": outcome["available"],
        "candidates": outcome["candidates"],
        "aggregates": [a.to_dict() for a in outcome["aggregates"]],
        "selected": [a.to_dict() for a in outcome["selected"]],
        "best_family": None if best is None else best.to_dict(),
        "portfolio": outcome["portfolio"],
        "note": outcome["note"],
    }
    (export_dir / "phase9_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote artifacts under {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
