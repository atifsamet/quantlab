"""Phase 1–7 entrypoint: market demo, synthetic/CSV/robust backtests."""

from __future__ import annotations

import argparse

from app.backtest.__main__ import main as run_backtest_demo
from app.backtest.robust import (
    export_period_artifacts,
    format_comparison,
    format_period_block,
    load_signaled_history,
    run_period,
    run_walk_forward,
)
from app.config import get_settings
from app.exchange.okx_client import OkxClient
from app.indicators import WARMUP_BARS, add_indicators
from app.strategy import generate_signals
from app.utils.logger import setup_logging

CANDLE_LIMIT = 300
DISPLAY_ROWS = 5
DISPLAY_COLS = [
    "timestamp",
    "close",
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "signal",
    "signal_reason",
]


def run_market_demo() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.assert_trading_disabled()

    with OkxClient(settings=settings) as client:
        candles = client.get_candles("BTC-USDT", bar="1m", limit=CANDLE_LIMIT)

    framed = add_indicators(candles)
    signaled = generate_signals(framed)

    latest = signaled.iloc[-1]
    print(
        f"BTC-USDT 1m + indicators + baseline signal "
        f"(last {DISPLAY_ROWS} rows; warm-up={WARMUP_BARS} bars for EMA 200)"
    )
    print(f"Latest signal: {latest['signal']} - {latest['signal_reason']}")
    print(signaled.loc[:, DISPLAY_COLS].tail(DISPLAY_ROWS).to_string(index=False))


def run_csv_backtest(
    data_path: str,
    *,
    eval_start: str | None,
    eval_end: str | None,
    use_risk: bool,
    instrument: str,
    timeframe: str,
    export: bool,
) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.assert_trading_disabled()

    history = load_signaled_history(data_path)
    start = eval_start or str(history["timestamp"].iloc[0])
    end = eval_end or str(history["timestamp"].iloc[-1])
    period = run_period(
        history,
        label="full",
        start=start,
        end=end,
        instrument=instrument,
        timeframe=timeframe,
        use_risk=use_risk,
    )
    print(format_period_block(period))
    if export:
        paths = export_period_artifacts(period, prefix="full")
        for key, path in paths.items():
            print(f"{key:12}: {path}")


def run_train_test_backtest(
    data_path: str,
    *,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    use_risk: bool,
    instrument: str,
    timeframe: str,
    export: bool,
    compare_risk: bool,
) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.assert_trading_disabled()

    history = load_signaled_history(data_path)

    train = run_period(
        history,
        label="train",
        start=train_start,
        end=train_end,
        instrument=instrument,
        timeframe=timeframe,
        use_risk=use_risk,
    )
    test = run_period(
        history,
        label="test",
        start=test_start,
        end=test_end,
        instrument=instrument,
        timeframe=timeframe,
        use_risk=use_risk,
    )

    print(format_period_block(train))
    print()
    print(format_period_block(test))

    if compare_risk:
        print()
        p4 = run_period(
            history,
            label="test-phase4",
            start=test_start,
            end=test_end,
            instrument=instrument,
            timeframe=timeframe,
            use_risk=False,
        )
        p5 = run_period(
            history,
            label="test-phase5",
            start=test_start,
            end=test_end,
            instrument=instrument,
            timeframe=timeframe,
            use_risk=True,
        )
        print(format_comparison(p4, p5))
        if export:
            export_period_artifacts(p4, prefix="test_phase4")
            export_period_artifacts(p5, prefix="test_phase5")

    if export:
        for period in (train, test):
            paths = export_period_artifacts(period)
            print(f"Exported {period.label}:")
            for key, path in paths.items():
                print(f"  {key}: {path}")


def run_walk_forward_backtest(
    data_path: str,
    *,
    overall_start: str,
    overall_end: str,
    train_days: int,
    test_days: int,
    step_days: int | None,
    use_risk: bool,
    instrument: str,
    timeframe: str,
    export: bool,
) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    settings.assert_trading_disabled()

    history = load_signaled_history(data_path)
    folds = run_walk_forward(
        history,
        overall_start=overall_start,
        overall_end=overall_end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        instrument=instrument,
        timeframe=timeframe,
        use_risk=use_risk,
    )
    if not folds:
        print("No walk-forward folds fit in the requested overall window.")
        return

    print(f"Walk-forward folds: {len(folds)} (strategy parameters unchanged)")
    for fold in folds:
        print()
        print(format_period_block(fold.train))
        print()
        print(format_period_block(fold.test))
        if export:
            export_period_artifacts(fold.train, prefix=f"wf{fold.fold_index}_train")
            export_period_artifacts(fold.test, prefix=f"wf{fold.fold_index}_test")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto trading bot demos / robust backtests (no live trading)."
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest (synthetic unless --data is provided).",
    )
    parser.add_argument("--data", type=str, default=None, help="Historical OHLCV CSV")
    parser.add_argument("--eval-start", default=None, help="Single-period eval start UTC")
    parser.add_argument("--eval-end", default=None, help="Single-period eval end UTC")
    parser.add_argument("--train-start", default=None, help="Train window start UTC")
    parser.add_argument("--train-end", default=None, help="Train window end UTC")
    parser.add_argument("--test-start", default=None, help="Test window start UTC")
    parser.add_argument("--test-end", default=None, help="Test window end UTC")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run basic walk-forward evaluation (no parameter optimization).",
    )
    parser.add_argument("--wf-train-days", type=int, default=180)
    parser.add_argument("--wf-test-days", type=int, default=90)
    parser.add_argument("--wf-step-days", type=int, default=None)
    parser.add_argument("--wf-start", default=None, help="Walk-forward overall start")
    parser.add_argument("--wf-end", default=None, help="Walk-forward overall end")
    parser.add_argument(
        "--compare-risk",
        action="store_true",
        help="Compare Phase 4 (no risk) vs Phase 5 (risk) on the test window.",
    )
    parser.add_argument(
        "--no-risk",
        action="store_true",
        help="Disable Phase 5 risk manager (Phase 4 full-capital sizing).",
    )
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["1m", "5m", "15m", "1h"],
        help="Timeframe label / CSV convention (1h and 15m recommended).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Write equity/trades/summary/monthly CSVs under data/results/",
    )
    args = parser.parse_args()

    train_test_requested = any(
        [
            args.train_start,
            args.train_end,
            args.test_start,
            args.test_end,
        ]
    )

    if args.backtest and args.data and args.walk_forward:
        wf_start = args.wf_start or args.train_start or args.eval_start
        wf_end = args.wf_end or args.test_end or args.eval_end
        if not wf_start or not wf_end:
            parser.error("Walk-forward requires --wf-start/--wf-end (or train/test/eval dates)")
        run_walk_forward_backtest(
            args.data,
            overall_start=wf_start,
            overall_end=wf_end,
            train_days=args.wf_train_days,
            test_days=args.wf_test_days,
            step_days=args.wf_step_days,
            use_risk=not args.no_risk,
            instrument=args.symbol,
            timeframe=args.timeframe,
            export=args.export,
        )
        return

    if args.backtest and args.data and train_test_requested:
        missing = [
            name
            for name, val in [
                ("--train-start", args.train_start),
                ("--train-end", args.train_end),
                ("--test-start", args.test_start),
                ("--test-end", args.test_end),
            ]
            if not val
        ]
        if missing:
            parser.error(f"Train/test mode requires all of: {', '.join(missing)}")
        run_train_test_backtest(
            args.data,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            use_risk=not args.no_risk,
            instrument=args.symbol,
            timeframe=args.timeframe,
            export=args.export,
            compare_risk=args.compare_risk,
        )
        return

    if args.backtest and args.data:
        run_csv_backtest(
            args.data,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            use_risk=not args.no_risk,
            instrument=args.symbol,
            timeframe=args.timeframe,
            export=args.export,
        )
        return

    if args.backtest:
        run_backtest_demo()
        return

    if args.data:
        parser.error("--data requires --backtest")

    run_market_demo()


if __name__ == "__main__":
    main()
