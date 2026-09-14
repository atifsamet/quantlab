"""Phase 9 unit tests (offline, no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.backtest.engine import BacktestConfig, BacktestEngine
from app.data.okx_historical import normalize_timeframe
from app.research.costs import CostScenario, has_funding_column
from app.research.multi import (
    aggregate_family,
    build_phase9_candidates,
    evaluate_cell,
    measure_signal_frequency,
    prepare_history,
)
from app.research.params import ResearchParams, validate_research_params
from app.research.portfolio import PortfolioConfig, run_portfolio_backtest
from app.research.search import RobustnessConfig
from app.research.universe import csv_path_for
from app.research.variants import build_strategy


def _ohlcv(n: int = 400, start: str = "2024-08-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    close = pd.Series(range(n), dtype=float) + 100.0
    # mild oscillation so RSI/MACD vary
    close = close + (pd.Series(range(n)) % 17 - 8) * 0.5
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        }
    )


def test_normalize_4h_timeframe() -> None:
    assert normalize_timeframe("4h") == "4h"
    assert normalize_timeframe("4H") == "4h"


def test_phase9_families_build() -> None:
    for params in build_phase9_candidates():
        validate_research_params(params)
        strat = build_strategy(params)
        assert strat is not None


def test_mean_reversion_and_momentum_signals_deterministic() -> None:
    ohlcv = _ohlcv()
    for variant in ("momentum", "mean_reversion", "trend_following", "breakout"):
        params = ResearchParams(variant=variant)
        a = prepare_history(ohlcv, params)
        b = prepare_history(ohlcv, params)
        assert list(a["signal"]) == list(b["signal"])
        freq = measure_signal_frequency(a)
        assert freq.rows == len(a)
        assert abs(freq.long_pct + freq.short_pct + freq.hold_pct - 100.0) < 1e-6


def test_no_lookahead_breakout_prior_high() -> None:
    ohlcv = _ohlcv(250)
    params = ResearchParams(variant="breakout", breakout_lookback=10)
    hist = prepare_history(ohlcv, params)
    # prior_high at i must equal max(high[i-10:i]) not including i
    for i in range(20, 40):
        expected = float(ohlcv["high"].iloc[i - 10 : i].max())
        assert hist["prior_high"].iloc[i] == pytest.approx(expected)


def test_param_validation_rejects_bad_mean_reversion() -> None:
    with pytest.raises(ValueError):
        validate_research_params(
            ResearchParams(variant="mean_reversion", rsi_oversold=80, rsi_overbought=20)
        )


def test_train_val_test_separation_in_cell(tmp_path: Path) -> None:
    ohlcv = _ohlcv(9000, start="2024-08-01")
    # Ensure enough span for train/val windows
    params = ResearchParams(variant="momentum")
    cell = evaluate_cell(
        ohlcv,
        params,
        symbol="BTC-USDT",
        timeframe="1h",
        train_start="2024-09-01",
        train_end="2025-03-01",
        val_start="2025-03-01",
        val_end="2025-09-01",
        robustness=RobustnessConfig(min_trades=1, max_drawdown_pct=100.0),
    )
    assert cell.final_test is None  # selection stage must not touch final test
    assert cell.train.end <= cell.validation.start or cell.train.end == cell.validation.start


def test_aggregate_requires_multi_asset_for_robustness() -> None:
    ohlcv = _ohlcv(5000, start="2024-08-01")
    params = ResearchParams(variant="baseline")
    cells = []
    for sym in ("BTC-USDT", "ETH-USDT"):
        cells.append(
            evaluate_cell(
                ohlcv,
                params,
                symbol=sym,
                timeframe="1h",
                train_start="2024-09-01",
                train_end="2025-01-01",
                val_start="2025-01-01",
                val_end="2025-05-01",
                robustness=RobustnessConfig(min_trades=1000),  # force reject
            )
        )
    fam = aggregate_family(cells, min_assets=2)
    assert fam.rejected is True


def test_fee_slippage_config_changes_pnl() -> None:
    ohlcv = _ohlcv(300)
    params = ResearchParams(variant="momentum")
    hist = prepare_history(ohlcv, params)
    base = BacktestEngine(
        config=BacktestConfig(fee_rate=0.0, slippage_rate=0.0)
    ).run(hist, compute_indicators=False, compute_signals=False)
    costly = BacktestEngine(
        config=BacktestConfig(fee_rate=0.01, slippage_rate=0.01)
    ).run(hist, compute_indicators=False, compute_signals=False)
    # With higher costs, final equity should not improve
    assert costly.final_equity <= base.final_equity + 1e-9


def test_funding_not_invented() -> None:
    ohlcv = _ohlcv(300)
    params = ResearchParams(variant="momentum")
    hist = prepare_history(ohlcv, params)
    assert has_funding_column(hist) is False
    # apply_funding True without column must behave like fee/slip only
    result = BacktestEngine(
        config=BacktestConfig(apply_funding=True, fee_rate=0.0, slippage_rate=0.0)
    ).run(hist, compute_indicators=False, compute_signals=False)
    assert all(t.funding == 0.0 for t in result.trades)


def test_funding_applied_when_column_present() -> None:
    ohlcv = _ohlcv(50)
    # Force a long then hold with funding
    frame = ohlcv.copy()
    for col in (
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
    ):
        frame[col] = 1.0
    frame["signal"] = "HOLD"
    frame.loc[0, "signal"] = "LONG"
    frame["funding_rate"] = 0.001  # 0.1% per bar
    result = BacktestEngine(
        config=BacktestConfig(apply_funding=True, fee_rate=0.0, slippage_rate=0.0)
    ).run(frame, compute_indicators=False, compute_signals=False)
    assert result.total_trades >= 1
    assert any(t.funding > 0 for t in result.trades)


def test_portfolio_max_positions() -> None:
    frames = {}
    for i, sym in enumerate(("BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT")):
        df = _ohlcv(80, start="2024-09-01")
        df["atr_14"] = 2.0
        df["signal"] = "HOLD"
        # Stagger entries so multiple want to open
        df.loc[5 + i, "signal"] = "LONG"
        frames[sym] = df
    result = run_portfolio_backtest(
        frames,
        config=PortfolioConfig(max_open_positions=2, risk_per_trade=0.01),
    )
    assert result.open_position_peak <= 2


def test_csv_path_helper() -> None:
    assert csv_path_for("data/historical", "btc-usdt", "1h").name == "BTC-USDT_1h.csv"


def test_cost_scenario_builds_config() -> None:
    sc = CostScenario("x", fee_rate=0.002, slippage_rate=0.001)
    cfg = sc.to_backtest_config()
    assert cfg.fee_rate == 0.002
    assert cfg.leverage == 1.0
