"""
Historical backtest engine (Phase 4 + optional Phase 5 risk).

Execution convention (look-ahead safe)
--------------------------------------
A signal computed on candle N is executed at the **open of candle N+1**,
adjusted for slippage. The final candle cannot produce an executable order
because there is no N+1 bar.

Stop-loss / take-profit (risk mode only)
----------------------------------------
After entry, SL/TP are evaluated on each candle using that candle's high/low
(including the entry candle, after the open fill).

* LONG: SL if low <= stop; TP if high >= take-profit
* SHORT: SL if high >= stop; TP if low <= take-profit

If **both** SL and TP are touched in the same candle, assume **stop-loss
first** (conservative, deterministic). Fills use the SL/TP price (no extra
intra-bar path speculation).

Slippage (entry / signal exits)
-------------------------------
* Buying (long entry / short exit): fill = open * (1 + slippage_rate)
* Selling (long exit / short entry): fill = open * (1 - slippage_rate)

Fees
----
``fee_rate`` is charged on notional of each executed side (entry and exit).

Funding (optional)
------------------
When ``apply_funding=True`` **and** the evaluation frame includes a real
``funding_rate`` column, each open position accrues
``notional * funding_rate`` on that bar (sign: longs pay positive rates).
Rates are never synthesized — without the column, funding stays zero.
This module never places live orders or calls private exchange APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from app.backtest.metrics import compute_metrics
from app.backtest.models import (
    BacktestResult,
    EquityPoint,
    PositionSide,
    Trade,
)
from app.indicators.technical import REQUIRED_OHLCV, add_indicators
from app.market.candles import CANDLE_COLUMNS
from app.risk.manager import RiskManager
from app.strategy.baseline import (
    REQUIRED_INDICATORS,
    BaselineStrategy,
    Signal,
    StrategySignal,
)


class StrategyLike(Protocol):
    """Minimal strategy interface pluggable into the backtester."""

    def generate_signal(self, row: Any) -> StrategySignal: ...


class BacktestDataError(ValueError):
    """Raised when historical data fails validation."""


@dataclass(slots=True)
class BacktestConfig:
    """Configurable simulation assumptions (not hardcoded in the loop)."""

    initial_capital: float = 1000.0
    position_fraction: float = 1.0  # 100% of available capital (legacy Phase 4)
    leverage: float = 1.0  # 1x only in research stages
    fee_rate: float = 0.001  # 0.1% per side
    slippage_rate: float = 0.0005  # 0.05% per side
    # Funding is applied ONLY when a real ``funding_rate`` column is present
    # on the evaluation frame. Rates are never invented.
    apply_funding: bool = False
    funding_rate_column: str = "funding_rate"

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if not (0 < self.position_fraction <= 1.0):
            raise ValueError("position_fraction must be in (0, 1]")
        if self.leverage != 1.0:
            raise ValueError("Research stages support leverage=1.0 only")
        if self.fee_rate < 0 or self.slippage_rate < 0:
            raise ValueError("fee_rate and slippage_rate must be >= 0")


@dataclass(slots=True)
class _OpenPosition:
    side: PositionSide
    entry_timestamp: pd.Timestamp
    entry_price: float
    quantity: float
    entry_fee: float
    capital_at_entry: float
    stop_loss: float | None = None
    take_profit: float | None = None
    accrued_funding: float = 0.0


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """
    Load historical OHLCV candles from CSV.

    Required columns: timestamp, open, high, low, close, volume
    """
    frame = pd.read_csv(path)
    missing = [c for c in CANDLE_COLUMNS if c not in frame.columns]
    if missing:
        raise BacktestDataError(f"CSV missing required columns: {missing}")
    return frame.loc[:, list(CANDLE_COLUMNS)].copy()


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize OHLCV for backtesting."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise BacktestDataError("Input must be a pandas DataFrame")
    if df.empty:
        raise BacktestDataError("Candle DataFrame is empty")

    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise BacktestDataError(f"Missing required columns: {missing}")

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().any():
        raise BacktestDataError("One or more timestamps are invalid / missing")
    if frame["timestamp"].duplicated().any():
        raise BacktestDataError("Duplicate timestamps are not allowed")
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if frame["timestamp"].duplicated().any():
            raise BacktestDataError("Duplicate timestamps are not allowed")
        if not frame["timestamp"].is_monotonic_increasing:
            raise BacktestDataError("Timestamps must be in chronological order")

    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise BacktestDataError(f"Column '{col}' contains invalid values")

    return frame.reset_index(drop=True)


def _buy_price(open_price: float, slippage_rate: float) -> float:
    return open_price * (1.0 + slippage_rate)


def _sell_price(open_price: float, slippage_rate: float) -> float:
    return open_price * (1.0 - slippage_rate)


def _funding_payment(
    position: _OpenPosition,
    row: pd.Series,
    cfg: BacktestConfig,
) -> float:
    """
    Accrue funding for one bar when enabled and a real rate is present.

    Positive ``funding_rate`` means longs pay / shorts receive.
    Returns cash delta (negative = paid). Never invents rates.
    """
    if not cfg.apply_funding:
        return 0.0
    col = cfg.funding_rate_column
    if col not in row.index:
        return 0.0
    rate_raw = row[col]
    if pd.isna(rate_raw):
        return 0.0
    rate = float(rate_raw)
    notional = position.quantity * float(row["close"])
    if position.side is PositionSide.LONG:
        payment = -notional * rate
    else:
        payment = notional * rate
    position.accrued_funding += -payment  # store cost as positive when paid
    return payment


def resolve_sl_tp_hit(
    *,
    side: PositionSide,
    stop_loss: float,
    take_profit: float,
    high: float,
    low: float,
) -> str | None:
    """
    Return ``\"SL\"``, ``\"TP\"``, or ``None``.

    Same-candle ambiguity → stop-loss first (conservative).
    """
    if side is PositionSide.LONG:
        sl_hit = low <= stop_loss
        tp_hit = high >= take_profit
        if sl_hit and tp_hit:
            return "SL"
        if sl_hit:
            return "SL"
        if tp_hit:
            return "TP"
        return None

    sl_hit = high >= stop_loss
    tp_hit = low <= take_profit
    if sl_hit and tp_hit:
        return "SL"
    if sl_hit:
        return "SL"
    if tp_hit:
        return "TP"
    return None


class BacktestEngine:
    """
    Run OHLCV → indicators → strategy signals → simulated fills → metrics.

    Pass a ``RiskManager`` to enable Phase 5 sizing and SL/TP. Without it,
    Phase 4 full-capital behavior is preserved.
    """

    def __init__(
        self,
        strategy: StrategyLike | None = None,
        config: BacktestConfig | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.strategy: StrategyLike = strategy or BaselineStrategy()
        self.config = config or BacktestConfig()
        self.risk_manager = risk_manager

    def run(
        self,
        df: pd.DataFrame,
        *,
        compute_indicators: bool = True,
        compute_signals: bool = True,
    ) -> BacktestResult:
        """
        Simulate the strategy on historical candles.

        Parameters
        ----------
        df:
            OHLCV (and optionally precomputed indicators / signals).
        compute_indicators:
            When True, run Phase 2 ``add_indicators``.
        compute_signals:
            When True, generate signals via the configured strategy for each row.
            When False, require existing ``signal`` column values
            (``LONG`` / ``SHORT`` / ``HOLD``).
        """
        ohlcv = validate_ohlcv(df)

        if compute_indicators:
            working = add_indicators(ohlcv)
        else:
            working = ohlcv.copy()
            missing_ind = [c for c in REQUIRED_INDICATORS if c not in working.columns]
            if missing_ind:
                raise BacktestDataError(
                    f"compute_indicators=False but missing indicators: {missing_ind}"
                )

        if compute_signals:
            signals = [
                self.strategy.generate_signal(row).signal
                for _, row in working.iterrows()
            ]
            working["signal"] = [s.value for s in signals]
        else:
            if "signal" not in working.columns:
                raise BacktestDataError("compute_signals=False requires a signal column")
            working["signal"] = working["signal"].astype(str).str.upper()

        if self.risk_manager is not None:
            return self._simulate_with_risk(working)
        return self._simulate(working)

    def _simulate(self, df: pd.DataFrame) -> BacktestResult:
        """Legacy Phase 4 simulator (100% capital, signal exits / EOD)."""
        cfg = self.config
        cash = float(cfg.initial_capital)
        position: _OpenPosition | None = None
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        next_trade_id = 1

        n = len(df)
        for i in range(n):
            row = df.iloc[i]
            ts = pd.Timestamp(row["timestamp"])
            close_px = float(row["close"])

            if position is not None:
                cash += _funding_payment(position, row, cfg)

            equity_curve.append(
                EquityPoint(timestamp=ts, equity=_mark_equity(cash, position, close_px))
            )

            if i >= n - 1:
                continue

            signal = _parse_signal(row["signal"])
            if signal is Signal.HOLD:
                continue

            exec_row = df.iloc[i + 1]
            exec_ts = pd.Timestamp(exec_row["timestamp"])
            exec_open = float(exec_row["open"])
            if exec_open <= 0:
                continue

            desired = (
                PositionSide.LONG if signal is Signal.LONG else PositionSide.SHORT
            )

            if position is None:
                position, cash, opened = _open_position_full(
                    side=desired,
                    exec_ts=exec_ts,
                    exec_open=exec_open,
                    cash=cash,
                    cfg=cfg,
                )
                if not opened:
                    position = None
                continue

            if position.side is desired:
                continue

            trade, cash = _close_position(
                position=position,
                exec_ts=exec_ts,
                fill_price=_exit_fill_from_open(position.side, exec_open, cfg),
                cash=cash,
                cfg=cfg,
                trade_id=next_trade_id,
            )
            trades.append(trade)
            next_trade_id += 1
            position = None

            position, cash, opened = _open_position_full(
                side=desired,
                exec_ts=exec_ts,
                exec_open=exec_open,
                cash=cash,
                cfg=cfg,
            )
            if not opened:
                position = None

        if position is not None:
            last = df.iloc[-1]
            trade, cash = _close_position(
                position=position,
                exec_ts=pd.Timestamp(last["timestamp"]),
                fill_price=_exit_fill_from_open(
                    position.side, float(last["close"]), cfg
                ),
                cash=cash,
                cfg=cfg,
                trade_id=next_trade_id,
            )
            trades.append(trade)
            equity_curve[-1] = EquityPoint(
                timestamp=equity_curve[-1].timestamp,
                equity=max(0.0, cash),
            )

        return compute_metrics(
            initial_capital=cfg.initial_capital,
            final_equity=max(0.0, cash),
            trades=trades,
            equity_curve=equity_curve,
        )

    def _simulate_with_risk(self, df: pd.DataFrame) -> BacktestResult:
        """Phase 5 path: risk sizing + ATR stops / take-profits."""
        assert self.risk_manager is not None
        risk = self.risk_manager
        cfg = self.config
        cash = float(cfg.initial_capital)
        position: _OpenPosition | None = None
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        next_trade_id = 1

        n = len(df)
        for i in range(n):
            row = df.iloc[i]
            ts = pd.Timestamp(row["timestamp"])
            high = float(row["high"])
            low = float(row["low"])
            close_px = float(row["close"])

            if position is not None:
                cash += _funding_payment(position, row, cfg)

            # SL/TP on current bar (position opened at or before this open).
            if position is not None and position.stop_loss is not None and position.take_profit is not None:
                hit = resolve_sl_tp_hit(
                    side=position.side,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    high=high,
                    low=low,
                )
                if hit is not None:
                    fill = position.stop_loss if hit == "SL" else position.take_profit
                    trade, cash = _close_position(
                        position=position,
                        exec_ts=ts,
                        fill_price=float(fill),
                        cash=cash,
                        cfg=cfg,
                        trade_id=next_trade_id,
                    )
                    trades.append(trade)
                    next_trade_id += 1
                    position = None
                    risk.register_closed_trade(
                        trade.net_pnl, timestamp=ts, equity=max(0.0, cash)
                    )

            equity_now = _mark_equity(cash, position, close_px)
            equity_curve.append(EquityPoint(timestamp=ts, equity=equity_now))

            if i >= n - 1:
                continue

            signal = _parse_signal(row["signal"])
            if signal is Signal.HOLD:
                continue

            exec_row = df.iloc[i + 1]
            exec_ts = pd.Timestamp(exec_row["timestamp"])
            exec_open = float(exec_row["open"])
            if exec_open <= 0:
                continue

            desired = (
                PositionSide.LONG if signal is Signal.LONG else PositionSide.SHORT
            )

            # ATR from signal candle N only (no look-ahead into N+1 indicators).
            atr_raw = row["atr_14"] if "atr_14" in df.columns else None

            if position is not None and position.side is desired:
                continue

            if position is not None and position.side is not desired:
                # Opposite signal: close at next open, then maybe reverse.
                trade, cash = _close_position(
                    position=position,
                    exec_ts=exec_ts,
                    fill_price=_exit_fill_from_open(position.side, exec_open, cfg),
                    cash=cash,
                    cfg=cfg,
                    trade_id=next_trade_id,
                )
                trades.append(trade)
                next_trade_id += 1
                position = None
                risk.register_closed_trade(
                    trade.net_pnl, timestamp=exec_ts, equity=max(0.0, cash)
                )

            if position is not None:
                continue

            entry_fill = (
                _buy_price(exec_open, cfg.slippage_rate)
                if desired is PositionSide.LONG
                else _sell_price(exec_open, cfg.slippage_rate)
            )
            equity_for_risk = max(0.0, cash)
            decision = risk.evaluate(
                side=desired.value,
                entry_price=entry_fill,
                atr=atr_raw if atr_raw is not None else float("nan"),
                equity=equity_for_risk,
                timestamp=exec_ts,
            )
            if not decision.approved or decision.quantity is None:
                continue

            position, cash, opened = _open_position_risk(
                side=desired,
                exec_ts=exec_ts,
                entry_fill=entry_fill,
                quantity=float(decision.quantity),
                stop_loss=float(decision.stop_loss) if decision.stop_loss is not None else None,
                take_profit=float(decision.take_profit) if decision.take_profit is not None else None,
                cash=cash,
                cfg=cfg,
            )
            if not opened:
                position = None

        if position is not None:
            last = df.iloc[-1]
            trade, cash = _close_position(
                position=position,
                exec_ts=pd.Timestamp(last["timestamp"]),
                fill_price=_exit_fill_from_open(
                    position.side, float(last["close"]), cfg
                ),
                cash=cash,
                cfg=cfg,
                trade_id=next_trade_id,
            )
            trades.append(trade)
            risk.register_closed_trade(
                trade.net_pnl,
                timestamp=last["timestamp"],
                equity=max(0.0, cash),
            )
            equity_curve[-1] = EquityPoint(
                timestamp=equity_curve[-1].timestamp,
                equity=max(0.0, cash),
            )

        return compute_metrics(
            initial_capital=cfg.initial_capital,
            final_equity=max(0.0, cash),
            trades=trades,
            equity_curve=equity_curve,
        )


def _parse_signal(value: Any) -> Signal:
    text = str(value).upper()
    if text == Signal.LONG.value:
        return Signal.LONG
    if text == Signal.SHORT.value:
        return Signal.SHORT
    return Signal.HOLD


def _exit_fill_from_open(side: PositionSide, ref_price: float, cfg: BacktestConfig) -> float:
    if side is PositionSide.LONG:
        return _sell_price(ref_price, cfg.slippage_rate)
    return _buy_price(ref_price, cfg.slippage_rate)


def _mark_equity(cash: float, position: _OpenPosition | None, mark_price: float) -> float:
    if position is None:
        return max(0.0, cash)
    if position.side is PositionSide.LONG:
        unrealized = position.quantity * (mark_price - position.entry_price)
    else:
        unrealized = position.quantity * (position.entry_price - mark_price)
    return max(0.0, cash + position.capital_at_entry + unrealized)


def _open_position_full(
    *,
    side: PositionSide,
    exec_ts: pd.Timestamp,
    exec_open: float,
    cash: float,
    cfg: BacktestConfig,
) -> tuple[_OpenPosition | None, float, bool]:
    if cash <= 0:
        return None, max(0.0, cash), False

    notional = cash * cfg.position_fraction * cfg.leverage
    if notional <= 0:
        return None, cash, False

    if side is PositionSide.LONG:
        fill = _buy_price(exec_open, cfg.slippage_rate)
    else:
        fill = _sell_price(exec_open, cfg.slippage_rate)

    if fill <= 0:
        return None, cash, False

    quantity = notional / fill
    entry_fee = notional * cfg.fee_rate
    residual = cash - notional - entry_fee
    if residual < 0:
        affordable = cash / (1.0 + cfg.fee_rate)
        notional = affordable * cfg.position_fraction
        quantity = notional / fill
        entry_fee = notional * cfg.fee_rate
        residual = cash - notional - entry_fee
        if residual < -1e-9 or notional <= 0:
            return None, max(0.0, cash), False
        residual = max(0.0, residual)

    position = _OpenPosition(
        side=side,
        entry_timestamp=exec_ts,
        entry_price=fill,
        quantity=quantity,
        entry_fee=entry_fee,
        capital_at_entry=notional,
    )
    return position, residual, True


def _open_position_risk(
    *,
    side: PositionSide,
    exec_ts: pd.Timestamp,
    entry_fill: float,
    quantity: float,
    stop_loss: float | None,
    take_profit: float | None,
    cash: float,
    cfg: BacktestConfig,
) -> tuple[_OpenPosition | None, float, bool]:
    if cash <= 0 or quantity <= 0 or entry_fill <= 0:
        return None, max(0.0, cash), False

    notional = quantity * entry_fill
    entry_fee = notional * cfg.fee_rate
    if notional + entry_fee > cash + 1e-9:
        # Scale to available cash after fee.
        affordable_notional = cash / (1.0 + cfg.fee_rate)
        if affordable_notional <= 0:
            return None, max(0.0, cash), False
        quantity = affordable_notional / entry_fill
        notional = quantity * entry_fill
        entry_fee = notional * cfg.fee_rate

    residual = cash - notional - entry_fee
    if residual < -1e-9:
        return None, max(0.0, cash), False

    position = _OpenPosition(
        side=side,
        entry_timestamp=exec_ts,
        entry_price=entry_fill,
        quantity=quantity,
        entry_fee=entry_fee,
        capital_at_entry=notional,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    return position, max(0.0, residual), True


def _close_position(
    *,
    position: _OpenPosition,
    exec_ts: pd.Timestamp,
    fill_price: float,
    cash: float,
    cfg: BacktestConfig,
    trade_id: int,
) -> tuple[Trade, float]:
    fill = float(fill_price)
    if position.side is PositionSide.LONG:
        gross = position.quantity * (fill - position.entry_price)
    else:
        gross = position.quantity * (position.entry_price - fill)

    exit_notional = position.quantity * fill
    exit_fee = exit_notional * cfg.fee_rate
    fees = position.entry_fee + exit_fee
    funding = float(position.accrued_funding)
    # Funding cash impact is applied bar-by-bar while open; do not deduct again.
    net = gross - fees - funding
    proceeds = position.capital_at_entry + gross - exit_fee
    new_cash = max(0.0, cash + proceeds)

    return_pct = (
        (net / position.capital_at_entry * 100.0) if position.capital_at_entry else 0.0
    )

    trade = Trade(
        trade_id=trade_id,
        side=position.side,
        entry_timestamp=position.entry_timestamp,
        exit_timestamp=exec_ts,
        entry_price=position.entry_price,
        exit_price=fill,
        quantity=position.quantity,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        return_pct=return_pct,
        funding=funding,
    )
    return trade, new_cash
