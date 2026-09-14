"""Phase 8A diagnostics: which baseline filters suppress signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from app.data.okx_historical import filter_date_range, parse_utc_timestamp
from app.strategy.baseline import REQUIRED_INDICATORS, _as_finite_float


@dataclass(slots=True)
class DiagnosticReport:
    rows: int
    ready_rows: int
    long_signals: int
    short_signals: int
    hold_signals: int
    hold_pct: float
    long_pct: float
    short_pct: float
    ema_bullish_pct: float
    ema_bearish_pct: float
    rsi_long_band_pct: float
    rsi_short_band_pct: float
    macd_bullish_pct: float
    macd_bearish_pct: float
    long_all_conditions_pct: float
    short_all_conditions_pct: float
    long_block_rsi_pct: float
    long_block_macd_pct: float
    long_block_both_pct: float
    short_block_rsi_pct: float
    short_block_macd_pct: float
    short_block_both_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ready(row: pd.Series) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for key in REQUIRED_INDICATORS:
        parsed = _as_finite_float(row.get(key))
        if parsed is None:
            return None
        values[key] = parsed
    return values


def diagnose_baseline_filters(
    df: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> DiagnosticReport:
    """
    Measure how often each Phase-3 baseline condition is true and what blocks trades.

    Expects a frame with Phase-2 indicators (and optional signal columns).
    """
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if start is not None or end is not None:
        frame = filter_date_range(
            frame,
            start=parse_utc_timestamp(start) if start is not None else None,
            end=parse_utc_timestamp(end) if end is not None else None,
        )

    n = len(frame)
    ready = 0
    long_n = short_n = hold_n = 0
    ema_bull = ema_bear = 0
    rsi_long = rsi_short = 0
    macd_bull = macd_bear = 0
    long_all = short_all = 0
    # Blocking among EMA-aligned rows
    ema_bull_rows = 0
    long_block_rsi = long_block_macd = long_block_both = 0
    ema_bear_rows = 0
    short_block_rsi = short_block_macd = short_block_both = 0

    for _, row in frame.iterrows():
        ind = _ready(row)
        if ind is None:
            hold_n += 1
            continue
        ready += 1
        bull = ind["ema_20"] > ind["ema_50"] > ind["ema_200"]
        bear = ind["ema_20"] < ind["ema_50"] < ind["ema_200"]
        rsi_l = 50.0 <= ind["rsi_14"] <= 70.0
        rsi_s = 30.0 <= ind["rsi_14"] <= 50.0
        macd_l = ind["macd"] > ind["macd_signal"] and ind["macd_hist"] > 0.0
        macd_s = ind["macd"] < ind["macd_signal"] and ind["macd_hist"] < 0.0

        ema_bull += int(bull)
        ema_bear += int(bear)
        rsi_long += int(rsi_l)
        rsi_short += int(rsi_s)
        macd_bull += int(macd_l)
        macd_bear += int(macd_s)

        is_long = bull and rsi_l and macd_l
        is_short = bear and rsi_s and macd_s
        long_all += int(is_long)
        short_all += int(is_short)

        if is_long:
            long_n += 1
        elif is_short:
            short_n += 1
        else:
            hold_n += 1

        if bull and not is_long:
            ema_bull_rows += 1
            if (not rsi_l) and (not macd_l):
                long_block_both += 1
            elif not rsi_l:
                long_block_rsi += 1
            elif not macd_l:
                long_block_macd += 1
        if bear and not is_short:
            ema_bear_rows += 1
            if (not rsi_s) and (not macd_s):
                short_block_both += 1
            elif not rsi_s:
                short_block_rsi += 1
            elif not macd_s:
                short_block_macd += 1

    def pct(num: int, den: int) -> float:
        return (num / den * 100.0) if den else 0.0

    return DiagnosticReport(
        rows=n,
        ready_rows=ready,
        long_signals=long_n,
        short_signals=short_n,
        hold_signals=hold_n,
        hold_pct=pct(hold_n, n),
        long_pct=pct(long_n, n),
        short_pct=pct(short_n, n),
        ema_bullish_pct=pct(ema_bull, ready),
        ema_bearish_pct=pct(ema_bear, ready),
        rsi_long_band_pct=pct(rsi_long, ready),
        rsi_short_band_pct=pct(rsi_short, ready),
        macd_bullish_pct=pct(macd_bull, ready),
        macd_bearish_pct=pct(macd_bear, ready),
        long_all_conditions_pct=pct(long_all, ready),
        short_all_conditions_pct=pct(short_all, ready),
        long_block_rsi_pct=pct(long_block_rsi, ema_bull_rows),
        long_block_macd_pct=pct(long_block_macd, ema_bull_rows),
        long_block_both_pct=pct(long_block_both, ema_bull_rows),
        short_block_rsi_pct=pct(short_block_rsi, ema_bear_rows),
        short_block_macd_pct=pct(short_block_macd, ema_bear_rows),
        short_block_both_pct=pct(short_block_both, ema_bear_rows),
    )


def format_diagnostic_report(report: DiagnosticReport) -> str:
    lines = [
        "=== Phase 8A baseline signal diagnostics ===",
        f"Rows                    : {report.rows}",
        f"Indicator-ready rows    : {report.ready_rows}",
        f"LONG signals            : {report.long_signals} ({report.long_pct:.2f}%)",
        f"SHORT signals           : {report.short_signals} ({report.short_pct:.2f}%)",
        f"HOLD                    : {report.hold_signals} ({report.hold_pct:.2f}%)",
        "",
        "Condition frequency (among ready rows):",
        f"  EMA bullish stack     : {report.ema_bullish_pct:.2f}%",
        f"  EMA bearish stack     : {report.ema_bearish_pct:.2f}%",
        f"  RSI long band 50-70   : {report.rsi_long_band_pct:.2f}%",
        f"  RSI short band 30-50  : {report.rsi_short_band_pct:.2f}%",
        f"  MACD bullish          : {report.macd_bullish_pct:.2f}%",
        f"  MACD bearish          : {report.macd_bearish_pct:.2f}%",
        f"  ALL long conditions   : {report.long_all_conditions_pct:.2f}%",
        f"  ALL short conditions  : {report.short_all_conditions_pct:.2f}%",
        "",
        "Among EMA-bullish rows that did NOT fire LONG, blockers:",
        f"  RSI only              : {report.long_block_rsi_pct:.2f}%",
        f"  MACD only             : {report.long_block_macd_pct:.2f}%",
        f"  RSI + MACD            : {report.long_block_both_pct:.2f}%",
        "Among EMA-bearish rows that did NOT fire SHORT, blockers:",
        f"  RSI only              : {report.short_block_rsi_pct:.2f}%",
        f"  MACD only             : {report.short_block_macd_pct:.2f}%",
        f"  RSI + MACD            : {report.short_block_both_pct:.2f}%",
    ]
    return "\n".join(lines)
