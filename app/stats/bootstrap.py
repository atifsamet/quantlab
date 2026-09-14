"""Bootstrap confidence intervals for trade statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from app.backtest.models import Trade
from app.stats import MIN_TRADES_FOR_INFERENCE


@dataclass(slots=True)
class BootstrapStat:
    metric: str
    point_estimate: float | None
    p05: float | None
    p95: float | None
    n_trades: int
    n_boot: int
    reliable: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profit_factor(pnls: np.ndarray) -> float | None:
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else None
    return float(wins / losses)


def bootstrap_trade_metrics(
    trades: Sequence[Trade],
    *,
    initial_capital: float = 1000.0,
    n_boot: int = 5000,
    seed: int = 42,
    min_trades: int = MIN_TRADES_FOR_INFERENCE,
) -> list[BootstrapStat]:
    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    n = len(pnls)
    if n == 0:
        return [
            BootstrapStat(
                metric=m,
                point_estimate=None,
                p05=None,
                p95=None,
                n_trades=0,
                n_boot=0,
                reliable=False,
                note="No trades.",
            )
            for m in (
                "avg_trade_pnl",
                "expectancy",
                "total_return_pct",
                "win_rate",
                "profit_factor",
            )
        ]

    reliable = n >= min_trades
    note = (
        "OK"
        if reliable
        else f"Sample size n={n} < {min_trades}; intervals are exploratory only."
    )
    rng = np.random.default_rng(seed)

    def point(metric: str) -> float | None:
        if metric in ("avg_trade_pnl", "expectancy"):
            return float(pnls.mean())
        if metric == "total_return_pct":
            return float(pnls.sum() / initial_capital * 100.0)
        if metric == "win_rate":
            return float(np.mean(pnls > 0) * 100.0)
        if metric == "profit_factor":
            return _profit_factor(pnls)
        raise KeyError(metric)

    metrics = ("avg_trade_pnl", "expectancy", "total_return_pct", "win_rate", "profit_factor")
    out: list[BootstrapStat] = []
    boot: dict[str, np.ndarray] = {m: np.empty(n_boot, dtype=float) for m in metrics}

    for i in range(n_boot):
        sample = rng.choice(pnls, size=n, replace=True)
        boot["avg_trade_pnl"][i] = sample.mean()
        boot["expectancy"][i] = sample.mean()
        boot["total_return_pct"][i] = sample.sum() / initial_capital * 100.0
        boot["win_rate"][i] = np.mean(sample > 0) * 100.0
        pf = _profit_factor(sample)
        boot["profit_factor"][i] = pf if pf is not None and np.isfinite(pf) else np.nan

    for m in metrics:
        pe = point(m)
        arr = boot[m]
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            out.append(
                BootstrapStat(m, pe, None, None, n, n_boot, reliable, note)
            )
            continue
        out.append(
            BootstrapStat(
                metric=m,
                point_estimate=pe if pe != float("inf") else None,
                p05=float(np.percentile(finite, 5)),
                p95=float(np.percentile(finite, 95)),
                n_trades=n,
                n_boot=n_boot,
                reliable=reliable,
                note=note,
            )
        )
    return out
