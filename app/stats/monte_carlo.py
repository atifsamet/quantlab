"""Monte Carlo trade resampling (model-based uncertainty, not future proof)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from app.backtest.models import Trade
from app.stats.risk_metrics import max_drawdown_from_equity


@dataclass(slots=True)
class MonteCarloSummary:
    n_sims: int
    n_trades: int
    observed_return_pct: float
    observed_max_dd_pct: float
    median_return: float
    p05_return: float
    p25_return: float
    p75_return: float
    p95_return: float
    median_max_dd: float
    p05_max_dd: float
    p95_max_dd: float
    worst_max_dd: float
    prob_losing: float
    prob_exceed_observed_return: float
    prob_dd_gt_5: float
    prob_dd_gt_10: float
    prob_dd_gt_20: float
    losing_streak_p50: float
    losing_streak_p95: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _losing_streak(pnls: np.ndarray) -> int:
    best = 0
    cur = 0
    for x in pnls:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def run_trade_monte_carlo(
    trades: Sequence[Trade],
    *,
    initial_capital: float = 1000.0,
    n_sims: int = 5000,
    seed: int = 42,
    observed_return_pct: float | None = None,
    observed_max_dd_pct: float | None = None,
) -> tuple[MonteCarloSummary, np.ndarray, np.ndarray]:
    """
    Resample trade net_pnl with replacement; rebuild equity path.

    This estimates sampling uncertainty of the *observed trade set*, not
    future market risk. Label as model-based uncertainty.
    """
    pnls = np.array([t.net_pnl for t in trades], dtype=float)
    n = len(pnls)
    note = (
        "Monte Carlo resamples empirical trade PnLs with replacement. "
        "It does NOT prove future risk or guarantee drawdown bounds."
    )
    if n == 0:
        empty = MonteCarloSummary(
            n_sims=0,
            n_trades=0,
            observed_return_pct=0.0,
            observed_max_dd_pct=0.0,
            median_return=0.0,
            p05_return=0.0,
            p25_return=0.0,
            p75_return=0.0,
            p95_return=0.0,
            median_max_dd=0.0,
            p05_max_dd=0.0,
            p95_max_dd=0.0,
            worst_max_dd=0.0,
            prob_losing=0.0,
            prob_exceed_observed_return=0.0,
            prob_dd_gt_5=0.0,
            prob_dd_gt_10=0.0,
            prob_dd_gt_20=0.0,
            losing_streak_p50=0.0,
            losing_streak_p95=0.0,
            note=note + " No trades to resample.",
        )
        return empty, np.array([]), np.array([])

    rng = np.random.default_rng(seed)
    obs_ret = (
        observed_return_pct
        if observed_return_pct is not None
        else float(pnls.sum() / initial_capital * 100.0)
    )
    returns = np.empty(n_sims, dtype=float)
    max_dds = np.empty(n_sims, dtype=float)
    streaks = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        sample = rng.choice(pnls, size=n, replace=True)
        equity = np.empty(n + 1, dtype=float)
        equity[0] = initial_capital
        for j, pnl in enumerate(sample):
            equity[j + 1] = max(0.0, equity[j] + pnl)
        final = equity[-1]
        returns[i] = (final / initial_capital - 1.0) * 100.0
        max_dds[i] = max_drawdown_from_equity(equity)
        streaks[i] = _losing_streak(sample)

    obs_dd = observed_max_dd_pct if observed_max_dd_pct is not None else float(np.nan)
    summary = MonteCarloSummary(
        n_sims=n_sims,
        n_trades=n,
        observed_return_pct=float(obs_ret),
        observed_max_dd_pct=float(obs_dd) if obs_dd == obs_dd else 0.0,
        median_return=float(np.median(returns)),
        p05_return=float(np.percentile(returns, 5)),
        p25_return=float(np.percentile(returns, 25)),
        p75_return=float(np.percentile(returns, 75)),
        p95_return=float(np.percentile(returns, 95)),
        median_max_dd=float(np.median(max_dds)),
        p05_max_dd=float(np.percentile(max_dds, 5)),
        p95_max_dd=float(np.percentile(max_dds, 95)),
        worst_max_dd=float(np.max(max_dds)),
        prob_losing=float(np.mean(returns < 0)),
        prob_exceed_observed_return=float(np.mean(returns >= obs_ret)),
        prob_dd_gt_5=float(np.mean(max_dds > 5)),
        prob_dd_gt_10=float(np.mean(max_dds > 10)),
        prob_dd_gt_20=float(np.mean(max_dds > 20)),
        losing_streak_p50=float(np.median(streaks)),
        losing_streak_p95=float(np.percentile(streaks, 95)),
        note=note,
    )
    return summary, returns, max_dds
