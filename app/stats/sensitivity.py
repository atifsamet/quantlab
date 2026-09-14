"""Fee / slippage sensitivity for a fixed strategy (no optimization)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import pandas as pd

from app.backtest.engine import BacktestConfig
from app.research.params import ResearchParams
from app.risk.config import RiskConfig
from app.stats.benchmarks import run_signaled_backtest


@dataclass(frozen=True, slots=True)
class CostScenario:
    name: str
    fee_rate: float
    slippage_rate: float


DEFAULT_COST_SCENARIOS: tuple[CostScenario, ...] = (
    CostScenario("LOW", fee_rate=0.0005, slippage_rate=0.0002),
    CostScenario("BASE", fee_rate=0.001, slippage_rate=0.0005),
    CostScenario("HIGH", fee_rate=0.0015, slippage_rate=0.001),
)


@dataclass(slots=True)
class CostResult:
    scenario: str
    fee_rate: float
    slippage_rate: float
    return_pct: float
    trades: int
    max_dd: float
    total_fees: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cost_sensitivity(
    signaled: pd.DataFrame,
    *,
    scenarios: Sequence[CostScenario] = DEFAULT_COST_SCENARIOS,
    params: ResearchParams | None = None,
) -> list[CostResult]:
    params = params or ResearchParams(variant="breakout")
    risk_cfg = RiskConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
    )
    out: list[CostResult] = []
    for sc in scenarios:
        cfg = BacktestConfig(fee_rate=sc.fee_rate, slippage_rate=sc.slippage_rate)
        result = run_signaled_backtest(
            signaled, use_risk=True, config=cfg, risk_config=risk_cfg
        )
        out.append(
            CostResult(
                scenario=sc.name,
                fee_rate=sc.fee_rate,
                slippage_rate=sc.slippage_rate,
                return_pct=float(result.total_return_pct),
                trades=int(result.total_trades),
                max_dd=float(result.max_drawdown_pct),
                total_fees=float(sum(t.fees for t in result.trades)),
            )
        )
    return out


def slippage_break_even(
    signaled: pd.DataFrame,
    *,
    fee_rate: float = 0.001,
    slippage_grid: Sequence[float] | None = None,
    params: ResearchParams | None = None,
) -> dict[str, Any]:
    """
    Find approximate slippage where total return crosses <= 0 (fee fixed).

    Returns the first grid point that is unprofitable, or None if always > 0.
    """
    params = params or ResearchParams(variant="breakout")
    grid = list(slippage_grid or (0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01))
    risk_cfg = RiskConfig(
        atr_multiplier=params.atr_stop_multiplier,
        risk_reward_ratio=params.risk_reward_ratio,
    )
    rows: list[dict[str, Any]] = []
    break_even: float | None = None
    for slip in grid:
        cfg = BacktestConfig(fee_rate=fee_rate, slippage_rate=float(slip))
        result = run_signaled_backtest(
            signaled, use_risk=True, config=cfg, risk_config=risk_cfg
        )
        rows.append(
            {
                "fee_rate": fee_rate,
                "slippage_rate": float(slip),
                "return_pct": float(result.total_return_pct),
                "trades": int(result.total_trades),
                "max_dd": float(result.max_drawdown_pct),
            }
        )
        if break_even is None and result.total_return_pct <= 0:
            break_even = float(slip)
    return {
        "fee_rate_fixed": fee_rate,
        "break_even_slippage": break_even,
        "grid": rows,
        "note": (
            "Break-even is the lowest tested slippage with return <= 0 at fixed fee. "
            "Not optimized; grid search only."
        ),
    }
