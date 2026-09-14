"""Transaction-cost helpers for Phase 9 (fee / slippage / funding-aware modes)."""

from __future__ import annotations

from dataclasses import dataclass

from app.backtest.engine import BacktestConfig


@dataclass(frozen=True, slots=True)
class CostScenario:
    """Named cost assumption set for sensitivity analysis."""

    name: str
    fee_rate: float
    slippage_rate: float
    apply_funding: bool = False

    def to_backtest_config(self, *, initial_capital: float = 1000.0) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            apply_funding=self.apply_funding,
            leverage=1.0,
        )


# A) fee + slippage only (default research path)
# B) funding-aware — only meaningful when a real funding_rate column exists
DEFAULT_COST_SCENARIOS: tuple[CostScenario, ...] = (
    CostScenario("base_fee_slip", fee_rate=0.001, slippage_rate=0.0005, apply_funding=False),
    CostScenario("higher_fee", fee_rate=0.0015, slippage_rate=0.0005, apply_funding=False),
    CostScenario("higher_slip", fee_rate=0.001, slippage_rate=0.001, apply_funding=False),
    CostScenario(
        "funding_aware_if_data",
        fee_rate=0.001,
        slippage_rate=0.0005,
        apply_funding=True,
    ),
)


def has_funding_column(frame) -> bool:
    return "funding_rate" in getattr(frame, "columns", [])
