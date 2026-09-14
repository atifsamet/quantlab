"""Phase 8/9 strategy research (offline, no live trading)."""

from app.research.diagnostics import diagnose_baseline_filters, format_diagnostic_report
from app.research.params import ResearchParams, validate_research_params
from app.research.search import (
    CandidateResult,
    RobustnessConfig,
    build_small_search_grid,
    rank_candidates,
    run_research_pipeline,
)
from app.research.variants import (
    STRATEGY_FACTORIES,
    BreakoutMomentumStrategy,
    BreakoutStrategy,
    EmaCrossRsiStrategy,
    EmaMacdStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
    TrendMomentumStrategy,
    build_strategy,
)

__all__ = [
    "BreakoutMomentumStrategy",
    "BreakoutStrategy",
    "CandidateResult",
    "EmaCrossRsiStrategy",
    "EmaMacdStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "ResearchParams",
    "RobustnessConfig",
    "STRATEGY_FACTORIES",
    "TrendFollowingStrategy",
    "TrendMomentumStrategy",
    "build_small_search_grid",
    "build_strategy",
    "diagnose_baseline_filters",
    "format_diagnostic_report",
    "rank_candidates",
    "run_research_pipeline",
    "validate_research_params",
]
