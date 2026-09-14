"""Optional simple plots for Phase 11 (matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _try_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def save_histogram(
    values: Sequence[float],
    *,
    path: Path,
    title: str,
    xlabel: str,
    observed: float | None = None,
) -> bool:
    plt = _try_pyplot()
    if plt is None or len(values) == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.asarray(values, dtype=float), bins=40, color="#4a6fa5", alpha=0.85)
    if observed is not None:
        ax.axvline(observed, color="#c0392b", linewidth=2, label=f"observed={observed:.2f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def save_equity_curve(
    timestamps: Sequence,
    equity: Sequence[float],
    *,
    path: Path,
    title: str,
) -> bool:
    plt = _try_pyplot()
    if plt is None or len(equity) == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timestamps, equity, color="#2c3e50", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def save_benchmark_bars(
    labels: Sequence[str],
    returns: Sequence[float],
    *,
    path: Path,
    title: str,
) -> bool:
    plt = _try_pyplot()
    if plt is None or not labels:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4a6fa5" if r >= 0 else "#c0392b" for r in returns]
    ax.bar(list(labels), list(returns), color=colors)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("return %")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
