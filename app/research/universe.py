"""Phase 9 universe helpers: symbols, timeframes, CSV resolution."""

from __future__ import annotations

from pathlib import Path

DEFAULT_SYMBOLS = ("BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT")
DEFAULT_TIMEFRAMES = ("15m", "1h", "4h")
PHASE9_FAMILIES = ("baseline", "trend_following", "momentum", "breakout", "mean_reversion")


def csv_path_for(data_dir: str | Path, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "-").upper()
    return Path(data_dir) / f"{safe}_{timeframe}.csv"


def discover_available(
    data_dir: str | Path,
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES,
) -> list[tuple[str, str, Path]]:
    """Return (symbol, timeframe, path) for CSVs that exist and are non-empty."""
    root = Path(data_dir)
    found: list[tuple[str, str, Path]] = []
    for symbol in symbols:
        for tf in timeframes:
            path = csv_path_for(root, symbol, tf)
            if path.is_file() and path.stat().st_size > 0:
                found.append((symbol, tf, path))
    return found
