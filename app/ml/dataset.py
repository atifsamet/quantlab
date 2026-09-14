"""Build supervised datasets from deterministic strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.backtest.engine import load_ohlcv_csv
from app.ml.features import FEATURE_COLUMNS, enrich_feature_frame, feature_matrix
from app.ml.labels import LabelConfig, label_signals
from app.research.params import ResearchParams
from app.research.variants import build_strategy


@dataclass(slots=True)
class DatasetStats:
    n_rows: int
    n_long: int
    n_short: int
    n_positive: int
    n_negative: int
    positive_pct: float
    negative_pct: float
    symbols: list[str]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_long": self.n_long,
            "n_short": self.n_short,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_pct": self.positive_pct,
            "negative_pct": self.negative_pct,
            "symbols": self.symbols,
        }


def prepare_signaled_frame(
    ohlcv: pd.DataFrame,
    params: ResearchParams,
) -> pd.DataFrame:
    """Indicators + strategy signals + ML features."""
    framed = enrich_feature_frame(ohlcv, params)
    strategy = build_strategy(params)
    return strategy.generate_signals(framed)


def build_samples_from_frame(
    framed: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    label_cfg: LabelConfig | None = None,
    include_asset_id: bool = False,
) -> pd.DataFrame:
    """
    Extract one row per LONG/SHORT signal with features + label.

    Features are from the signal bar only. Labels use future path after entry.
    """
    label_cfg = label_cfg or LabelConfig()
    labels = label_signals(framed, cfg=label_cfg)
    rows: list[dict] = []
    for i in range(len(framed)):
        side = str(framed.iloc[i]["signal"]).upper()
        if side not in ("LONG", "SHORT"):
            continue
        y = labels.iloc[i]
        if pd.isna(y):
            continue
        row = framed.iloc[i]
        sample = {col: row[col] for col in FEATURE_COLUMNS if col != "side_sign"}
        sample["side_sign"] = 1.0 if side == "LONG" else -1.0
        sample["label"] = float(y)
        sample["side"] = side
        sample["symbol"] = symbol
        sample["timeframe"] = timeframe
        sample["timestamp"] = row["timestamp"]
        sample["signal_index"] = i
        if include_asset_id:
            # Simple ordinal placeholder; one-hots applied later if enabled.
            sample["asset_code"] = hash(symbol) % 997 / 997.0
        rows.append(sample)
    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS) + ["label", "side", "symbol", "timeframe", "timestamp"])
    return pd.DataFrame(rows)


def build_multi_asset_dataset(
    paths: Iterable[tuple[str, str, Path]],
    params: ResearchParams,
    *,
    label_cfg: LabelConfig | None = None,
    include_asset_id: bool = False,
) -> pd.DataFrame:
    """Load many CSVs and stack signal samples (shared feature space)."""
    parts: list[pd.DataFrame] = []
    for symbol, timeframe, path in paths:
        ohlcv = load_ohlcv_csv(path)
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        framed = prepare_signaled_frame(ohlcv, params)
        part = build_samples_from_frame(
            framed,
            symbol=symbol,
            timeframe=timeframe,
            label_cfg=label_cfg,
            include_asset_id=include_asset_id,
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    # Drop incomplete feature rows
    feat = feature_matrix(out)
    mask = feat.notna().all(axis=1)
    return out.loc[mask].reset_index(drop=True)


def chronological_split(
    samples: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict chronological splits by signal timestamp (no shuffle)."""
    ts = pd.to_datetime(samples["timestamp"], utc=True)
    train = samples[(ts >= pd.Timestamp(train_start, tz="UTC")) & (ts < pd.Timestamp(train_end, tz="UTC"))]
    val = samples[(ts >= pd.Timestamp(val_start, tz="UTC")) & (ts < pd.Timestamp(val_end, tz="UTC"))]
    test = samples[(ts >= pd.Timestamp(test_start, tz="UTC")) & (ts < pd.Timestamp(test_end, tz="UTC"))]
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def dataset_stats(samples: pd.DataFrame) -> DatasetStats:
    if samples.empty:
        return DatasetStats(0, 0, 0, 0, 0, 0.0, 100.0, [])
    n = len(samples)
    n_pos = int((samples["label"] == 1).sum())
    n_neg = int((samples["label"] == 0).sum())
    return DatasetStats(
        n_rows=n,
        n_long=int((samples["side"] == "LONG").sum()),
        n_short=int((samples["side"] == "SHORT").sum()),
        n_positive=n_pos,
        n_negative=n_neg,
        positive_pct=100.0 * n_pos / n,
        negative_pct=100.0 * n_neg / n,
        symbols=sorted(samples["symbol"].unique().tolist()),
    )
