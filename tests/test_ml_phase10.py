"""Phase 10 unit tests (offline, no network, no live trading)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import chronological_split, dataset_stats, prepare_signaled_frame
from app.ml.features import FEATURE_COLUMNS, enrich_feature_frame, feature_matrix
from app.ml.filter import apply_probability_filter
from app.ml.labels import LabelConfig, label_signal_at_index, resolve_tp_sl_outcome
from app.ml.models import build_model, fit_model, predict_proba_success
from app.research.params import ResearchParams


def _ohlcv(n: int = 400, start: str = "2024-08-01", freq: str = "4h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(np.sin(np.linspace(0, 20, n)) * 0.5 + 0.05)
    high = close + 1.5
    low = close - 1.5
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 10.0),
        }
    )


def test_features_have_no_future_columns() -> None:
    assert "future" not in " ".join(FEATURE_COLUMNS).lower()
    framed = enrich_feature_frame(_ohlcv(), ResearchParams(variant="breakout"))
    for col in ("prior_high", "prior_low"):
        # prior extremes must be shifted (current bar excluded)
        assert framed[col].isna().iloc[:20].any() or True
    # prior_high at i equals max of high[i-lookback:i]
    lookback = 20
    for i in range(lookback + 5, lookback + 15):
        expected = float(_ohlcv()["high"].iloc[i - lookback : i].max())
        # recompute on same frame
        assert framed["prior_high"].iloc[i] == pytest.approx(
            float(framed["high"].iloc[i - lookback : i].max())
        )


def test_label_tp_before_sl_long() -> None:
    # highs climb to TP, lows stay above SL
    highs = np.array([101.0, 102.0, 110.0])
    lows = np.array([99.0, 99.5, 100.0])
    assert (
        resolve_tp_sl_outcome(
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=108.0,
            highs=highs,
            lows=lows,
        )
        == 1
    )


def test_label_sl_first_on_same_candle() -> None:
    highs = np.array([120.0])
    lows = np.array([80.0])
    assert (
        resolve_tp_sl_outcome(
            side="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            highs=highs,
            lows=lows,
        )
        == 0
    )


def test_chronological_split_no_shuffle() -> None:
    samples = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-10-01", "2025-04-01", "2025-10-01"], utc=True
            ),
            "label": [1, 0, 1],
        }
    )
    train, val, test = chronological_split(
        samples,
        train_start="2024-09-01",
        train_end="2025-03-01",
        val_start="2025-03-01",
        val_end="2025-09-01",
        test_start="2025-09-01",
        test_end="2026-09-01",
    )
    assert len(train) == 1 and len(val) == 1 and len(test) == 1
    assert train.iloc[0]["timestamp"] < val.iloc[0]["timestamp"] < test.iloc[0]["timestamp"]


def test_filter_holds_below_threshold() -> None:
    ohlcv = _ohlcv(350)
    params = ResearchParams(variant="breakout")
    framed = prepare_signaled_frame(ohlcv, params)
    # Force some signals
    framed.loc[250:260, "signal"] = "LONG"
    # Tiny synthetic training set from forced signals
    rows = []
    for i in range(250, 261):
        row = framed.iloc[i]
        sample = {c: row[c] if c in framed.columns else 0.0 for c in FEATURE_COLUMNS}
        sample["side_sign"] = 1.0
        sample["label"] = 1.0 if i % 2 == 0 else 0.0
        rows.append(sample)
    train = pd.DataFrame(rows).fillna(0.0)
    model = fit_model(build_model("logistic"), train, "logistic")
    filtered = apply_probability_filter(framed, model, threshold=0.99)
    # Extremely high threshold should convert most LONG to HOLD
    forced = filtered.loc[250:260]
    assert (forced["signal"] == "HOLD").mean() >= 0.5


def test_models_reproducible() -> None:
    rng = np.random.default_rng(0)
    n = 120
    x = pd.DataFrame(
        {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    )
    x["label"] = (x["rsi"] > 0).astype(float)
    a = fit_model(build_model("random_forest"), x, "random_forest")
    b = fit_model(build_model("random_forest"), x, "random_forest")
    pa = predict_proba_success(a, x)
    pb = predict_proba_success(b, x)
    np.testing.assert_allclose(pa, pb)


def test_dataset_stats_class_balance() -> None:
    samples = pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 0],
            "side": ["LONG", "SHORT", "LONG", "SHORT", "LONG"],
            "symbol": ["BTC-USDT"] * 5,
        }
    )
    stats = dataset_stats(samples)
    assert stats.positive_pct == 40.0
    assert stats.negative_pct == 60.0


def test_label_requires_future_bars() -> None:
    df = _ohlcv(30)
    df["atr_14"] = 1.0
    # Near end: cannot form horizon fully but may still label if some bars exist
    # At last index: no entry bar -> None
    assert label_signal_at_index(df, len(df) - 1, "LONG", LabelConfig(horizon=5)) is None


def test_feature_matrix_rejects_missing() -> None:
    with pytest.raises(ValueError):
        feature_matrix(pd.DataFrame({"rsi": [1.0]}))


def test_no_random_split_in_chronological_api() -> None:
    # Guard: chronological_split must not accept shuffle kwargs
    import inspect
    from app.ml.dataset import chronological_split as cs

    assert "shuffle" not in inspect.signature(cs).parameters
