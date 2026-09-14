"""Apply ML success probability as a HOLD filter on strategy signals."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from app.ml.features import FEATURE_COLUMNS, feature_matrix


DEFAULT_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)


def apply_probability_filter(
    signaled: pd.DataFrame,
    model: Pipeline,
    *,
    threshold: float,
    probability_column: str = "ml_p_success",
) -> pd.DataFrame:
    """
    Keep LONG/SHORT only when P(success) >= threshold; otherwise HOLD.

    Rows that are already HOLD stay HOLD. Feature NaNs on signal rows -> HOLD.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    out = signaled.copy()
    out[probability_column] = float("nan")
    out["signal_raw"] = out["signal"].astype(str).str.upper()
    out["signal"] = out["signal_raw"]

    mask = out["signal_raw"].isin(["LONG", "SHORT"])
    if not mask.any():
        return out

    # Build feature rows aligned to signaled index
    feat_source = out.loc[mask].copy()
    feat_source["side_sign"] = feat_source["signal_raw"].map({"LONG": 1.0, "SHORT": -1.0})
    for col in FEATURE_COLUMNS:
        if col not in feat_source.columns and col != "side_sign":
            raise ValueError(f"Missing feature column '{col}' for ML filter")

    x = feature_matrix(feat_source)
    valid = x.notna().all(axis=1)
    proba_full = pd.Series(float("nan"), index=feat_source.index, dtype=float)
    if valid.any():
        clf = model.named_steps["clf"]
        classes = list(clf.classes_)
        raw = model.predict_proba(x.loc[valid])
        if 1 in classes:
            proba_full.loc[valid] = raw[:, classes.index(1)]
        else:
            proba_full.loc[valid] = 0.0

    out.loc[feat_source.index, probability_column] = proba_full
    keep = mask & (out[probability_column] >= threshold)
    drop = mask & ~keep
    out.loc[drop, "signal"] = "HOLD"
    if "signal_reason" in out.columns:
        out.loc[drop, "signal_reason"] = "ml_filter_reject"
        out.loc[keep, "signal_reason"] = out.loc[keep, "signal_reason"].astype(str) + "|ml_pass"
    return out
