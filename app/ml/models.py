"""Simple interpretable classifiers for Phase 10 signal filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.ml.features import FEATURE_COLUMNS, feature_matrix

ModelName = Literal["logistic", "random_forest", "gradient_boosting"]
RANDOM_SEED = 42


@dataclass(slots=True)
class ClassificationReport:
    model: str
    n_samples: int
    positive_pct: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    calibration_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_samples": self.n_samples,
            "positive_pct": self.positive_pct,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "calibration_error": self.calibration_error,
        }


def build_model(name: ModelName) -> Pipeline:
    """
    Limited, reproducible model configs (no unrestricted hyperparameter search).
    """
    if name == "logistic":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])
    if name == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
        )
        return Pipeline([("clf", clf)])
    raise ValueError(f"Unknown model: {name}")


def fit_model(model: Pipeline, samples: pd.DataFrame, name: ModelName) -> Pipeline:
    x = feature_matrix(samples)
    y = samples["label"].astype(int).to_numpy()
    if name == "gradient_boosting":
        # sklearn GB has no class_weight; use sample weights
        sw = compute_sample_weight("balanced", y)
        model.fit(x, y, clf__sample_weight=sw)
    else:
        model.fit(x, y)
    return model


def predict_proba_success(model: Pipeline, samples: pd.DataFrame) -> np.ndarray:
    x = feature_matrix(samples)
    proba = model.predict_proba(x)
    # Column for class 1
    classes = list(model.named_steps["clf"].classes_)
    if 1 not in classes:
        return np.zeros(len(samples), dtype=float)
    idx = classes.index(1)
    return proba[:, idx]


def evaluate_classification(model: Pipeline, samples: pd.DataFrame, name: str) -> ClassificationReport:
    if samples.empty:
        return ClassificationReport(name, 0, 0.0, 0.0, 0.0, 0.0, None, None, None)
    y_true = samples["label"].astype(int).to_numpy()
    proba = predict_proba_success(model, samples)
    y_pred = (proba >= 0.5).astype(int)
    pos_pct = float(y_true.mean() * 100.0)
    roc = None
    pr = None
    cal_err = None
    if len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, proba))
        pr = float(average_precision_score(y_true, proba))
        try:
            frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
            cal_err = float(np.mean(np.abs(frac_pos - mean_pred)))
        except ValueError:
            cal_err = None
    return ClassificationReport(
        model=name,
        n_samples=len(samples),
        positive_pct=pos_pct,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
        calibration_error=cal_err,
    )


def feature_importance(model: Pipeline, name: ModelName) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    if name == "logistic":
        coefs = clf.coef_[0]
        return (
            pd.DataFrame({"feature": list(FEATURE_COLUMNS), "importance": coefs, "kind": "coefficient"})
            .assign(abs_importance=lambda d: d["importance"].abs())
            .sort_values("abs_importance", ascending=False)
            .drop(columns=["abs_importance"])
            .reset_index(drop=True)
        )
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        return (
            pd.DataFrame({"feature": list(FEATURE_COLUMNS), "importance": imp, "kind": "gini"})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    raise ValueError(f"No importance available for {name}")
