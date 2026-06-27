"""
machine_learning.py
===================
A lightweight AutoML engine. Given a target column it automatically:

* detects whether the task is classification or regression,
* builds a clean feature matrix,
* trains and cross-validates several models,
* evaluates them on a hold-out split,
* ranks them and recommends the best, with ROC / confusion-matrix / feature
  importance artefacts for the winner.

Everything is wrapped in try/except so a single failing model never breaks the
whole comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import ML
from preprocessing import DataProfile, encode_features
from utils import safe_sample

try:  # XGBoost is optional.
    from xgboost import XGBClassifier, XGBRegressor

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    estimator: object = None
    roc: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    confusion: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class AutoMLReport:
    task: str = "none"               # classification | regression | none
    target: Optional[str] = None
    results: List[ModelResult] = field(default_factory=list)
    best: Optional[str] = None
    feature_importance: Optional[pd.DataFrame] = None
    class_labels: List[str] = field(default_factory=list)
    n_features: int = 0
    note: str = ""


class AutoML:
    """Train and compare models for an automatically detected task."""

    def __init__(self, config=ML):
        self.cfg = config

    # ------------------------------------------------------------------ #
    def detect_task(self, y: pd.Series) -> str:
        """Classification if the target is discrete/low-cardinality, else regression."""
        y = y.dropna()
        if y.empty:
            return "none"
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        if pd.api.types.is_bool_dtype(y):
            return "classification"
        nunique = y.nunique()
        if nunique <= 20 and (y.dropna() % 1 == 0).all():
            return "classification"
        return "regression"

    # ------------------------------------------------------------------ #
    def run(
        self, df: pd.DataFrame, profile: DataProfile, target: str
    ) -> AutoMLReport:
        report = AutoMLReport(target=target)
        if target not in df.columns:
            report.note = "Target column not found."
            return report

        data = df.dropna(subset=[target]).copy()
        data = safe_sample(data, self.cfg.max_training_rows, self.cfg.random_state)
        if len(data) < 30:
            report.note = "Not enough rows to train reliable models (need ≥ 30)."
            return report

        task = self.detect_task(data[target])
        report.task = task
        if task == "none":
            report.note = "Could not determine a modelling task for this target."
            return report

        # Build feature set: everything except target & identifiers.
        feature_cols = [
            c for c in df.columns
            if c != target and c not in profile.identifier
            and df[c].nunique(dropna=True) <= self.cfg.max_feature_cardinality
            or c in profile.numeric
        ]
        feature_cols = [c for c in dict.fromkeys(feature_cols) if c != target]
        if not feature_cols:
            report.note = "No usable feature columns after filtering."
            return report

        X = encode_features(data, feature_cols, self.cfg.max_feature_cardinality)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        report.n_features = X.shape[1]

        y = data[target]
        encoder = None
        if task == "classification":
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index)
            report.class_labels = [str(c) for c in encoder.classes_]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state,
            stratify=y if task == "classification" and y.nunique() > 1 else None,
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = self._models(task)
        for name, (model, needs_scaling) in models.items():
            result = self._train_one(
                name, model, needs_scaling, task,
                X_train, X_test, X_train_s, X_test_s, y_train, y_test,
            )
            report.results.append(result)

        valid = [r for r in report.results if r.error is None]
        if not valid:
            report.note = "All models failed to train on this dataset."
            return report

        # Rank: by F1 (classification) or R² (regression).
        rank_key = "f1" if task == "classification" else "r2"
        valid.sort(key=lambda r: r.metrics.get(rank_key, -np.inf), reverse=True)
        report.results = valid + [r for r in report.results if r.error is not None]
        report.best = valid[0].name
        report.feature_importance = self._importance(valid[0].estimator, X.columns)
        return report

    # ------------------------------------------------------------------ #
    def _models(self, task: str) -> Dict[str, tuple]:
        rs = self.cfg.random_state
        if task == "classification":
            models = {
                "Logistic Regression": (LogisticRegression(max_iter=1000), True),
                "Decision Tree": (DecisionTreeClassifier(random_state=rs), False),
                "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=rs, n_jobs=-1), False),
                "Gradient Boosting": (GradientBoostingClassifier(random_state=rs), False),
                "K-Nearest Neighbors": (KNeighborsClassifier(), True),
                "SVM": (SVC(probability=True, random_state=rs), True),
                "Naive Bayes": (GaussianNB(), True),
            }
            if _HAS_XGB:
                models["XGBoost"] = (
                    XGBClassifier(eval_metric="logloss", random_state=rs,
                                  verbosity=0, n_jobs=-1), False,
                )
        else:
            models = {
                "Linear Regression": (LinearRegression(), True),
                "Decision Tree": (DecisionTreeRegressor(random_state=rs), False),
                "Random Forest": (RandomForestRegressor(n_estimators=150, random_state=rs, n_jobs=-1), False),
                "Gradient Boosting": (GradientBoostingRegressor(random_state=rs), False),
            }
            if _HAS_XGB:
                models["XGBoost"] = (
                    XGBRegressor(random_state=rs, verbosity=0, n_jobs=-1), False,
                )
        return models

    def _train_one(
        self, name, model, needs_scaling, task,
        X_train, X_test, X_train_s, X_test_s, y_train, y_test,
    ) -> ModelResult:
        try:
            xtr = X_train_s if needs_scaling else X_train.values
            xte = X_test_s if needs_scaling else X_test.values
            model.fit(xtr, y_train)
            preds = model.predict(xte)

            if task == "classification":
                metrics = self._classification_metrics(
                    model, xtr, xte, y_train, y_test, preds
                )
                roc = self._roc(model, xte, y_test)
                cm = confusion_matrix(y_test, preds)
                return ModelResult(name, metrics, model, roc, cm)
            metrics = self._regression_metrics(model, xtr, xte, y_train, y_test, preds)
            return ModelResult(name, metrics, model)
        except Exception as exc:  # noqa: BLE001
            return ModelResult(name, {}, error=str(exc)[:200])

    def _classification_metrics(self, model, xtr, xte, y_train, y_test, preds) -> Dict[str, float]:
        avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, preds, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, preds, average=avg, zero_division=0)),
        }
        try:
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                proba = model.predict_proba(xte)[:, 1]
                metrics["auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass
        try:
            cv = cross_val_score(model, xtr, y_train, cv=min(self.cfg.cv_folds, 5),
                                 scoring="accuracy", n_jobs=-1)
            metrics["cv_accuracy"] = float(cv.mean())
        except Exception:
            pass
        return metrics

    def _regression_metrics(self, model, xtr, xte, y_train, y_test, preds) -> Dict[str, float]:
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
        }
        try:
            cv = cross_val_score(model, xtr, y_train, cv=min(self.cfg.cv_folds, 5),
                                 scoring="r2", n_jobs=-1)
            metrics["cv_r2"] = float(cv.mean())
        except Exception:
            pass
        return metrics

    @staticmethod
    def _roc(model, xte, y_test):
        try:
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                proba = model.predict_proba(xte)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                return fpr, tpr, float(roc_auc_score(y_test, proba))
        except Exception:
            return None
        return None

    @staticmethod
    def _importance(model, columns) -> Optional[pd.DataFrame]:
        try:
            if hasattr(model, "feature_importances_"):
                vals = model.feature_importances_
            elif hasattr(model, "coef_"):
                vals = np.abs(np.ravel(model.coef_))
                if len(vals) != len(columns):
                    return None
            else:
                return None
            return (
                pd.DataFrame({"feature": columns, "importance": vals})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except Exception:
            return None


def comparison_table(report: AutoMLReport) -> pd.DataFrame:
    """Tidy dataframe comparing all successfully trained models."""
    rows = []
    for r in report.results:
        if r.error is not None:
            continue
        row = {"Model": r.name}
        row.update({k: round(v, 4) for k, v in r.metrics.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and report.best:
        df["Best"] = df["Model"].eq(report.best).map({True: "⭐", False: ""})
    return df
