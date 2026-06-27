"""
analytics.py
============
Descriptive and diagnostic analytics for any dataset.

* :class:`DescriptiveAnalytics` — summary, stats, correlations, quality score.
* :class:`DiagnosticAnalytics` — outliers, key drivers, anomalies, findings.

Every method is defensive: if an analysis can't run on the given data it returns
an empty/neutral result rather than raising, so the dashboard never crashes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import QUALITY_BANDS
from preprocessing import DataProfile
from utils import pct


@dataclass
class Finding:
    """A single diagnostic insight surfaced to the user / chatbot."""

    title: str
    detail: str
    severity: str = "info"  # info | warning | critical


class DescriptiveAnalytics:
    """Compute descriptive statistics and data-quality metrics."""

    def __init__(self, df: pd.DataFrame, profile: DataProfile):
        self.df = df
        self.profile = profile

    def summary(self) -> Dict[str, object]:
        df = self.df
        return {
            "rows": len(df),
            "columns": df.shape[1],
            "numeric_cols": len(self.profile.numeric),
            "categorical_cols": len(self.profile.categorical),
            "datetime_cols": len(self.profile.datetime),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_pct": pct(df.isna().sum().sum(), df.size),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_cells": int(df.size),
        }

    def column_info(self) -> pd.DataFrame:
        rows = []
        for col in self.df.columns:
            s = self.df[col]
            rows.append(
                {
                    "Column": col,
                    "Type": self.profile.column_type(col),
                    "Dtype": str(s.dtype),
                    "Non-Null": int(s.notna().sum()),
                    "Missing %": pct(s.isna().sum(), len(s)),
                    "Unique": int(s.nunique(dropna=True)),
                    "Example": str(s.dropna().iloc[0]) if s.notna().any() else "—",
                }
            )
        return pd.DataFrame(rows)

    def numeric_describe(self) -> pd.DataFrame:
        cols = [c for c in self.profile.numeric if c in self.df.columns]
        if not cols:
            return pd.DataFrame()
        desc = self.df[cols].describe().T
        desc["skew"] = self.df[cols].skew(numeric_only=True)
        desc["missing %"] = [pct(self.df[c].isna().sum(), len(self.df)) for c in cols]
        return desc.round(3)

    def correlation_matrix(self) -> pd.DataFrame:
        cols = [c for c in self.profile.numeric if c in self.df.columns]
        if len(cols) < 2:
            return pd.DataFrame()
        return self.df[cols].corr(numeric_only=True).round(3)

    def frequency_tables(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}
        for col in self.profile.categorical:
            if col not in self.df.columns:
                continue
            vc = self.df[col].value_counts(dropna=False).head(top_n)
            tables[col] = vc.rename_axis(col).reset_index(name="count")
        return tables

    def data_quality(self) -> Dict[str, object]:
        """Composite 0–100 data-quality score with component breakdown."""
        df = self.df
        completeness = 100 - pct(df.isna().sum().sum(), df.size)
        uniqueness = 100 - pct(df.duplicated().sum(), len(df)) if len(df) else 100
        # Consistency: fraction of columns with a single, sensible dtype.
        mixed = sum(
            1
            for c in df.columns
            if df[c].dtype == object and df[c].dropna().map(type).nunique() > 1
        )
        consistency = 100 - pct(mixed, df.shape[1]) if df.shape[1] else 100
        score = round(0.5 * completeness + 0.3 * uniqueness + 0.2 * consistency, 1)

        band = "Poor"
        for name, (lo, hi) in QUALITY_BANDS.items():
            if lo <= score < hi:
                band = name
                break
        return {
            "score": score,
            "band": band,
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "consistency": round(consistency, 1),
        }


class DiagnosticAnalytics:
    """Surface trends, anomalies, correlations and key drivers."""

    def __init__(self, df: pd.DataFrame, profile: DataProfile):
        self.df = df
        self.profile = profile

    def strong_correlations(self, threshold: float = 0.5) -> List[Dict[str, object]]:
        cols = [c for c in self.profile.numeric if c in self.df.columns]
        if len(cols) < 2:
            return []
        corr = self.df[cols].corr(numeric_only=True).abs()
        pairs = []
        seen = set()
        for a in cols:
            for b in cols:
                if a == b or (b, a) in seen:
                    continue
                seen.add((a, b))
                value = corr.loc[a, b]
                if pd.notna(value) and value >= threshold:
                    pairs.append(
                        {"a": a, "b": b, "corr": round(float(self.df[[a, b]].corr().iloc[0, 1]), 3)}
                    )
        return sorted(pairs, key=lambda d: abs(d["corr"]), reverse=True)

    def outlier_summary(self) -> List[Dict[str, object]]:
        results = []
        for col in self.profile.numeric:
            if col not in self.df.columns:
                continue
            s = self.df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((s < low) | (s > high)).sum())
            if n_out:
                results.append(
                    {"column": col, "count": n_out, "pct": pct(n_out, len(s))}
                )
        return sorted(results, key=lambda d: d["pct"], reverse=True)

    def key_drivers(self, target: str, top_n: int = 8) -> pd.DataFrame:
        """Rank features by correlation / association with ``target``."""
        if target not in self.df.columns:
            return pd.DataFrame()
        df = self.df.copy()
        y = df[target]
        scores = {}

        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            # Regression target: absolute Pearson correlation.
            for col in self.profile.numeric:
                if col == target or col not in df.columns:
                    continue
                c = df[[col, target]].corr().iloc[0, 1]
                if pd.notna(c):
                    scores[col] = abs(c)
        else:
            # Classification target: correlation ratio (eta) for numerics,
            # Cramér's V approximation for categoricals.
            for col in self.profile.numeric:
                if col == target or col not in df.columns:
                    continue
                scores[col] = self._correlation_ratio(y, df[col])
            for col in self.profile.categorical:
                if col == target or col not in df.columns:
                    continue
                scores[col] = self._cramers_v(df[col], y)

        if not scores:
            return pd.DataFrame()
        out = (
            pd.DataFrame({"feature": list(scores), "strength": list(scores.values())})
            .dropna()
            .sort_values("strength", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        out["strength"] = out["strength"].round(3)
        return out

    def time_trends(self) -> Dict[str, pd.DataFrame]:
        """Aggregate numeric columns over the first datetime column, if any."""
        if not self.profile.datetime or not self.profile.numeric:
            return {}
        date_col = self.profile.datetime[0]
        if date_col not in self.df.columns:
            return {}
        df = self.df.dropna(subset=[date_col]).copy()
        if df.empty:
            return {}
        df = df.set_index(date_col).sort_index()
        freq = "ME" if df.index.to_series().diff().median() and df.index.nunique() > 24 else "D"
        trends = {}
        for col in self.profile.numeric[:4]:
            if col in df.columns:
                trends[col] = (
                    df[col].resample(freq).mean().dropna().reset_index()
                )
        return trends

    def findings(self, target: str | None = None) -> List[Finding]:
        """Aggregate human-readable findings for the diagnostics panel."""
        out: List[Finding] = []

        missing = self.df.isna().mean()
        worst = missing[missing > 0.2]
        for col, frac in worst.items():
            out.append(
                Finding(
                    title=f"High missingness in '{col}'",
                    detail=f"{frac*100:.1f}% of values were missing before imputation.",
                    severity="warning",
                )
            )

        for o in self.outlier_summary()[:3]:
            out.append(
                Finding(
                    title=f"Outliers in '{o['column']}'",
                    detail=f"{o['count']} values ({o['pct']}%) fall outside 1.5×IQR.",
                    severity="warning" if o["pct"] > 5 else "info",
                )
            )

        for c in self.strong_correlations(0.7)[:3]:
            out.append(
                Finding(
                    title=f"Strong correlation: {c['a']} ↔ {c['b']}",
                    detail=f"Pearson r = {c['corr']}. These features move together.",
                    severity="info",
                )
            )

        if target:
            drivers = self.key_drivers(target)
            if not drivers.empty:
                top = drivers.iloc[0]
                out.append(
                    Finding(
                        title=f"Top driver of '{target}'",
                        detail=f"'{top['feature']}' shows the strongest association "
                        f"(strength {top['strength']}).",
                        severity="info",
                    )
                )
        return out

    # -- statistical helpers ---------------------------------------------- #
    @staticmethod
    def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        """Eta correlation ratio between a categorical and a numeric variable."""
        df = pd.DataFrame({"cat": categories, "val": values}).dropna()
        if df.empty:
            return np.nan
        groups = df.groupby("cat")["val"]
        y_mean = df["val"].mean()
        ss_between = sum(
            len(g) * (g.mean() - y_mean) ** 2 for _, g in groups
        )
        ss_total = ((df["val"] - y_mean) ** 2).sum()
        return float(np.sqrt(ss_between / ss_total)) if ss_total else np.nan

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Cramér's V association between two categorical variables."""
        confusion = pd.crosstab(x, y)
        if confusion.size == 0:
            return np.nan
        chi2 = _chi2(confusion.values)
        n = confusion.values.sum()
        if n == 0:
            return np.nan
        phi2 = chi2 / n
        r, k = confusion.shape
        denom = min(k - 1, r - 1)
        return float(np.sqrt(phi2 / denom)) if denom else np.nan


def _chi2(observed: np.ndarray) -> float:
    """Chi-squared statistic without a scipy dependency."""
    observed = observed.astype(float)
    total = observed.sum()
    if total == 0:
        return 0.0
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(stat.sum())
