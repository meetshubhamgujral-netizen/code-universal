"""
preprocessing.py
================
Universal, dataset-agnostic preprocessing.

The :class:`DataProfiler` inspects an arbitrary dataframe and classifies every
column (numeric / categorical / datetime / boolean / identifier) and proposes
candidate target columns. The :class:`DataPreprocessor` then performs automatic
cleaning: missing-value handling, duplicate removal, outlier flagging, safe
type conversion and optional encoding.

No dataset-specific rules are hard-coded — everything is heuristic and driven by
the thresholds in :mod:`config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config import DETECTION, PREPROCESS


# --------------------------------------------------------------------------- #
#  Data profile container
# --------------------------------------------------------------------------- #
@dataclass
class DataProfile:
    """A structured description of a dataset's columns and properties."""

    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    boolean: List[str] = field(default_factory=list)
    identifier: List[str] = field(default_factory=list)
    target_candidates: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    def column_type(self, col: str) -> str:
        """Return the inferred semantic type of a single column."""
        for kind in ("numeric", "categorical", "datetime", "boolean", "identifier"):
            if col in getattr(self, kind):
                return kind
        return "unknown"

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "datetime": self.datetime,
            "boolean": self.boolean,
            "identifier": self.identifier,
            "target_candidates": self.target_candidates,
        }


# --------------------------------------------------------------------------- #
#  Profiler
# --------------------------------------------------------------------------- #
class DataProfiler:
    """Detect column semantics for any structured dataset."""

    def __init__(self, config=DETECTION):
        self.cfg = config

    def profile(self, df: pd.DataFrame) -> DataProfile:
        profile = DataProfile(n_rows=len(df), n_cols=df.shape[1])
        n = max(len(df), 1)

        for col in df.columns:
            series = df[col]
            nunique = series.nunique(dropna=True)

            if self._is_boolean(series):
                profile.boolean.append(col)
            elif self._is_datetime(series):
                profile.datetime.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                # Low-cardinality integers behave like categories (e.g. ratings).
                if (
                    nunique <= self.cfg.categorical_numeric_max_unique
                    and pd.api.types.is_integer_dtype(series.dropna().astype("Int64"))
                ):
                    profile.categorical.append(col)
                else:
                    profile.numeric.append(col)
            else:
                # Object/string column: identifier vs categorical.
                if nunique / n >= self.cfg.identifier_unique_ratio or self._looks_like_id(col):
                    profile.identifier.append(col)
                else:
                    profile.categorical.append(col)

        profile.target_candidates = self._target_candidates(df, profile)
        return profile

    # -- individual heuristics --------------------------------------------- #
    @staticmethod
    def _is_boolean(series: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series):
            return True
        non_null = series.dropna()
        if non_null.empty:
            return False
        uniques = {str(v).strip().lower() for v in non_null.unique()}
        bool_sets = [{"true", "false"}, {"yes", "no"}, {"y", "n"}, {"0", "1"}, {"t", "f"}]
        return any(uniques.issubset(s) for s in bool_sets) and len(uniques) <= 2

    def _is_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return False
        sample = series.dropna().astype(str)
        if sample.empty:
            return False
        sample = sample.sample(
            min(len(sample), self.cfg.datetime_inference_sample), random_state=0
        )
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        ratio = parsed.notna().mean()
        return bool(ratio >= self.cfg.datetime_parse_threshold)

    @staticmethod
    def _looks_like_id(col: str) -> bool:
        name = col.lower().replace("_", "").replace(" ", "")
        return name in {"id", "index", "key", "uuid", "guid"} or name.endswith("id")

    def _target_candidates(self, df: pd.DataFrame, profile: DataProfile) -> List[str]:
        """Heuristically rank columns that could serve as a prediction target."""
        candidates: List[str] = []
        keywords = (
            "target", "label", "class", "outcome", "result", "churn", "default",
            "fraud", "price", "amount", "sales", "revenue", "score", "rating",
            "status", "approved", "risk", "survived", "diagnosis", "y",
        )
        # Categorical / boolean targets (classification).
        for col in profile.categorical + profile.boolean:
            nunique = df[col].nunique(dropna=True)
            if self.cfg.target_class_min <= nunique <= self.cfg.target_class_max:
                candidates.append(col)
        # Numeric targets (regression) — continuous columns.
        candidates.extend(profile.numeric)

        # Promote columns whose name hints at being a target.
        named = [c for c in df.columns if any(k in c.lower() for k in keywords)]
        ordered = [c for c in named if c in candidates]
        ordered += [c for c in candidates if c not in ordered]
        return ordered


# --------------------------------------------------------------------------- #
#  Preprocessor
# --------------------------------------------------------------------------- #
class DataPreprocessor:
    """Automatic, fault-tolerant cleaning for arbitrary datasets."""

    def __init__(self, config=PREPROCESS):
        self.cfg = config
        self.report: Dict[str, object] = {}

    def clean(self, df: pd.DataFrame, profile: DataProfile) -> pd.DataFrame:
        """Return a cleaned copy of ``df`` and populate ``self.report``."""
        out = df.copy()
        report: Dict[str, object] = {}

        # 1. Parse datetime columns so downstream analytics can use them.
        for col in profile.datetime:
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")

        # 2. Coerce boolean-like columns to real booleans.
        for col in profile.boolean:
            out[col] = self._to_boolean(out[col])

        # 3. Safe numeric coercion for numeric columns stored as strings.
        for col in profile.numeric:
            if not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(
                    out[col].astype(str).str.replace(r"[,$%]", "", regex=True),
                    errors="coerce",
                )

        # 4. Drop duplicate rows.
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = before - len(out)

        # 5. Drop columns that are almost entirely empty.
        missing_frac = out.isna().mean()
        dropped = missing_frac[missing_frac > self.cfg.max_missing_fraction].index.tolist()
        out = out.drop(columns=dropped)
        report["columns_dropped_high_missing"] = dropped

        # 6. Impute remaining missing values.
        report["missing_filled"] = self._impute(out)

        # 7. Flag outliers (does not delete — analytics decide what to do).
        report["outliers"] = self._flag_outliers(
            out, [c for c in profile.numeric if c in out.columns]
        )

        self.report = report
        return out

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _to_boolean(series: pd.Series) -> pd.Series:
        mapping = {
            "true": True, "false": False, "yes": True, "no": False,
            "y": True, "n": False, "t": True, "f": False, "1": True, "0": False,
        }
        if pd.api.types.is_bool_dtype(series):
            return series
        return series.astype(str).str.strip().str.lower().map(mapping)

    def _impute(self, df: pd.DataFrame) -> Dict[str, int]:
        filled: Dict[str, int] = {}
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            filled[col] = n_missing
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        return filled

    def _flag_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                counts[col] = 0
                continue
            low = q1 - self.cfg.outlier_iqr_multiplier * iqr
            high = q3 + self.cfg.outlier_iqr_multiplier * iqr
            counts[col] = int(((s < low) | (s > high)).sum())
        return counts


def encode_features(
    df: pd.DataFrame, feature_cols: List[str], max_cardinality: int
) -> pd.DataFrame:
    """One-hot/label encode categorical features for modelling.

    High-cardinality categoricals are label-encoded to avoid feature explosion.
    """
    encoded = df[feature_cols].copy()
    for col in encoded.columns:
        if pd.api.types.is_bool_dtype(encoded[col]):
            encoded[col] = encoded[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(encoded[col]):
            if encoded[col].nunique() <= max_cardinality:
                dummies = pd.get_dummies(encoded[col], prefix=col, dummy_na=False)
                encoded = pd.concat([encoded.drop(columns=[col]), dummies], axis=1)
            else:
                encoded[col] = encoded[col].astype("category").cat.codes
    return encoded.astype(float)
