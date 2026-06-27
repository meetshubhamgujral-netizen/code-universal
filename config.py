"""
config.py
=========
Central configuration for the Universal AI Analytics Dashboard.

Holds constants, thresholds, theme palettes, ML settings and the Gemini API
key resolution logic. No secrets are hard-coded here: the key is read from the
environment or from Streamlit secrets at runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


# --------------------------------------------------------------------------- #
#  Gemini / LLM configuration
# --------------------------------------------------------------------------- #
# Default model. `gemini-2.5-flash` is a stable, fast and inexpensive choice.
# Override with the GEMINI_MODEL environment variable if you want a newer one.
DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Order in which we look for the API key.
GEMINI_ENV_KEYS: List[str] = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]


def resolve_api_key() -> str | None:
    """Resolve the Gemini API key from env vars first, then Streamlit secrets.

    Returns ``None`` if no key is configured so the UI can degrade gracefully.
    """
    for key in GEMINI_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value.strip()

    # Streamlit secrets are optional; importing lazily avoids a hard dependency
    # when the module is used outside of a Streamlit context (e.g. unit tests).
    try:
        import streamlit as st  # noqa: WPS433 (local import is intentional)

        for key in GEMINI_ENV_KEYS:
            if key in st.secrets:
                return str(st.secrets[key]).strip()
    except Exception:  # pragma: no cover - secrets not available
        pass
    return None


# --------------------------------------------------------------------------- #
#  Column-type detection thresholds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionConfig:
    """Heuristics used to classify columns without dataset-specific logic."""

    # A numeric column with <= this many unique values is treated as categorical.
    categorical_numeric_max_unique: int = 15
    # Ratio of unique values above which an object column is an identifier.
    identifier_unique_ratio: float = 0.9
    # A column is a good classification target if it has between these many
    # distinct classes.
    target_class_min: int = 2
    target_class_max: int = 20
    # Sample size used when inferring datetime columns (perf optimisation).
    datetime_inference_sample: int = 200
    # Fraction of sampled values that must parse as dates to call it a date col.
    datetime_parse_threshold: float = 0.7


# --------------------------------------------------------------------------- #
#  Preprocessing configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreprocessConfig:
    """Settings that control automatic cleaning."""

    # Drop columns with a missing fraction above this value.
    max_missing_fraction: float = 0.6
    # IQR multiplier for outlier flagging.
    outlier_iqr_multiplier: float = 1.5
    # Cap categorical cardinality used for one-hot encoding.
    max_onehot_cardinality: int = 30


# --------------------------------------------------------------------------- #
#  Machine-learning configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MLConfig:
    """Settings for the AutoML engine."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    # Cap rows used for model training to keep the UI responsive on big data.
    max_training_rows: int = 50_000
    # Drop high-cardinality categoricals from the feature set.
    max_feature_cardinality: int = 50


# --------------------------------------------------------------------------- #
#  Theme / colour palettes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    """Colour palettes shared across the visualisation layer."""

    primary: str = "#6C5CE7"
    secondary: str = "#00CEC9"
    accent: str = "#FD79A8"
    success: str = "#00B894"
    warning: str = "#FDCB6E"
    danger: str = "#E17055"
    # A pleasant qualitative sequence for categorical charts.
    qualitative: List[str] = field(
        default_factory=lambda: [
            "#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E", "#00B894",
            "#E17055", "#0984E3", "#A29BFE", "#55EFC4", "#FAB1A0",
        ]
    )
    # Continuous scale used for heatmaps / correlation matrices.
    continuous: str = "Tealrose"


# Pre-built singletons imported elsewhere.
DETECTION = DetectionConfig()
PREPROCESS = PreprocessConfig()
ML = MLConfig()
THEME = Theme()

# Friendly labels for the data-quality score bands.
QUALITY_BANDS: Dict[str, tuple] = {
    "Excellent": (90, 101),
    "Good": (75, 90),
    "Fair": (60, 75),
    "Poor": (0, 60),
}

APP_TITLE = "Universal AI Analytics Dashboard"
APP_ICON = "📊"
MAX_UPLOAD_MB = 200
