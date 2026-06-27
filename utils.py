"""
utils.py
========
Small, dependency-light helper functions shared across the application.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd


def human_format(num: float | int | None) -> str:
    """Format a number compactly, e.g. 1_500_000 -> '1.5M'."""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "—"
    try:
        num = float(num)
    except (TypeError, ValueError):
        return str(num)

    sign = "-" if num < 0 else ""
    num = abs(num)
    for unit in ["", "K", "M", "B", "T"]:
        if num < 1000:
            if unit == "" and float(num).is_integer():
                return f"{sign}{int(num)}"
            return f"{sign}{num:.1f}{unit}".rstrip("0").rstrip(".") + ("" if unit else "")
        num /= 1000.0
    return f"{sign}{num:.1f}P"


def pct(part: float, whole: float) -> float:
    """Safe percentage that never divides by zero."""
    return round((part / whole) * 100, 2) if whole else 0.0


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to shrink the memory footprint of large frames."""
    out = df.copy()
    for col in out.select_dtypes(include=["int", "int64", "int32"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["float", "float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return the dataframe memory footprint in megabytes."""
    return round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)


def safe_sample(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    """Return a random sample if the frame exceeds ``max_rows`` else the frame."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state)
    return df


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a dataframe to CSV bytes for download buttons."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def truncate_text(value: Any, length: int = 60) -> str:
    """Truncate long strings for display in tables/markdown."""
    text = str(value)
    return text if len(text) <= length else text[: length - 1] + "…"
