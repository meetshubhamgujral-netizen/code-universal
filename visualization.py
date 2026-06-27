"""
visualization.py
================
A Plotly chart factory. Every method returns a ``plotly.graph_objects.Figure``
and degrades gracefully (returns ``None``) when the data can't support the
chart, so the dashboard can simply skip empty charts.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import THEME
from preprocessing import DataProfile


class Visualizer:
    """Generate interactive Plotly figures for any dataset."""

    def __init__(self, df: pd.DataFrame, profile: DataProfile, dark: bool = True):
        self.df = df
        self.profile = profile
        self.template = "plotly_dark" if dark else "plotly_white"
        self.palette = THEME.qualitative

    def _style(self, fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            title=title,
            template=self.template,
            colorway=self.palette,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig

    # -- distributions ----------------------------------------------------- #
    def histogram(self, col: str, color: Optional[str] = None) -> Optional[go.Figure]:
        if col not in self.df.columns:
            return None
        fig = px.histogram(self.df, x=col, color=color, marginal="box", nbins=40)
        return self._style(fig, f"Distribution of {col}")

    def box(self, col: str, group: Optional[str] = None) -> Optional[go.Figure]:
        if col not in self.df.columns:
            return None
        fig = px.box(self.df, y=col, x=group, color=group, points="outliers")
        return self._style(fig, f"Box plot of {col}")

    def violin(self, col: str, group: Optional[str] = None) -> Optional[go.Figure]:
        if col not in self.df.columns:
            return None
        fig = px.violin(self.df, y=col, x=group, color=group, box=True, points=False)
        return self._style(fig, f"Violin plot of {col}")

    # -- relationships ----------------------------------------------------- #
    def scatter(
        self, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None
    ) -> Optional[go.Figure]:
        if x not in self.df.columns or y not in self.df.columns:
            return None
        # OLS trendline needs statsmodels; fall back silently if unavailable.
        trendline = None
        try:
            import statsmodels.api  # noqa: F401

            if self.df[[x, y]].dropna().shape[0] > 2:
                trendline = "ols"
        except Exception:
            trendline = None
        fig = px.scatter(self.df, x=x, y=y, color=color, size=size,
                         opacity=0.7, trendline=trendline)
        return self._style(fig, f"{y} vs {x}")

    def correlation_heatmap(self) -> Optional[go.Figure]:
        cols = [c for c in self.profile.numeric if c in self.df.columns]
        if len(cols) < 2:
            return None
        corr = self.df[cols].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto", color_continuous_scale=THEME.continuous,
            zmin=-1, zmax=1,
        )
        return self._style(fig, "Correlation matrix")

    def pair_matrix(self, max_cols: int = 4) -> Optional[go.Figure]:
        cols = [c for c in self.profile.numeric if c in self.df.columns][:max_cols]
        if len(cols) < 2:
            return None
        sample = self.df[cols].dropna()
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=0)
        fig = px.scatter_matrix(sample, dimensions=cols)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
        return self._style(fig, "Pair plot")

    def parallel_coordinates(self, max_cols: int = 6) -> Optional[go.Figure]:
        cols = [c for c in self.profile.numeric if c in self.df.columns][:max_cols]
        if len(cols) < 2:
            return None
        sample = self.df[cols].dropna()
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=0)
        fig = px.parallel_coordinates(sample, color=cols[0],
                                      color_continuous_scale=THEME.continuous)
        return self._style(fig, "Parallel coordinates")

    # -- composition ------------------------------------------------------- #
    def bar(self, col: str, top_n: int = 12) -> Optional[go.Figure]:
        if col not in self.df.columns:
            return None
        vc = self.df[col].value_counts().head(top_n)
        fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={"x": col, "y": "count"})
        return self._style(fig, f"Top {col} categories")

    def pie(self, col: str, top_n: int = 8, donut: bool = False) -> Optional[go.Figure]:
        if col not in self.df.columns:
            return None
        vc = self.df[col].value_counts().head(top_n)
        fig = px.pie(values=vc.values, names=vc.index.astype(str),
                     hole=0.45 if donut else 0)
        return self._style(fig, f"{'Donut' if donut else 'Pie'}: {col}")

    def treemap(self, path: List[str], value: Optional[str] = None) -> Optional[go.Figure]:
        path = [c for c in path if c in self.df.columns]
        if not path:
            return None
        try:
            fig = px.treemap(self.df, path=path, values=value)
        except Exception:
            return None
        return self._style(fig, "Treemap: " + " → ".join(path))

    def sunburst(self, path: List[str], value: Optional[str] = None) -> Optional[go.Figure]:
        path = [c for c in path if c in self.df.columns]
        if not path:
            return None
        try:
            fig = px.sunburst(self.df, path=path, values=value)
        except Exception:
            return None
        return self._style(fig, "Sunburst: " + " → ".join(path))

    # -- time series ------------------------------------------------------- #
    def time_series(self, date_col: str, value_col: str) -> Optional[go.Figure]:
        if date_col not in self.df.columns or value_col not in self.df.columns:
            return None
        df = self.df.dropna(subset=[date_col]).sort_values(date_col)
        if df.empty:
            return None
        agg = df.groupby(date_col)[value_col].mean().reset_index()
        fig = px.line(agg, x=date_col, y=value_col, markers=True)
        return self._style(fig, f"{value_col} over time")

    # -- advanced ---------------------------------------------------------- #
    def radar(self, group_col: str, metrics: List[str]) -> Optional[go.Figure]:
        metrics = [m for m in metrics if m in self.df.columns][:8]
        if group_col not in self.df.columns or len(metrics) < 3:
            return None
        grouped = self.df.groupby(group_col)[metrics].mean()
        # Normalise each metric 0–1 so the radar is comparable.
        norm = (grouped - grouped.min()) / (grouped.max() - grouped.min()).replace(0, 1)
        fig = go.Figure()
        for idx, (name, row) in enumerate(norm.head(6).iterrows()):
            fig.add_trace(
                go.Scatterpolar(
                    r=row.tolist() + [row.tolist()[0]],
                    theta=metrics + [metrics[0]],
                    fill="toself",
                    name=str(name),
                    line=dict(color=self.palette[idx % len(self.palette)]),
                )
            )
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        return self._style(fig, f"Radar by {group_col}")

    def bubble(self, x: str, y: str, size: str, color: Optional[str] = None) -> Optional[go.Figure]:
        for c in (x, y, size):
            if c not in self.df.columns:
                return None
        df = self.df.copy()
        df[size] = df[size].clip(lower=0)
        fig = px.scatter(df, x=x, y=y, size=size, color=color, opacity=0.6, size_max=40)
        return self._style(fig, f"Bubble: {y} vs {x}")


# --------------------------------------------------------------------------- #
#  ML-specific figures (kept here so the ML module stays framework-agnostic)
# --------------------------------------------------------------------------- #
def confusion_matrix_fig(matrix: np.ndarray, labels: List[str], dark: bool = True) -> go.Figure:
    fig = px.imshow(
        matrix, text_auto=True, color_continuous_scale=THEME.continuous,
        labels=dict(x="Predicted", y="Actual"), x=labels, y=labels,
    )
    fig.update_layout(
        title="Confusion matrix",
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def roc_curve_fig(curves: dict, dark: bool = True) -> go.Figure:
    """``curves`` maps model name -> (fpr, tpr, auc)."""
    fig = go.Figure()
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
                       line=dict(color=THEME.qualitative[i % len(THEME.qualitative)]))
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), showlegend=False))
    fig.update_layout(
        title="ROC curves", xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def feature_importance_fig(importances: pd.DataFrame, dark: bool = True) -> go.Figure:
    """``importances`` has columns ['feature', 'importance'] sorted descending."""
    data = importances.head(15).iloc[::-1]
    fig = px.bar(data, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=THEME.continuous)
    fig.update_layout(
        title="Feature importance",
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
    )
    return fig
