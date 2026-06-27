"""
chatbot.py
==========
Gemini-powered chatbot that answers questions about the uploaded dataset.

Uses the modern ``google-genai`` SDK. The chatbot builds a compact context
string from the dataset profile, descriptive stats and (optionally) the AutoML
report, so answers are grounded in the actual data rather than generic.

If no API key is configured the class still works in a degraded "offline" mode
that returns a helpful message instead of crashing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from config import DEFAULT_GEMINI_MODEL, resolve_api_key


class GeminiChatbot:
    """Thin wrapper around the Gemini API with dataset-grounded context."""

    SYSTEM_INSTRUCTION = (
        "You are a senior data analyst embedded in an analytics dashboard. "
        "Answer questions strictly using the dataset context provided. Be concise, "
        "quantitative and practical. When you reference numbers, take them from the "
        "context. If the context does not contain the answer, say so plainly and "
        "suggest what analysis would reveal it. Use clear business language."
    )

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL):
        self.model_name = model
        self.api_key = resolve_api_key()
        self._client = None
        self._context = ""
        self.history: List[Dict[str, str]] = []
        if self.api_key:
            self._init_client()

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self._client is not None

    def _init_client(self) -> None:
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        except Exception:
            self._client = None

    # ------------------------------------------------------------------ #
    def set_context(
        self,
        df: pd.DataFrame,
        profile,
        descriptive: Optional[Dict] = None,
        ml_report=None,
    ) -> None:
        """Build a compact, token-efficient context block from the analysis."""
        parts: List[str] = []
        parts.append(
            f"DATASET SHAPE: {len(df)} rows x {df.shape[1]} columns."
        )
        parts.append("COLUMN TYPES: " + ", ".join(
            f"{k}={v}" for k, v in {
                "numeric": profile.numeric,
                "categorical": profile.categorical,
                "datetime": profile.datetime,
                "boolean": profile.boolean,
            }.items() if v
        ))

        if descriptive:
            q = descriptive.get("quality", {})
            if q:
                parts.append(
                    f"DATA QUALITY: score {q.get('score')}/100 ({q.get('band')}), "
                    f"completeness {q.get('completeness')}%."
                )
            summ = descriptive.get("summary", {})
            if summ:
                parts.append(
                    f"MISSING CELLS: {summ.get('missing_cells')} "
                    f"({summ.get('missing_pct')}%); duplicates: {summ.get('duplicate_rows')}."
                )

        # Numeric stats (kept short).
        num_cols = [c for c in profile.numeric if c in df.columns][:10]
        if num_cols:
            stats = df[num_cols].describe().T[["mean", "min", "max"]].round(3)
            parts.append("NUMERIC SUMMARY:\n" + stats.to_string())

        # Categorical top values.
        for col in [c for c in profile.categorical if c in df.columns][:5]:
            top = df[col].value_counts().head(3)
            parts.append(
                f"TOP '{col}': " + ", ".join(f"{i}={v}" for i, v in top.items())
            )

        # Correlations — find the strongest off-diagonal pairs.
        if len(num_cols) >= 2:
            corr = df[num_cols].corr(numeric_only=True)
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    pairs.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
            pairs = sorted(pairs, key=lambda t: abs(t[2]), reverse=True)[:5]
            if pairs:
                parts.append("TOP CORRELATIONS: " + "; ".join(
                    f"{a}~{b}={c:.2f}" for a, b, c in pairs
                ))

        if ml_report and getattr(ml_report, "task", "none") != "none":
            parts.append(f"ML TASK: {ml_report.task} on target '{ml_report.target}'.")
            if ml_report.best:
                best = next((r for r in ml_report.results if r.name == ml_report.best), None)
                if best:
                    metric_str = ", ".join(f"{k}={v:.3f}" for k, v in best.metrics.items())
                    parts.append(f"BEST MODEL: {ml_report.best} ({metric_str}).")
            if ml_report.feature_importance is not None:
                top_feats = ml_report.feature_importance.head(5)["feature"].tolist()
                parts.append("TOP FEATURES: " + ", ".join(top_feats))

        self._context = "\n".join(parts)

    # ------------------------------------------------------------------ #
    def ask(self, question: str, has_data: bool) -> str:
        """Answer a question. Records the exchange in ``self.history``."""
        self.history.append({"role": "user", "content": question})

        if not has_data:
            answer = (
                "Please upload a dataset first — then I can answer questions about "
                "its columns, statistics, correlations, anomalies and model results."
            )
            self.history.append({"role": "assistant", "content": answer})
            return answer

        if not self.available:
            answer = (
                "The Gemini API key isn't configured, so I can't generate AI answers. "
                "Set GEMINI_API_KEY as an environment variable or in Streamlit secrets. "
                "All charts and statistics on the dashboard still work without it."
            )
            self.history.append({"role": "assistant", "content": answer})
            return answer

        try:
            answer = self._generate(question)
        except Exception as exc:  # noqa: BLE001
            answer = f"Sorry, I couldn't reach Gemini ({str(exc)[:120]}). Please try again."
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _generate(self, question: str) -> str:
        # Include the last few turns for conversational continuity.
        recent = self.history[-6:]
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent[:-1])
        prompt = (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            f"=== DATASET CONTEXT ===\n{self._context}\n\n"
            f"=== RECENT CONVERSATION ===\n{convo}\n\n"
            f"=== QUESTION ===\n{question}\n\nAnswer:"
        )
        response = self._client.models.generate_content(
            model=self.model_name, contents=prompt
        )
        return (getattr(response, "text", "") or "").strip() or "(No response generated.)"

    def reset(self) -> None:
        self.history.clear()
