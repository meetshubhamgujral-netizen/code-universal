"""
app.py
======
Universal AI-Powered Data Analytics Dashboard.

Upload any CSV/Excel file and the app automatically profiles the data, cleans
it, and renders descriptive, diagnostic and predictive analytics plus an
interactive Gemini chatbot — with no dataset-specific code.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import os
import sys
import hashlib

# Ensure the app's own directory is importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from config import APP_ICON, APP_TITLE, MAX_UPLOAD_MB, THEME
from analytics import DescriptiveAnalytics, DiagnosticAnalytics
from chatbot import GeminiChatbot
from machine_learning import AutoML, comparison_table
from preprocessing import DataPreprocessor, DataProfiler
from visualization import (
    Visualizer, confusion_matrix_fig, feature_importance_fig, roc_curve_fig,
)
from utils import df_to_csv_bytes, human_format

# --------------------------------------------------------------------------- #
#  Page config & global styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide",
                   initial_sidebar_state="expanded")


def inject_css(dark: bool) -> None:
    bg = "#0E1117" if dark else "#FFFFFF"
    card = "#1A1D26" if dark else "#F5F6FA"
    text = "#FFFFFF" if dark else "#1A1D26"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg}; }}
        .kpi-card {{
            background: {card}; border-radius: 16px; padding: 18px 20px;
            border: 1px solid rgba(108,92,231,0.25);
            box-shadow: 0 4px 18px rgba(0,0,0,0.18); transition: transform .2s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-3px); }}
        .kpi-value {{ font-size: 30px; font-weight: 800; color: {THEME.primary};
                      line-height: 1.1; }}
        .kpi-label {{ font-size: 13px; color: {text}; opacity: .7;
                      text-transform: uppercase; letter-spacing: .5px; }}
        .hero {{
            background: linear-gradient(120deg,{THEME.primary},{THEME.secondary});
            border-radius: 22px; padding: 38px 40px; color: white;
            margin-bottom: 18px;
        }}
        .hero h1 {{ margin: 0; font-size: 38px; }}
        .badge {{ display:inline-block; padding:4px 12px; border-radius:20px;
                  background: rgba(255,255,255,.18); margin: 4px 6px 0 0;
                  font-size: 12px; }}
        .finding {{ border-left: 4px solid {THEME.primary}; background:{card};
                    padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; }}
        section[data-testid="stSidebar"] {{ background: {card}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "raw_df": None, "clean_df": None, "profile": None, "clean_profile": None,
        "prep_report": None, "ml_report": None, "chat": GeminiChatbot(),
        "messages": [], "dark": True, "filename": None, "file_hash": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


@st.cache_data(show_spinner=False)
def load_file(data: bytes, name: str) -> pd.DataFrame:
    """Read an uploaded CSV/Excel file into a dataframe (cached)."""
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def analyse(_df: pd.DataFrame, cache_key: str):
    """Profile + clean a dataframe.

    ``cache_key`` (a hash of the uploaded file's contents) is what Streamlit
    keys the cache on — the dataframe itself is passed with a leading underscore
    so it is *not* hashed (hashing a large frame on every rerun is slow). This
    means every distinct file produces a fresh analysis instead of reusing the
    first file's cached result.
    """
    profiler = DataProfiler()
    profile = profiler.profile(_df)
    pre = DataPreprocessor()
    clean = pre.clean(_df, profile)
    clean_profile = profiler.profile(clean)
    return profile, clean, clean_profile, pre.report


# --------------------------------------------------------------------------- #
#  UI sections
# --------------------------------------------------------------------------- #
def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{APP_ICON} {APP_TITLE}</h1>
          <p style="font-size:17px;max-width:760px;">
            Upload any structured dataset and get automatic descriptive,
            diagnostic and predictive analytics, interactive visualisations and
            an AI analyst — no configuration required.
          </p>
          <div>
            <span class="badge">Any domain</span>
            <span class="badge">Auto preprocessing</span>
            <span class="badge">AutoML</span>
            <span class="badge">Plotly visuals</span>
            <span class="badge">Gemini chatbot</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing() -> None:
    render_hero()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("How it works")
        st.markdown(
            "1. **Upload** a CSV or Excel file from the sidebar.\n"
            "2. The app **detects column types** and cleans the data.\n"
            "3. Explore **descriptive, diagnostic & predictive** analytics.\n"
            "4. Ask the **Gemini chatbot** questions in plain English."
        )
    with c2:
        st.subheader("Works with")
        st.markdown(
            "Insurance · Healthcare · Banking · Finance · Sales · HR · Retail · "
            "Marketing · Manufacturing · Education · Customer analytics · Logistics "
            "— and anything else with rows and columns."
        )
    st.info("⬅️ Upload a dataset in the sidebar to generate your dashboard.")


def kpi_row(items) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def section_overview(clean, profile, desc, report) -> None:
    st.header("📋 Overview")
    s = desc.summary()
    q = desc.data_quality()
    kpi_row([
        ("Rows", human_format(s["rows"])),
        ("Columns", s["columns"]),
        ("Missing %", f"{s['missing_pct']}%"),
        ("Duplicates removed", report.get("duplicates_removed", 0)),
        ("Quality score", f"{q['score']} · {q['band']}"),
    ])
    st.markdown("#### Column information")
    st.dataframe(desc.column_info(), width='stretch', hide_index=True)

    with st.expander("Cleaning report"):
        st.json({
            "duplicates_removed": report.get("duplicates_removed"),
            "columns_dropped_high_missing": report.get("columns_dropped_high_missing"),
            "missing_values_filled": report.get("missing_filled"),
            "outliers_flagged": report.get("outliers"),
        })
    st.download_button("⬇️ Download cleaned data (CSV)",
                       df_to_csv_bytes(clean), "cleaned_data.csv", "text/csv")


def section_descriptive(clean, profile, desc, viz) -> None:
    st.header("📈 Descriptive analytics")
    describe = desc.numeric_describe()
    if not describe.empty:
        st.markdown("#### Numeric summary")
        st.dataframe(describe, width='stretch')

    num = [c for c in profile.numeric if c in clean.columns]
    cat = [c for c in profile.categorical if c in clean.columns]

    if num:
        st.markdown("#### Distributions")
        sel = st.selectbox("Numeric column", num, key="dist_col")
        c1, c2 = st.columns(2)
        with c1:
            fig = viz.histogram(sel)
            if fig:
                st.plotly_chart(fig, width='stretch')
        with c2:
            fig = viz.box(sel)
            if fig:
                st.plotly_chart(fig, width='stretch')

    if cat:
        st.markdown("#### Category breakdowns")
        sel_c = st.selectbox("Categorical column", cat, key="cat_col")
        c1, c2 = st.columns(2)
        with c1:
            fig = viz.bar(sel_c)
            if fig:
                st.plotly_chart(fig, width='stretch')
        with c2:
            fig = viz.pie(sel_c, donut=True)
            if fig:
                st.plotly_chart(fig, width='stretch')

    fig = viz.correlation_heatmap()
    if fig:
        st.markdown("#### Correlation matrix")
        st.plotly_chart(fig, width='stretch')


def section_diagnostic(clean, profile, diag, viz, target) -> None:
    st.header("🔍 Diagnostic analytics")
    findings = diag.findings(target)
    if findings:
        st.markdown("#### Key findings")
        icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        for f in findings:
            st.markdown(
                f'<div class="finding">{icons.get(f.severity,"•")} '
                f'<b>{f.title}</b><br><span style="opacity:.8">{f.detail}</span></div>',
                unsafe_allow_html=True,
            )

    outliers = diag.outlier_summary()
    if outliers:
        st.markdown("#### Outliers (1.5×IQR)")
        st.dataframe(pd.DataFrame(outliers), width='stretch', hide_index=True)

    if target:
        drivers = diag.key_drivers(target)
        if not drivers.empty:
            st.markdown(f"#### Key drivers of `{target}`")
            import plotly.express as px
            fig = px.bar(drivers.iloc[::-1], x="strength", y="feature",
                         orientation="h", color="strength",
                         color_continuous_scale=THEME.continuous)
            fig.update_layout(template="plotly_dark" if st.session_state.dark else "plotly_white",
                              paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

    trends = diag.time_trends()
    if trends:
        st.markdown("#### Time trends")
        date_col = profile.datetime[0]
        for col in list(trends)[:2]:
            fig = viz.time_series(date_col, col)
            if fig:
                st.plotly_chart(fig, width='stretch')


def section_visuals(clean, profile, viz) -> None:
    st.header("🎨 Visual explorer")
    num = [c for c in profile.numeric if c in clean.columns]
    cat = [c for c in profile.categorical if c in clean.columns]

    chart = st.selectbox(
        "Chart type",
        ["Scatter", "Bubble", "Violin", "Treemap", "Sunburst",
         "Parallel coordinates", "Pair plot", "Radar"],
    )
    try:
        if chart == "Scatter" and len(num) >= 2:
            x = st.selectbox("X", num, key="sx")
            y = st.selectbox("Y", num, index=1, key="sy")
            color = st.selectbox("Colour", [None] + cat, key="sc")
            fig = viz.scatter(x, y, color)
        elif chart == "Bubble" and len(num) >= 3:
            x = st.selectbox("X", num, key="bx")
            y = st.selectbox("Y", num, index=1, key="by")
            size = st.selectbox("Size", num, index=2, key="bs")
            fig = viz.bubble(x, y, size, cat[0] if cat else None)
        elif chart == "Violin" and num:
            col = st.selectbox("Value", num, key="vv")
            grp = st.selectbox("Group", [None] + cat, key="vg")
            fig = viz.violin(col, grp)
        elif chart == "Treemap" and cat:
            path = st.multiselect("Hierarchy", cat, default=cat[:2], key="tm")
            fig = viz.treemap(path)
        elif chart == "Sunburst" and cat:
            path = st.multiselect("Hierarchy", cat, default=cat[:2], key="sb")
            fig = viz.sunburst(path)
        elif chart == "Parallel coordinates":
            fig = viz.parallel_coordinates()
        elif chart == "Pair plot":
            fig = viz.pair_matrix()
        elif chart == "Radar" and cat and len(num) >= 3:
            grp = st.selectbox("Group", cat, key="rg")
            fig = viz.radar(grp, num)
        else:
            fig = None
            st.info("Not enough suitable columns for this chart.")
        if fig:
            st.plotly_chart(fig, width='stretch')
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not render this chart: {exc}")


def section_predictive(clean, profile) -> None:
    st.header("🤖 Predictive analytics (AutoML)")
    candidates = [c for c in profile.target_candidates if c in clean.columns]
    if not candidates:
        st.info("No suitable target column detected for modelling.")
        return

    target = st.selectbox("Target column to predict", candidates, key="ml_target")
    if st.button("🚀 Train & compare models", type="primary"):
        with st.spinner("Training models…"):
            report = AutoML().run(clean, profile, target)
        st.session_state.ml_report = report

    report = st.session_state.ml_report
    if not report or report.target != target:
        st.caption("Pick a target and click **Train & compare models**.")
        return

    if report.task == "none" or not report.best:
        st.warning(report.note or "Modelling could not be completed.")
        return

    st.success(f"Detected a **{report.task}** task · best model: **{report.best}** "
               f"· {report.n_features} features used.")
    table = comparison_table(report)
    st.markdown("#### Model comparison")
    st.dataframe(table, width='stretch', hide_index=True)

    best = next((r for r in report.results if r.name == report.best), None)
    c1, c2 = st.columns(2)
    if best and best.confusion is not None:
        with c1:
            st.plotly_chart(
                confusion_matrix_fig(best.confusion, report.class_labels,
                                     st.session_state.dark),
                width='stretch',
            )
    roc_curves = {r.name: r.roc for r in report.results if r.roc}
    if roc_curves:
        with c2:
            st.plotly_chart(roc_curve_fig(roc_curves, st.session_state.dark),
                            width='stretch')
    if report.feature_importance is not None:
        st.plotly_chart(
            feature_importance_fig(report.feature_importance, st.session_state.dark),
            width='stretch',
        )

    metric = "f1" if report.task == "classification" else "r2"
    best_val = best.metrics.get(metric) if best else None
    st.markdown(
        f"**Recommendation:** *{report.best}* achieved the best "
        f"{'F1 score' if report.task=='classification' else 'R²'} "
        f"({best_val:.3f}) on the hold-out set, making it the recommended model "
        f"for predicting `{target}`."
    )


def section_chat(has_data: bool) -> None:
    st.header("💬 AI data analyst")
    chat: GeminiChatbot = st.session_state.chat
    if not chat.available:
        st.caption("⚠️ Gemini API key not set — configure GEMINI_API_KEY to enable AI answers.")

    if has_data:
        st.markdown("**Try:** "
                    "*Summarize this dataset* · *Which features matter most?* · "
                    "*What are the major risks?* · *Explain the correlations*")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about your data…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = chat.ask(prompt, has_data)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    inject_css(st.session_state.dark)

    with st.sidebar:
        st.markdown(f"### {APP_ICON} {APP_TITLE}")
        st.session_state.dark = st.toggle("🌙 Dark theme", value=st.session_state.dark)
        st.divider()
        upload = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"],
                                  help=f"Max ~{MAX_UPLOAD_MB} MB")
        if upload is not None:
            data = upload.getvalue()
            file_hash = hashlib.md5(data).hexdigest()
            # Reload whenever the file *content* changes (not just the name),
            # so re-uploading an edited file with the same name also refreshes.
            if file_hash != st.session_state.get("file_hash"):
                try:
                    df = load_file(data, upload.name)
                    st.session_state.raw_df = df
                    st.session_state.filename = upload.name
                    st.session_state.file_hash = file_hash
                    st.session_state.ml_report = None
                    st.session_state.messages = []
                    st.session_state.chat.reset()
                    st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read file: {exc}")

        page = None
        if st.session_state.raw_df is not None:
            st.divider()
            page = st.radio(
                "Navigate",
                ["Overview", "Descriptive", "Diagnostic", "Visual explorer",
                 "Predictive (AutoML)", "AI chatbot"],
            )
            if st.button("🔄 Clear dataset"):
                for k in ["raw_df", "clean_df", "profile", "ml_report",
                          "filename", "file_hash", "messages"]:
                    st.session_state[k] = None if k != "messages" else []
                st.session_state.chat.reset()
                st.rerun()

    # No data → landing page (+ chatbot still available).
    if st.session_state.raw_df is None:
        render_landing()
        st.divider()
        section_chat(has_data=False)
        return

    # Analyse (cached per uploaded file via its content hash).
    raw = st.session_state.raw_df
    cache_key = st.session_state.get("file_hash") or str(raw.shape)
    try:
        profile, clean, clean_profile, report = analyse(raw, cache_key)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
        return

    desc = DescriptiveAnalytics(clean, clean_profile)
    diag = DiagnosticAnalytics(clean, clean_profile)
    viz = Visualizer(clean, clean_profile, dark=st.session_state.dark)
    target = clean_profile.target_candidates[0] if clean_profile.target_candidates else None

    # Refresh chatbot context with the latest analysis.
    try:
        st.session_state.chat.set_context(
            clean, clean_profile,
            descriptive={"summary": desc.summary(), "quality": desc.data_quality()},
            ml_report=st.session_state.ml_report,
        )
    except Exception:
        pass

    if page == "Overview":
        section_overview(clean, clean_profile, desc, report)
    elif page == "Descriptive":
        section_descriptive(clean, clean_profile, desc, viz)
    elif page == "Diagnostic":
        section_diagnostic(clean, clean_profile, diag, viz, target)
    elif page == "Visual explorer":
        section_visuals(clean, clean_profile, viz)
    elif page == "Predictive (AutoML)":
        section_predictive(clean, clean_profile)
    elif page == "AI chatbot":
        section_chat(has_data=True)


if __name__ == "__main__":
    main()
