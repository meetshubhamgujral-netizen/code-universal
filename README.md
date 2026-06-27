# 📊 Universal AI-Powered Data Analytics Dashboard

A production-ready Streamlit application that analyzes **any** structured dataset
(CSV or Excel) with **zero configuration**. Upload your data and the app
automatically detects column types, cleans the data, and generates descriptive,
diagnostic and predictive analytics, interactive Plotly visualizations, and a
**Gemini-powered AI chatbot** that answers questions grounded in your data.

It works across domains — insurance, healthcare, banking, finance, sales, HR,
retail, marketing, manufacturing, education, customer analytics, logistics and
anything else with rows and columns. No dataset-specific logic is hard-coded.

---

## ✨ Features

| Area | What you get |
|------|--------------|
| **Universal ingestion** | Auto-detects delimiters, reads CSV & Excel, profiles every column (numeric / categorical / datetime / boolean / identifier) and proposes target variables |
| **Auto preprocessing** | Type coercion, missing-value imputation, duplicate removal, outlier flagging, encoding — all fault-tolerant |
| **Descriptive analytics** | Summary, column info, descriptive statistics, correlation matrix, frequency tables, data-quality score |
| **Diagnostic analytics** | Outliers, strong correlations, key drivers of a target, time trends, AI-style written findings |
| **Predictive analytics (AutoML)** | Auto-detects classification vs regression, trains & cross-validates up to 8 models (incl. XGBoost), compares them and recommends the best with ROC, confusion matrix and feature importance |
| **Interactive visuals** | Histograms, box/violin, scatter, bubble, bar/pie/donut, heatmap, treemap, sunburst, pair plot, parallel coordinates, radar, time series — all zoomable, hoverable, exportable |
| **Gemini chatbot** | Ask questions in plain English; answers use real dataset statistics, metadata and model results as context; conversation history is kept per session |
| **Polished UI** | Sidebar navigation, KPI cards, dark/light theme, expandable sections, responsive layout |

---

## 🗂 Project structure

```
ai-analytics-dashboard/
├── app.py                     # Streamlit entry point & UI
├── config.py                  # Settings, thresholds, theme, API-key resolution
├── utils.py                   # Formatting & helper functions
├── preprocessing.py           # Profiling + cleaning (DataProfiler/DataPreprocessor)
├── analytics.py               # Descriptive & diagnostic analytics
├── visualization.py           # Plotly chart factory
├── machine_learning.py        # AutoML engine
├── chatbot.py                 # Gemini integration
├── data/
│   ├── generate_sample.py     # Creates a demo churn dataset
│   └── sample_customer_churn.csv
├── .streamlit/
│   ├── config.toml            # Theme & server settings
│   └── secrets.toml.example   # Template for your API key
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick start (local)

```bash
# 1. Clone & install
git clone <your-repo-url>
cd ai-analytics-dashboard
pip install -r requirements.txt

# 2. (Optional) add your Gemini API key for the chatbot
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   then edit secrets.toml and paste your key
#   — or export it instead:
export GEMINI_API_KEY="your-key"

# 3. Run
streamlit run app.py
```

Open the local URL Streamlit prints, then upload `data/sample_customer_churn.csv`
to see the full dashboard immediately.

> The dashboard works **without** a Gemini key — every chart, statistic and
> model still runs. Only the AI chatbot needs the key.

---

## 🔑 Getting a Gemini API key

1. Visit **https://aistudio.google.com/app/apikey**.
2. Create a key and copy it.
3. Provide it via **either**:
   - environment variable `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), **or**
   - `.streamlit/secrets.toml`.

The default model is `gemini-2.5-flash`. Override with the `GEMINI_MODEL`
environment variable (e.g. `gemini-3.5-flash`).

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to **https://share.streamlit.io** → **New app** and point it at `app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

No secrets are stored in the repo — `.streamlit/secrets.toml` is git-ignored.

---

## 🧠 How it works

1. **Profiling** (`DataProfiler`) classifies each column using cardinality,
   dtype and name heuristics — never dataset-specific rules.
2. **Cleaning** (`DataPreprocessor`) imputes, de-duplicates, coerces types and
   flags outliers, returning a clean frame plus a transparent report.
3. **Analytics** (`DescriptiveAnalytics`, `DiagnosticAnalytics`) compute stats,
   a data-quality score, key drivers and findings.
4. **AutoML** (`AutoML`) picks classification vs regression, trains a model
   suite, cross-validates, ranks and recommends the winner.
5. **Visualization** (`Visualizer`) renders interactive Plotly charts.
6. **Chatbot** (`GeminiChatbot`) packs the analysis into a compact context and
   queries Gemini so answers are grounded in your data.

Every layer is wrapped in defensive error handling: if one analysis can't run
on a given dataset, it is skipped and the rest of the dashboard still renders.

---

## ⚙️ Configuration

Tune behaviour in `config.py`:

- `DetectionConfig` — type-detection thresholds
- `PreprocessConfig` — missing-value / outlier / encoding limits
- `MLConfig` — test split, CV folds, max training rows, cardinality caps
- `Theme` — colour palette

---

## 📝 Notes & limitations

- XGBoost is optional; if it isn't installed the other models still run.
- Very large files (100k+ rows) are sampled for model training to keep the UI
  responsive (configurable via `MLConfig.max_training_rows`).
- The chatbot summarizes the dataset rather than sending raw rows, keeping
  token usage low and avoiding leaking full records to the API.

## 📄 License

MIT — use it freely.
