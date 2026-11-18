"""
Time Series Forecasting Benchmark Dashboard (R26, app template v2: shared frame, sidebar pages, a business question per page)

Reads pre-computed results from data/forecasts/ETT/*.json.
No model weights or raw data required at runtime (ETT CSVs used only in Explore the data and Statistical tests).

Pages: Overview · Explore the data · Fair benchmark · Forecast gallery · Model inspector · Statistical tests ·
Original benchmark (not like-for-like) · RL model selector · AI Assistant (gpt-4o-mini retrieval; replay by default, the viewer's own key).
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ETT Forecasting Benchmark",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts" / "ETT"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ETT"

PHI = 1.618  # golden ratio

# Model family mapping
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))   # app modules (R26)
from meta import FAMILY_COLORS, FAMILY_MAP, METRIC_META, MODEL_DISPLAY, MODEL_META  # noqa: E402

# ---------------------------------------------------------------------------
# Custom CSS — professional typography, card styling, golden-ratio spacing
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar refinements */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* KPI metric cards — golden ratio aspect (~1.618 wide per unit height) */
[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #4C8BF5;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8B949E;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
}

/* Section headers */
h2, h3 {
    color: #E6EDF3;
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* Divider */
hr {
    border-color: #21262D;
    margin: 1.2rem 0;
}

/* Code / monospace (formula display) */
code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82em;
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 4px;
    padding: 0.1em 0.4em;
    color: #79C0FF;
}

/* Tab bar */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Info / context blocks */
.context-card {
    background: #161B22;
    border-left: 3px solid #4C8BF5;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #C9D1D9;
    line-height: 1.6;
}
.context-card strong {
    color: #E6EDF3;
}

/* Family badge */
.badge-stat  { background:#F5A62322; color:#F5A623; border:1px solid #F5A62366; border-radius:4px; padding:1px 8px; font-size:0.78rem; font-weight:600; }
.badge-ml    { background:#4C8BF522; color:#4C8BF5; border:1px solid #4C8BF566; border-radius:4px; padding:1px 8px; font-size:0.78rem; font-weight:600; }
.badge-dl    { background:#9B8FFF22; color:#9B8FFF; border:1px solid #9B8FFF66; border-radius:4px; padding:1px 8px; font-size:0.78rem; font-weight:600; }

/* Chat messages (Ask AI tab) */
.chat-user {
    background: #21262D;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.92rem;
    color: #E6EDF3;
}
.chat-ai {
    background: #0D2137;
    border-left: 3px solid #4C8BF5;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.92rem;
    color: #C9D1D9;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly template (consistent across all charts)
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(family="Inter, sans-serif", color="#C9D1D9", size=12),
        title=dict(font=dict(size=14, color="#E6EDF3"), x=0),
        xaxis=dict(
            gridcolor="#21262D", gridwidth=1,
            linecolor="#30363D", tickcolor="#30363D",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="#21262D", gridwidth=1,
            linecolor="#30363D", tickcolor="#30363D",
            tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor="#161B22", bordercolor="#30363D", borderwidth=1,
            font=dict(size=11), orientation="h",
        ),
        margin=dict(t=50, b=50, l=60, r=20),
        colorway=["#4C8BF5", "#F5A623", "#9B8FFF", "#00C48C", "#FF6B6B",
                  "#79C0FF", "#FFB946", "#D2A8FF", "#56D364", "#FFA198"],
    )
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_results() -> dict:
    """Load all model result metadata. y_true/y_pred arrays are excluded — use load_model_arrays()."""
    results = {}
    _skip_suffixes = ("_feature_importance", "_shap")
    for path in sorted(FORECAST_DIR.glob("*.json")):
        if any(path.stem.endswith(s) for s in _skip_suffixes):
            continue
        with open(path) as f:
            data = json.load(f)
        key = path.stem
        model_key = data.get("model", key.split("_h")[0].split("_m")[0])
        # Strip large arrays from the persistent cache; load on demand via load_model_arrays()
        slim = {k: v for k, v in data.items() if k not in ("y_true", "y_pred")}
        results[key] = {**slim, "_model_key": model_key}
    return results


@st.cache_data
def load_model_arrays(run_key: str) -> dict[str, list]:
    """Load y_true and y_pred for one model from its JSON file on demand."""
    path = FORECAST_DIR / f"{run_key}.json"
    if not path.exists():
        return {"y_true": [], "y_pred": []}
    with open(path) as f:
        data = json.load(f)
    return {"y_true": data.get("y_true", []), "y_pred": data.get("y_pred", [])}


@st.cache_data
def load_ett_data(variant: str = "h1") -> pd.DataFrame | None:
    path = RAW_DIR / f"ETT{variant}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def _compute_acf_pacf(ot_tuple: tuple, max_lag: int = 72) -> tuple[list, list, float]:
    """Compute ACF and PACF via Yule-Walker. Cached to avoid recomputing on every tab visit."""
    from numpy.linalg import solve as _np_solve
    ot = np.array(ot_tuple)
    ot_c = ot - ot.mean()
    acf_vals = [1.0]
    for lag in range(1, max_lag + 1):
        acf_vals.append(float(np.corrcoef(ot_c[lag:], ot_c[:-lag])[0, 1]))
    _acf = np.array(acf_vals)
    _phi = {1: [_acf[1]]}
    for k in range(2, min(max_lag + 1, 49)):
        R = np.array([[_acf[abs(i - j)] for j in range(k)] for i in range(k)])
        try:
            _phi[k] = _np_solve(R, _acf[1:k + 1]).tolist()
        except Exception:
            _phi[k] = [0.0] * k
    pacf_vals = [1.0] + [_phi[k][-1] if k in _phi else 0.0 for k in range(1, min(max_lag + 1, 49))]
    conf_bound = 1.96 / np.sqrt(len(ot))
    gc.collect()
    return acf_vals, pacf_vals, conf_bound


def get_family(model_key: str) -> str:
    return FAMILY_MAP.get(model_key, "Unknown")


def get_color(model_key: str) -> str:
    return FAMILY_COLORS.get(get_family(model_key), "#8B949E")


def display_name(run_key: str, model_key: str) -> str:
    base = MODEL_DISPLAY.get(model_key, model_key.replace("_", " ").title())
    if "_multivariate" in run_key:
        base += " (MV)"
    if "_tuned" in run_key:
        base += " (Tuned)"
    return base


# ---------------------------------------------------------------------------
# Build metrics DataFrame
# ---------------------------------------------------------------------------

results = load_results()

if not results:
    st.error(
        "No forecast results found in `data/forecasts/ETT/`. "
        "Run `make train-all` first."
    )
    st.stop()

rows = []
for run_key, data in results.items():
    model_key = data.get("_model_key", run_key)
    row = {
        "run_key": run_key,
        "Model": display_name(run_key, model_key),
        "Family": get_family(model_key),
        "_color": get_color(model_key),
    }
    row.update(data.get("metrics", {}))
    rows.append(row)

metrics_df = pd.DataFrame(rows).set_index("run_key")
available_metrics = [m for m in ["RMSE", "MAE", "MAPE", "SMAPE", "MASE", "R2"]
                     if m in metrics_df.columns]
best_model_key = metrics_df["RMSE"].idxmin() if "RMSE" in metrics_df.columns else None

# ---------------------------------------------------------------------------
# Pages and sidebar (shared app frame, R26: brand, pages, at-a-glance stats, filters, links)
# ---------------------------------------------------------------------------

import frame as F  # noqa: E402  (shared app frame, identical in every portfolio app)
import showcase  # noqa: E402

F.setup(st, accent="#79C0FF")
PAGES = [
    F.Page("overview", "Overview", ":material/home:", "The results, the brief and where to start."),
    F.Page("explore", "Explore the data", ":material/query_stats:",
           "The raw ETT readings: oil temperature and six load columns, their daily and weekly patterns and how they move together."),
    F.Page("fair", "Fair benchmark", ":material/balance:",
           "Every model forecasts the next 24 hours from the same history at 145 daily start points; ranked by MASE (ADR-008)."),
    F.Page("gallery", "Forecast gallery", ":material/show_chart:",
           "Forecasts against the real temperature, hour by hour, for the models picked in the sidebar (original single split)."),
    F.Page("inspector", "Model inspector", ":material/biotech:",
           "One model at a time: its errors over time, residuals, feature importance and a plain-English explanation."),
    F.Page("tests", "Statistical tests", ":material/functions:",
           "Stationarity, Granger causality, autocorrelation and Diebold–Mariano tests: which differences are real."),
    F.Page("original", "Original benchmark", ":material/history:",
           "The first single-split table, kept for the record: machine-learning models saw the true previous values (not like-for-like)."),
    F.Page("rl", "RL model selector", ":material/casino:",
           "A contextual bandit that picks a model for each 24-hour window, against always using one model and a perfect daily pick."),
    F.Page("agent", "AI Assistant", ":material/smart_toy:",
           "Ask about the models, the fair benchmark or the data: recorded answers replay for free with cited sources; your own key runs it live."),
]
F.keep(st, ["ask_byo_key", "ask_chat_mode"])          # the viewer's key (session only) also serves the inspector's Explain button
page = F.navigate(st, PAGES, brand="Forecasting benchmark", tagline="15 methods, one fair 24-hour test", icon="〜")
_h = showcase.headline()
F.sidebar_stats(st, [(f"{_h['mase_f']:.3f}", "Chronos-2 MASE, zero-shot"), (f"{_h['mase_t']:.3f}", "Holt-Winters MASE, best trained"),
                     (f"{_h['origins']}", "daily test forecasts"), (f"{_h['methods']}", "methods on the fair test")])

with st.sidebar:
    F.sidebar_label(st, "Filters (original-split pages)")

    # -- Filter: Model Families --
    with st.expander("Model Families", expanded=True):
        st.caption("Select which algorithm families to include.")
        show_statistical = st.checkbox("Statistical", value=True)
        show_ml = st.checkbox("Machine Learning", value=True)
        show_dl = st.checkbox("Deep Learning", value=True)

    # -- Filter: Individual Models --
    all_run_keys = list(results.keys())
    family_filter = set()
    if show_statistical:
        family_filter.add("Statistical")
    if show_ml:
        family_filter.add("ML")
    if show_dl:
        family_filter.add("Deep Learning")

    eligible_keys = [
        k for k in all_run_keys
        if get_family(results[k].get("_model_key", k)) in family_filter
    ]

    with st.expander("Individual Models", expanded=False):
        st.caption("Fine-tune which models appear in charts.")
        selected_keys = []
        for k in eligible_keys:
            mk = results[k].get("_model_key", k)
            label = display_name(k, mk)
            fam = get_family(mk)
            badge_class = "badge-stat" if fam == "Statistical" else ("badge-ml" if fam == "ML" else "badge-dl")
            checked = st.checkbox(
                label,
                value=True,
                key=f"chk_{k}",
            )
            if checked:
                selected_keys.append(k)

    if not selected_keys:
        selected_keys = eligible_keys

    st.divider()

    # -- Primary metric --
    with st.expander("Display Options", expanded=True):
        selected_metric = st.selectbox(
            "Primary ranking metric",
            available_metrics,
            index=0,
            help="This metric is used for the bar chart ranking and KPI highlight.",
        )
        n_steps = st.slider(
            "Forecast steps to display",
            min_value=48,
            max_value=1000,
            value=240,
            step=24,
            help="Number of test-set steps shown in the Forecast Gallery. Shorter windows load faster.",
        )

    st.divider()

    # -- Dataset info --
    with st.expander("Dataset Information", expanded=False):
        st.markdown("""
        **ETT (Electricity Transformer Temperature)**

        - Source: [ETDataset](https://github.com/zhouhaoyi/ETDataset) (MIT)
        - 4 variants: ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-min)
        - Target: **OT** — transformer oil temperature (°C)
        - Split: 70/10/20 chronological
        - Horizon: 24 steps

        > OT reflects thermal stress on the transformer, driven by aggregate electrical load.
        """)


# ---------------------------------------------------------------------------
# Filter DataFrames
# ---------------------------------------------------------------------------

sel_df = metrics_df.loc[metrics_df.index.isin(selected_keys)].copy()
F.sidebar_footer(st, {"Results page": showcase.PAGES, "Walkthrough": showcase.PAGES + "build.html", "Source": showcase.REPO},
                 "Data: ETT (Electricity Transformer Temperature), ETTh1, Zhou et al. 2021, MIT licence.")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

import agent as _agent_engine  # noqa: E402  (shared chat engine: key redaction)
import ask_ai  # noqa: E402
import chat  # noqa: E402

if page.key == "overview":                           # R22, R26 item 7: results and the brief on the first screen
    showcase.render(st)
    F.use_cards(st, PAGES)


# ===========================================================================
# TAB 0 — Fair rolling benchmark (ADR-008): same history for every model
# ===========================================================================

ROLLING_DIR = FORECAST_DIR / "rolling"
FAIR_NAMES = {"chronos2": "Chronos-2", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "holt_winters": "Holt-Winters",
              "nbeats": "N-BEATS", "random_forest": "Random Forest", "seasonal_naive_24": "Same hour yesterday", "arima": "ARIMA",
              "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM", "prophet": "Prophet",
              "seasonal_naive_168": "Same hour last week", "lstm": "LSTM"}
FAIR_KIND = {"foundation": ("Zero-shot", "#0f6e74"), "trained": ("Trained", "#b06a1f"), "baseline": ("Simple rule", "#8a8f96")}


@st.cache_data
def load_fair() -> tuple[dict, dict, dict]:
    summary = json.loads((ROLLING_DIR / "summary.json").read_text())
    ch = json.loads((ROLLING_DIR / "chronos2.json").read_text())["test"]
    hw = json.loads((ROLLING_DIR / "holt_winters.json").read_text())["test"]
    return summary, ch, hw


if page.key == "fair":
    _fs = json.loads((ROLLING_DIR / "summary.json").read_text())
    _g = _fs["gate"]
    F.header(st, PAGES[2], "Which model should we run in production?",
             f"**{FAIR_NAMES.get(_g['foundation_champion'])}**, zero-shot: test MASE {_g['test_mase'][_g['foundation_champion']]:.3f}, "
             f"{_g['foundation_lower_by']:.0%} lower than the best trained model, {FAIR_NAMES.get(_g['trained_champion'])} "
             f"({_g['test_mase'][_g['trained_champion']]:.3f}); Diebold–Mariano p = {_g['dm_hln']['p']:.0e} over {_g['dm_hln']['n']} forecasts. "
             f"Verdict, set before any run: {_g['verdict']}.")
    st.markdown(
        '<div class="context-card">'
        "Every model forecasts the next 24 hours from earlier readings only, at 145 daily start points (Feb–Jun 2018). "
        "Three zero-shot foundation models run with no training on this data. The pass mark was set before any run (ADR-008)."
        "</div>",
        unsafe_allow_html=True,
    )
    if not (ROLLING_DIR / "summary.json").exists():
        st.info("Fair benchmark results not found.")
    else:
        fs, fch, fhw = load_fair()
        fm, fg = fs["models"], fs["gate"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Typical hourly miss, Chronos-2", f"{fm['chronos2']['test']['mae']:.2f} °C",
                  help=f"Best trained model, Holt-Winters: {fm['holt_winters']['test']['mae']:.2f} °C")
        k2.metric("Error vs best trained model", f"{100 * fg['foundation_lower_by']:.0f}% lower", help="Bar set in advance: 5%")
        k3.metric("Time per 24-hour forecast", f"{fs['latency_idle']['models']['chronos2']['p50_ms'] / 1000:.2f} s", help="Laptop CPU, no GPU")
        st.caption("MASE: average error ÷ the error of \"same hour yesterday\" on the training data. Below 1 beats it.")
        uni = sorted([k for k in fm if fm[k]["track"] == "univariate"], key=lambda k: fm[k]["test"]["mase"], reverse=True)
        fig = go.Figure(go.Bar(
            x=[fm[k]["test"]["mase"] for k in uni], y=[FAIR_NAMES[k] for k in uni], orientation="h",
            marker_color=[FAIR_KIND[fm[k]["group"]][1] for k in uni],
            customdata=[[FAIR_KIND[fm[k]["group"]][0], fm[k]["test"]["mae"], fm[k]["clean_tail"]["mase"]] for k in uni],
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Test MASE %{x:.3f}<br>Typical miss %{customdata[1]:.2f} °C<br>"
                          "Last 41 days %{customdata[2]:.3f}<extra></extra>"))
        fig.add_vline(x=1, line_dash="dash", line_color="#8a8f96", annotation_text="same hour yesterday")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Test MASE (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
        ett = load_ett_data("h1")
        if ett is not None:
            day = st.slider("Test day", 0, len(fch["origins"]) - 1, 72)
            o = fch["origins"][day]
            seg = ett.iloc[o:o + 24]
            fig2 = go.Figure()
            fig2.add_scatter(x=seg["date"], y=seg["OT"], name="Actual", line=dict(color="#E6EDF3", width=2.5))
            fig2.add_scatter(x=seg["date"], y=fch["y_pred"][day], name="Chronos-2 (zero-shot)", line=dict(color="#0f6e74", dash="dash"))
            fig2.add_scatter(x=seg["date"], y=fhw["y_pred"][day], name="Holt-Winters (best trained)", line=dict(color="#b06a1f", dash="dash"))
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Oil temperature (°C)", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            f"- **Pass mark met:** {100 * fg['foundation_lower_by']:.1f}% lower error (bar 5%), Diebold–Mariano p = {fg['dm_hln']['p']:.0e}, "
            f"and lower on the last 41 days ({fg['clean_tail_mase']['chronos2']:.3f} vs {fg['clean_tail_mase']['holt_winters']:.3f}).\n"
            "- **Why the original benchmark differs:** it gave ML the true past values at every hour; without them, ML errors grow 3.3–3.5×.\n"
            "- **Dropped:** Transformer and TFT (CPU retraining exceeded 2 hours; logged in ADR-008)."
        )

# ===========================================================================
# TAB 1 — Original benchmark (oracle lags, not like-for-like)
# ===========================================================================

if page.key == "original":
    _ovr = json.loads((ROLLING_DIR / "summary.json").read_text())["old_vs_rolling"]["random_forest"]
    F.header(st, PAGES[6], "Why did machine-learning models look best in the original benchmark?",
             f"They were given the true previous hours at every step (look-ahead). Random Forest scored RMSE {_ovr['old_rmse']:.2f} °C here "
             f"and {_ovr['rolling_rmse']:.2f} °C in the fair test. Use the Fair benchmark page to choose a model.")
    st.warning("Kept for the record. ML models here received the true past values at every hour (ADR-005); see the Fair benchmark page.")
    st.markdown(
        '<div class="context-card">'
        "All 11 models trained on <strong>ETTh1</strong> (17,420 hourly observations, 2016–2018). "
        "ML models use oracle lag features (ground-truth OT at previous steps as inputs), "
        "giving them an inherent advantage over statistical models that forecast 3,484 steps in one shot. "
        "Deep learning models ran for 5–30 epochs — undertrained relative to their full potential."
        "</div>",
        unsafe_allow_html=True,
    )

    # KPI row — golden ratio: 4 equal cards across the width
    best = best_model_key
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Models in benchmark", len(metrics_df))
    with kpi2:
        if best:
            bname = metrics_df.loc[best, "Model"]
            st.metric("Best model (RMSE)", bname)
    with kpi3:
        if best and "RMSE" in metrics_df.columns:
            v = metrics_df.loc[best, "RMSE"]
            st.metric("Best RMSE", f"{v:.4f} °C")
    with kpi4:
        if best and "R2" in metrics_df.columns:
            v = metrics_df.loc[best, "R2"]
            st.metric("Best R²", f"{v:.4f}")

    st.divider()

    # Main layout: bar chart (wider) + metric table (narrower) — golden ratio [φ, 1] ≈ [5, 3]
    col_chart, col_table = st.columns([PHI, 1])

    with col_chart:
        st.markdown(f"#### {METRIC_META.get(selected_metric, {}).get('full', selected_metric)} by Model")

        # Context for selected metric
        if selected_metric in METRIC_META:
            m = METRIC_META[selected_metric]
            st.markdown(
                f'<div class="context-card">'
                f"<strong>{m['full']}</strong> &mdash; "
                f"<code>{m['formula']}</code><br>"
                f"{m['description']}"
                f"</div>",
                unsafe_allow_html=True,
            )

        direction = METRIC_META.get(selected_metric, {}).get("direction", "lower")
        plot_df = sel_df[["Model", "Family", "_color", selected_metric]].dropna().copy()
        asc = direction == "lower"
        plot_df = plot_df.sort_values(selected_metric, ascending=asc)

        fig_bar = go.Figure()
        for _, row in plot_df.iterrows():
            is_best = (
                (asc and row[selected_metric] == plot_df[selected_metric].min()) or
                (not asc and row[selected_metric] == plot_df[selected_metric].max())
            )
            fig_bar.add_trace(go.Bar(
                x=[row["Model"]],
                y=[row[selected_metric]],
                name=row["Family"],
                marker_color=row["_color"],
                marker_line_color="#E6EDF3" if is_best else row["_color"],
                marker_line_width=2 if is_best else 0,
                opacity=1.0 if is_best else 0.8,
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['Model']}</b><br>"
                    f"{selected_metric}: {row[selected_metric]:.4f}<br>"
                    f"Family: {row['Family']}"
                    "<extra></extra>"
                ),
            ))

        # Family legend traces
        for fam, col in FAMILY_COLORS.items():
            fig_bar.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=col, symbol="square"),
                name=fam, showlegend=True,
            ))

        best_label = "Lower is better" if direction == "lower" else "Higher is better"
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            yaxis_title=f"{selected_metric} ({METRIC_META.get(selected_metric, {}).get('unit', '')})",
            xaxis_title="",
            height=int(280 * PHI),  # ≈ 453px — golden proportion
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            annotations=[dict(
                text=best_label,
                xref="paper", yref="paper", x=0, y=-0.12,
                showarrow=False, font=dict(size=10, color="#8B949E"),
            )],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.markdown("#### All Metrics")
        st.caption("Green = best in column. Bold border = currently selected metric.")

        def _fmt(val):
            try:
                return f"{float(val):.4f}"
            except (TypeError, ValueError):
                return str(val)

        display_cols = ["Model", "Family"] + available_metrics
        table_df = sel_df[display_cols].copy()

        styled = table_df.style
        for m in available_metrics:
            if m in table_df.columns:
                direc = METRIC_META.get(m, {}).get("direction", "lower")
                if direc == "lower":
                    styled = styled.highlight_min(subset=[m], color="#1A3A1A", props="color: #56D364; font-weight: 600")
                else:
                    styled = styled.highlight_max(subset=[m], color="#1A3A1A", props="color: #56D364; font-weight: 600")

        styled = styled.format({m: _fmt for m in available_metrics})
        st.dataframe(styled, use_container_width=True, height=int(280 * PHI))

    st.divider()

    # Radar chart — normalised multi-metric view
    st.markdown("#### Multi-Metric Radar — Normalised Performance")
    st.markdown(
        '<div class="context-card">'
        "Each axis is normalised so that the <strong>best possible score points outward</strong> "
        "(lower metrics like RMSE are inverted). A larger polygon means better overall performance. "
        "Note that ML models benefit from oracle lag features — their advantage on the radar "
        "reflects this experimental design choice."
        "</div>",
        unsafe_allow_html=True,
    )

    radar_metrics = [m for m in ["RMSE", "MAE", "SMAPE", "MASE", "R2"] if m in sel_df.columns]
    if len(radar_metrics) >= 3:
        radar_df = sel_df[["Model", "_color"] + radar_metrics].dropna()

        # Normalise: for lower-better, invert so larger = better
        norm_data: dict[str, list] = {}
        for m in radar_metrics:
            vals = radar_df[m].values.astype(float)
            direc = METRIC_META.get(m, {}).get("direction", "lower")
            if direc == "lower":
                mn, mx = vals.min(), vals.max()
                norm_data[m] = list(1 - (vals - mn) / (mx - mn + 1e-9))
            else:
                mn, mx = vals.min(), vals.max()
                norm_data[m] = list((vals - mn) / (mx - mn + 1e-9))

        fig_radar = go.Figure()
        for i, (_, row) in enumerate(radar_df.iterrows()):
            r_vals = [norm_data[m][i] for m in radar_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals + [r_vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                name=row["Model"],
                line=dict(color=row["_color"], width=2),
                fill="toself",
                fillcolor=row["_color"],
                opacity=0.15,
                hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.3f}<extra></extra>",
            ))
        fig_radar.update_layout(
            template=PLOTLY_TEMPLATE,
            polar=dict(
                bgcolor="#0D1117",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor="#21262D"),
                angularaxis=dict(tickfont=dict(size=11), gridcolor="#21262D"),
            ),
            height=380,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ===========================================================================
# TAB 2 — Forecast Gallery
# ===========================================================================

if page.key == "gallery":
    _best = sel_df["RMSE"].idxmin() if "RMSE" in sel_df.columns and not sel_df.empty else None
    F.header(st, PAGES[3], "Do the forecasts follow the real temperature hour by hour?",
             (f"Of the models picked, {sel_df.loc[_best, 'Model']} tracks closest here (RMSE {sel_df.loc[_best, 'RMSE']:.2f} °C), but these are "
              "original single-split forecasts in which machine-learning models saw the true previous values; choose with the Fair benchmark page.")
             if _best else "Pick at least one model in the sidebar.")
    st.markdown(
        '<div class="context-card">'
        "Test set predictions (last 20% of ETTh1 = <strong>3,484 hourly steps ≈ 145 days</strong>). "
        "Use the <em>Forecast steps to display</em> slider in the sidebar to zoom into a shorter window. "
        "All predictions begin from the same ground truth — differences in shape reveal each model's "
        "ability to track short-term variation, seasonal patterns, and long-term drift."
        "</div>",
        unsafe_allow_html=True,
    )

    if not selected_keys:
        st.info("Select at least one model in the sidebar.")
        st.stop()

    # Build overlay
    _first_arrays = load_model_arrays(selected_keys[0])
    y_true_full = _first_arrays.get("y_true", [])
    total_steps = len(y_true_full)
    steps = min(n_steps, total_steps)

    fig_gallery = go.Figure()

    # Ground truth — always white, thicker
    fig_gallery.add_trace(go.Scatter(
        x=list(range(steps)),
        y=y_true_full[:steps],
        name="Ground Truth (OT)",
        line=dict(color="#FFFFFF", width=2.5),
        hovertemplate="<b>Ground Truth</b><br>Step %{x}<br>OT: %{y:.2f} °C<extra></extra>",
    ))

    # Model predictions
    for run_key in selected_keys:
        data = results[run_key]
        _run_arrays = load_model_arrays(run_key)
        if not _run_arrays.get("y_pred"):
            continue
        y_pred = _run_arrays["y_pred"][:steps]
        model_key = data.get("_model_key", run_key)
        name = display_name(run_key, model_key)
        color = get_color(model_key)
        rmse = data.get("metrics", {}).get("RMSE", None)
        hover_suffix = f" | RMSE={rmse:.3f}" if rmse else ""

        fig_gallery.add_trace(go.Scatter(
            x=list(range(steps)),
            y=y_pred,
            name=name,
            line=dict(color=color, width=1.4),
            opacity=0.85,
            hovertemplate=f"<b>{name}{hover_suffix}</b><br>Step %{{x}}<br>Pred: %{{y:.2f}} °C<extra></extra>",
        ))

    # Shaded reference for near-zero OT (winter dip region)
    fig_gallery.add_hrect(y0=-5, y1=0, fillcolor="#FF6B6B", opacity=0.04,
                          annotation_text="Near-zero OT (MAPE excluded)", annotation_position="top left",
                          annotation_font_size=9, annotation_font_color="#FF6B6B")

    fig_gallery.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title=f"Test step (1 step = 1 h on ETTh1, showing {steps:,} of {total_steps:,})",
        yaxis_title="Oil Temperature (°C)",
        height=int(240 * PHI) + 120,
        hovermode="x unified",
        margin=dict(t=40, b=170, l=60, r=20),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.32,
            xanchor="left", x=0,
            font=dict(size=10),
            itemwidth=90,
        ),
    )
    st.plotly_chart(fig_gallery, use_container_width=True)

    # Family-level RMSE summary below the chart
    st.markdown("#### RMSE by Model Family")
    fam_col1, fam_col2, fam_col3 = st.columns(3)
    for col_widget, fam in zip([fam_col1, fam_col2, fam_col3], ["Statistical", "ML", "Deep Learning"]):
        fam_rows = sel_df[sel_df["Family"] == fam]
        if not fam_rows.empty and "RMSE" in fam_rows.columns:
            best_fam = fam_rows["RMSE"].min()
            best_name = fam_rows.loc[fam_rows["RMSE"].idxmin(), "Model"]
            badge_class = "badge-stat" if fam == "Statistical" else ("badge-ml" if fam == "ML" else "badge-dl")
            col_widget.markdown(
                f'<div class="context-card">'
                f'<span class="{badge_class}">{fam}</span><br><br>'
                f"Best: <strong>{best_name}</strong><br>"
                f"RMSE: <strong>{best_fam:.4f} °C</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ===========================================================================
# TAB 3 — Model Inspector
# ===========================================================================

if page.key == "inspector":
    F.header(st, PAGES[4], "How does one model behave, and where does it miss?",
             f"Choose one of the {len(selected_keys)} models with the control on the left: its errors over time, residuals and what drives it. "
             "Original single-split results; for ranking, use the Fair benchmark page.")

    # Golden ratio: inspector controls left [1], charts right [φ] ≈ [1, 1.618]
    ctrl_col, detail_col = st.columns([1, PHI])

    with ctrl_col:
        sel_options = {display_name(k, results[k].get("_model_key", k)): k for k in selected_keys}
        sel_label = st.selectbox("Select model", list(sel_options.keys()))
        sel_run_key = sel_options[sel_label]
        sel_data = results[sel_run_key]
        sel_model_key = sel_data.get("_model_key", sel_run_key)
        sel_meta = MODEL_META.get(sel_model_key, {})

        # Model info card
        fam = get_family(sel_model_key)
        badge_class = "badge-stat" if fam == "Statistical" else ("badge-ml" if fam == "ML" else "badge-dl")
        st.markdown(
            f'<div class="context-card">'
            f'<span class="{badge_class}">{fam}</span>'
            f"<h4 style='margin:0.6rem 0 0.3rem; color:#E6EDF3;'>{sel_meta.get('label', sel_label)}</h4>"
            f"<div style='font-size:0.82rem; color:#8B949E; margin-bottom:0.5rem;'>{sel_meta.get('algorithm','')}</div>"
            f"{sel_meta.get('description','')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if sel_meta.get("formula"):
            st.markdown(f"**Formula:** `{sel_meta['formula']}`")

        st.markdown("**Strengths**")
        st.caption(sel_meta.get("strengths", "—"))
        st.markdown("**Weaknesses**")
        st.caption(sel_meta.get("weaknesses", "—"))

        st.divider()
        st.markdown("**Metrics**")
        m_dict = sel_data.get("metrics", {})
        for mk, mv in m_dict.items():
            direction = METRIC_META.get(mk, {}).get("direction", "lower")
            best_val = metrics_df[mk].min() if direction == "lower" else metrics_df[mk].max()
            is_best = abs(mv - best_val) < 1e-9
            label = METRIC_META.get(mk, {}).get("full", mk)
            col_left, col_right = st.columns([3, 2])
            col_left.caption(label)
            color_style = "color:#56D364; font-weight:600;" if is_best else ""
            col_right.markdown(f"<span style='{color_style}'>{mv:.4f}</span>", unsafe_allow_html=True)

        # Overfitting diagnostic: train vs test RMSE gap
        train_m = sel_data.get("train_metrics", {})
        test_rmse = m_dict.get("RMSE")
        train_rmse = train_m.get("RMSE")
        st.divider()
        st.markdown("**Overfitting Diagnostic**")
        if train_rmse and test_rmse:
            gap_ratio = test_rmse / train_rmse
            if gap_ratio < 1.2:
                fit_label, fit_color, fit_note = "Well-fitted", "#56D364", "Train ≈ Test — generalises well."
            elif gap_ratio < 2.0:
                fit_label, fit_color, fit_note = "Moderate overfit", "#F5A623", "Some gap — regularisation may help."
            else:
                fit_label, fit_color, fit_note = "Overfitting", "#FF6B6B", "Large gap — model memorised training data."
            st.markdown(
                f'<div style="font-size:0.80rem;">'
                f"Train RMSE: <strong>{train_rmse:.4f}</strong><br>"
                f"Test RMSE: <strong>{test_rmse:.4f}</strong><br>"
                f"Ratio: <span style='color:{fit_color}; font-weight:600'>{gap_ratio:.2f}×</span> "
                f"→ <span style='color:{fit_color}'>{fit_label}</span><br>"
                f"<span style='color:#8B949E; font-size:0.75rem;'>{fit_note}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No train metrics — retrain to generate them.")

    with detail_col:
        _sel_arrays = load_model_arrays(sel_run_key)
        if _sel_arrays.get("y_true") and _sel_arrays.get("y_pred"):
            y_true = np.array(_sel_arrays["y_true"][:n_steps])
            y_pred = np.array(_sel_arrays["y_pred"][:n_steps])
            residuals = y_true - y_pred

            # Chart 1 & 2 side-by-side — actual vs predicted + residuals
            r1, r2 = st.columns(2)

            with r1:
                st.markdown("##### Actual vs Predicted")
                fig_av = go.Figure()
                x_ax = list(range(len(y_true)))
                fig_av.add_trace(go.Scatter(x=x_ax, y=y_true.tolist(), name="Actual",
                                            line=dict(color="#FFFFFF", width=2)))
                fig_av.add_trace(go.Scatter(x=x_ax, y=y_pred.tolist(), name="Predicted",
                                            line=dict(color=get_color(sel_model_key), width=1.8)))
                fig_av.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    xaxis_title="Step", yaxis_title="OT (°C)",
                    legend=dict(orientation="h", y=1.12),
                    margin=dict(t=30, b=40, l=50, r=10),
                )
                st.plotly_chart(fig_av, use_container_width=True)

            with r2:
                st.markdown("##### Residuals Over Time")
                fig_res = go.Figure()
                fig_res.add_hline(y=0, line_dash="dash", line_color="#30363D")
                fig_res.add_trace(go.Scatter(
                    x=list(range(len(residuals))),
                    y=residuals.tolist(),
                    mode="lines",
                    name="Residual",
                    line=dict(color="#FFB946", width=1.2),
                ))
                fig_res.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    xaxis_title="Step", yaxis_title="Error (°C)",
                    margin=dict(t=30, b=40, l=50, r=10),
                )
                st.plotly_chart(fig_res, use_container_width=True)

            # Chart 3 & 4 — error distribution + QQ-like scatter
            r3, r4 = st.columns(2)

            with r3:
                st.markdown("##### Residual Distribution")
                st.markdown(
                    '<div class="context-card" style="font-size:0.80rem;">'
                    "A well-calibrated model has residuals centred near zero with "
                    "small spread. Heavy tails indicate large isolated errors."
                    "</div>",
                    unsafe_allow_html=True,
                )
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=residuals.tolist(), nbinsx=50,
                    name="Residual",
                    marker_color=get_color(sel_model_key),
                    opacity=0.8,
                ))
                # Mean line
                mean_res = float(np.mean(residuals))
                fig_hist.add_vline(x=mean_res, line_dash="dash", line_color="#FF6B6B",
                                   annotation_text=f"Mean={mean_res:.3f}", annotation_font_size=9)
                fig_hist.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    xaxis_title="Error (°C)", yaxis_title="Count",
                    margin=dict(t=30, b=40, l=50, r=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with r4:
                st.markdown("##### Predicted vs Actual Scatter")
                st.markdown(
                    '<div class="context-card" style="font-size:0.80rem;">'
                    "Points along the diagonal (y=x) indicate perfect predictions. "
                    "Systematic deviations reveal bias patterns."
                    "</div>",
                    unsafe_allow_html=True,
                )
                # Sample for performance (max 800 points)
                idx = np.linspace(0, len(y_true) - 1, min(800, len(y_true)), dtype=int)
                fig_sc = go.Figure()
                rng = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
                fig_sc.add_trace(go.Scatter(
                    x=rng, y=rng, mode="lines",
                    line=dict(color="#30363D", dash="dash"), name="y = x (perfect)", showlegend=True,
                ))
                fig_sc.add_trace(go.Scatter(
                    x=y_true[idx].tolist(), y=y_pred[idx].tolist(),
                    mode="markers",
                    marker=dict(color=get_color(sel_model_key), size=3, opacity=0.5),
                    name="Prediction",
                    hovertemplate="Actual: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>",
                ))
                fig_sc.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    xaxis_title="Actual OT (°C)", yaxis_title="Predicted OT (°C)",
                    margin=dict(t=40, b=50, l=50, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                )
                st.plotly_chart(fig_sc, use_container_width=True)

            # ------------------------------------------------------------------
            # Feature Importance — reads pre-computed JSON (ML models only)
            # Generated by: PYTHONPATH=. poetry run python scripts/extract_feature_importances.py
            # ------------------------------------------------------------------
            st.divider()
            st.markdown("##### Feature Importance")

            ML_MODELS_SET = {"random_forest", "xgboost", "lightgbm", "catboost"}
            _fi_json_path = FORECAST_DIR / f"{sel_run_key}_feature_importance.json"

            # Known feature name mapping for index-based labels
            _known_feat_order = [
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
                "OT_rolling_mean_3", "OT_rolling_std_3", "OT_growth_rate", "OT_trend_3",
                "?", "?",
                "OT_lag_1", "OT_lag_2", "OT_lag_24",
            ]
            _load_feats_set = {"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"}

            if sel_model_key in ML_MODELS_SET:
                if _fi_json_path.exists():
                    with open(_fi_json_path) as _f:
                        _fi_data = json.load(_f)

                    _feat_names = _fi_data.get("feature_names", [])
                    _importances = _fi_data.get("importances", [])
                    n_feat = len(_importances)

                    # Resolve generic names using known position mapping
                    def _resolve_name(name: str, idx: int) -> str:
                        if not name.startswith("feature_"):
                            return name
                        if idx < len(_known_feat_order) and not _known_feat_order[idx].startswith("?"):
                            return _known_feat_order[idx]
                        return f"feature_{idx}"

                    _feat_names_resolved = [_resolve_name(n, i) for i, n in enumerate(_feat_names)]

                    st.markdown(
                        '<div class="context-card" style="font-size:0.82rem;">'
                        f"<strong>Mean Decrease in Impurity (MDI)</strong> for {MODEL_DISPLAY.get(sel_model_key, sel_model_key)} "
                        f"({n_feat} features). "
                        "MDI measures how much each feature reduces prediction error averaged across all trees. "
                        "Lag features dominate in ML time series models because they encode the most "
                        "recent OT value — a near-perfect predictor for 1-step-ahead tasks."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    _fi_df = pd.DataFrame({
                        "Feature": _feat_names_resolved,
                        "Importance": _importances,
                    }).sort_values("Importance", ascending=True)

                    def _fi_color(name: str) -> str:
                        if "lag" in name:
                            return "#4C8BF5"
                        if any(x in name for x in ["rolling", "growth", "trend"]):
                            return "#F5A623"
                        if name in _load_feats_set:
                            return "#9B8FFF"
                        return "#56D364"

                    fig_fi = go.Figure(go.Bar(
                        x=_fi_df["Importance"].tolist(),
                        y=_fi_df["Feature"].tolist(),
                        orientation="h",
                        marker_color=[_fi_color(n) for n in _fi_df["Feature"].tolist()],
                        hovertemplate="<b>%{y}</b><br>MDI: %{x:.6f}<extra></extra>",
                    ))
                    for _lname, _lcolor in [
                        ("OT Lags", "#4C8BF5"),
                        ("Rolling / trend", "#F5A623"),
                        ("Load covariates", "#9B8FFF"),
                        ("Cyclical time", "#56D364"),
                    ]:
                        fig_fi.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(size=9, color=_lcolor, symbol="square"),
                            name=_lname, showlegend=True,
                        ))
                    fig_fi.update_layout(
                        template=PLOTLY_TEMPLATE,
                        xaxis_title="Importance (MDI)",
                        height=max(300, n_feat * 24 + 100),
                        margin=dict(t=50, b=50, l=160, r=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

                else:
                    st.markdown(
                        '<div class="context-card" style="font-size:0.82rem;">'
                        f"Pre-computed importances not found for <strong>{sel_run_key}</strong>. "
                        "Run the extraction script to generate them:<br>"
                        "<code>PYTHONPATH=. poetry run python scripts/extract_feature_importances.py</code><br><br>"
                        "Note: LightGBM and XGBoost checkpoints may fail to load due to library version "
                        "mismatches — retrain with <code>make train-all</code> to resolve."
                        "</div>",
                        unsafe_allow_html=True,
                    )

            else:
                st.markdown(
                    '<div class="context-card" style="font-size:0.82rem;">'
                    f"<strong>{sel_meta.get('label', sel_label)}</strong> ({fam}) — "
                    "tabular feature importance does not apply. "
                    "Statistical models operate on the raw OT series directly. "
                    "Deep learning models distribute information across all sequence positions "
                    "through learned weights, not discrete feature contributions."
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------------------------
            # AI Explainability — LLM narrative (supplements feature importance)
            # ------------------------------------------------------------------
            st.divider()

            # SHAP values chart for ML models; GPT text for Statistical / DL
            _shap_json_path = FORECAST_DIR / f"{sel_run_key}_shap.json"

            if sel_model_key in ML_MODELS_SET and _shap_json_path.exists():
                st.markdown("##### SHAP Values")
                with open(_shap_json_path) as _f:
                    _shap_data = json.load(_f)

                _feat_names_shap = _shap_data["feature_names"]
                _mean_abs = _shap_data["mean_abs_shap"]
                _shap_sample = np.array(_shap_data["shap_sample"])     # (N, 15)
                _feat_sample = np.array(_shap_data["feature_sample"])  # (N, 15)

                st.markdown(
                    '<div class="context-card" style="font-size:0.82rem;">'
                    "<strong>SHAP (SHapley Additive exPlanations)</strong> — "
                    "each bar shows how much that feature pushes predictions away from the average. "
                    "Dot color shows the raw feature value (red = high, blue = low). "
                    "OT_lag_1 dominates because the most recent oil temperature is the strongest "
                    "1-step-ahead predictor."
                    "</div>",
                    unsafe_allow_html=True,
                )

                # Sort features by mean |SHAP|
                _order = np.argsort(_mean_abs)
                _sorted_names = [_feat_names_shap[i] for i in _order]
                _sorted_mean  = [_mean_abs[i] for i in _order]

                # Beeswarm-style: scatter each sample's SHAP value, coloured by feature value
                fig_shap = go.Figure()

                # Add mean |SHAP| bar as background reference
                fig_shap.add_trace(go.Bar(
                    x=_sorted_mean,
                    y=_sorted_names,
                    orientation="h",
                    marker_color="rgba(76,139,245,0.18)",
                    showlegend=False,
                    hoverinfo="skip",
                ))

                # Overlay dots (one per sample) coloured by feature value
                for _fi, _fname in zip(_order, _sorted_names):
                    _sv = _shap_sample[:, _fi]
                    _fv = _feat_sample[:, _fi]
                    # Normalise feature value 0→1 for color scale
                    _fv_norm = (_fv - _fv.min()) / (max(_fv.max() - _fv.min(), 1e-9))
                    # Bright diverging palette visible on dark background:
                    # low value → #60A5FA (sky blue), high value → #F87171 (coral red)
                    _colors = [
                        f"rgb({int(96 + 159*v)},{int(165 - 141*v)},{int(250 - 136*v)})"
                        for v in _fv_norm
                    ]
                    _y_jitter = [_fname] * len(_sv)
                    fig_shap.add_trace(go.Scatter(
                        x=_sv.tolist(),
                        y=_y_jitter,
                        mode="markers",
                        marker=dict(size=4, color=_colors, opacity=0.65),
                        showlegend=False,
                        hovertemplate=f"<b>{_fname}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
                    ))

                fig_shap.update_layout(
                    template=PLOTLY_TEMPLATE,
                    xaxis_title="SHAP value (impact on prediction, °C)",
                    xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                    height=max(320, len(_feat_names_shap) * 26 + 100),
                    margin=dict(t=20, b=50, l=140, r=20),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption(
                    f"Based on {_shap_data['n_explained']} test samples | "
                    f"{_shap_data['n_background']} background samples | TreeExplainer (interventional)"
                )

            else:
                # Statistical / DL models → GPT-4o mini text explanation
                st.markdown("##### AI Explanation")
                st.markdown(
                    '<div class="context-card" style="font-size:0.82rem;">'
                    "GPT-4o mini generates a plain-language diagnosis linking metrics, "
                    "residual statistics, and the algorithm's mechanics — contextualising "
                    "<em>why</em> this model achieves its RMSE and when to use it in production."
                    "</div>",
                    unsafe_allow_html=True,
                )
                if sel_model_key in ML_MODELS_SET:
                    st.caption(
                        f"SHAP not pre-computed for {sel_run_key}. "
                        "Run: `PYTHONPATH=. poetry run python scripts/extract_shap_values.py`"
                    )

            # AI text button — shown for non-ML models, or ML without SHAP
            _show_ai_btn = (sel_model_key not in ML_MODELS_SET) or not _shap_json_path.exists()

            # Gather AI key
            _ai_key = st.session_state.get("ask_byo_key") or None     # R26: only the viewer's own key, entered on the AI Assistant page (Live)

            if _show_ai_btn and not _ai_key:
                st.caption("AI explanations run on your own OpenAI key: open **AI Assistant**, choose Live and enter it; it stays in this session only.")

            if _show_ai_btn and _ai_key:
                explain_btn = st.button(
                    f"Explain {sel_meta.get('label', sel_label)} performance",
                    key=f"explain_btn_{sel_run_key}",
                    type="primary",
                )
                if explain_btn:
                    # Build context for the explanation prompt
                    m_dict_full = sel_data.get("metrics", {})
                    avg_rmse = float(metrics_df["RMSE"].mean()) if "RMSE" in metrics_df.columns else None
                    rank_rmse = int((metrics_df["RMSE"] <= m_dict_full.get("RMSE", 999)).sum()) if "RMSE" in metrics_df.columns else None
                    residual_mean = float(np.mean(residuals)) if len(residuals) > 0 else 0
                    residual_std = float(np.std(residuals)) if len(residuals) > 0 else 0
                    # Autocorrelation at lag 1 as a measure of residual structure
                    if len(residuals) > 2:
                        resid_autocorr = float(
                            np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                        )
                    else:
                        resid_autocorr = 0.0

                    avg_rmse_str = f"{avg_rmse:.4f}" if avg_rmse else "N/A"
                    rank_rmse_str = str(rank_rmse) if rank_rmse else "N/A"
                    autocorr_label = (
                        "structured residuals — model misses a pattern"
                        if abs(resid_autocorr) > 0.3
                        else "near-random residuals — well-calibrated"
                    )
                    metrics_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in m_dict_full.items())

                    explain_prompt = (
                        f"Model: {sel_meta.get('label', sel_label)} ({sel_meta.get('family', fam)})\n"
                        f"Dataset: ETTh1 — hourly oil temperature forecasting (17,420 rows)\n"
                        f"Metrics: {metrics_lines}\n"
                        f"RMSE rank: {rank_rmse_str}/{len(metrics_df)} (avg={avg_rmse_str})\n"
                        f"Residuals: bias={residual_mean:.4f}°C, std={residual_std:.4f}°C, "
                        f"lag-1 autocorr={resid_autocorr:.3f} ({autocorr_label})\n"
                        f"Strengths: {sel_meta.get('strengths', '')}\n"
                        f"Weaknesses: {sel_meta.get('weaknesses', '')}\n\n"
                        "Reply in exactly 3 bullet points using markdown (start each with '- **Label:**'):\n"
                        "- **RMSE:** why this score given the algorithm mechanics\n"
                        "- **Residuals:** what bias/std/autocorr reveal about failure modes\n"
                        "- **Production:** when to use or avoid this model\n"
                        "Be specific to the numbers above. No filler sentences."
                    )

                    st.markdown(f"**AI Analysis — {sel_meta.get('label', sel_label)}**")
                    try:
                        from openai import OpenAI as _OAI
                        _client = _OAI(api_key=_ai_key)
                        _stream = _client.chat.completions.create(
                            model="gpt-4o-mini",
                            max_tokens=500,
                            stream=True,
                            messages=[{"role": "user", "content": explain_prompt}],
                        )
                        st.write_stream(
                            chunk.choices[0].delta.content or ""
                            for chunk in _stream
                            if chunk.choices
                        )
                    except Exception as e:
                        st.error(f"OpenAI API error: {_agent_engine.redact(str(e), _ai_key)[:300]}")

        else:
            st.info("No predictions available for this model.")

# ===========================================================================
# TAB 4 — Data Explorer
# ===========================================================================

if page.key == "explore":
    _ett = load_ett_data("h1")
    if _ett is not None and "OT" in _ett.columns:
        _r24, _r168 = _ett["OT"].autocorr(24), _ett["OT"].autocorr(168)
        _ans = (f"{len(_ett):,} hourly readings of oil temperature (OT) and six load columns. OT one day apart correlates at "
                f"r = {_r24:.2f} and one week apart at r = {_r168:.2f}, so 'same hour yesterday' is a hard yardstick: MASE below 1 beats it.")
    else:
        _ans = "The ETT data file is not available in this deployment."
    F.header(st, PAGES[1], "What does the transformer's oil temperature look like, and what moves it?", _ans)
    st.markdown(
        '<div class="context-card">'
        "Visualise the raw ETT dataset: all 7 measurement columns and the OT target. "
        "Understanding the raw signal — its seasonality, trend, and covariate relationships — "
        "is essential context for interpreting the model benchmark."
        "</div>",
        unsafe_allow_html=True,
    )

    ett_variant = st.radio(
        "ETT variant", ["h1", "h2", "m1", "m2"],
        horizontal=True,
        help="h1/h2 = hourly (17,420 rows). m1/m2 = 15-minute (69,680 rows).",
    )
    ett_df = load_ett_data(ett_variant)

    if ett_df is None:
        st.warning(f"ETT{ett_variant}.csv not found in data/raw/ETT/.")
    else:
        n_rows = len(ett_df)
        col_def, col_stats = st.columns([PHI, 1])

        with col_def:
            # Time series plot with range selector
            st.markdown("#### Time Series — All Columns")
            st.markdown(
                '<div class="context-card" style="font-size:0.80rem;">'
                "<strong>HUFL/HULL</strong>: High-voltage useful/useless load (active/reactive) &bull; "
                "<strong>MUFL/MULL</strong>: Mid-voltage &bull; "
                "<strong>LUFL/LULL</strong>: Low-voltage &bull; "
                "<strong>OT</strong>: Oil temperature (target)"
                "</div>",
                unsafe_allow_html=True,
            )
            # Sample to 2000 points for speed
            step = max(1, n_rows // 2000)
            sample = ett_df.iloc[::step].copy()

            load_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
            col_colors = ["#4C8BF5", "#79C0FF", "#F5A623", "#FFB946", "#9B8FFF", "#D2A8FF"]

            fig_ts = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.55, 0.45],
                vertical_spacing=0.05,
            )
            for lc, lcolor in zip(load_cols, col_colors):
                fig_ts.add_trace(go.Scatter(
                    x=sample["date"], y=sample[lc],
                    name=lc, line=dict(color=lcolor, width=1.0), opacity=0.75,
                ), row=1, col=1)
            fig_ts.add_trace(go.Scatter(
                x=sample["date"], y=sample["OT"],
                name="OT (target)", line=dict(color="#00C48C", width=2.0),
            ), row=2, col=1)

            fig_ts.update_layout(
                template=PLOTLY_TEMPLATE,
                height=int(200 * PHI) + 140,
                xaxis2_title="Date",
                yaxis_title="Load (MW)",
                yaxis2_title="OT (°C)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=10),
                ),
                margin=dict(t=50, b=60, l=60, r=20),
            )
            fig_ts.update_xaxes(
                rangeslider=dict(visible=True, thickness=0.04),
                row=2, col=1
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        with col_stats:
            st.markdown("#### Summary Statistics")
            num_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
            desc = ett_df[num_cols].describe().round(2)
            st.dataframe(desc, use_container_width=True)
            st.caption(f"Rows: {n_rows:,} | Frequency: {'1 h' if ett_variant.startswith('h') else '15 min'}")

        st.divider()

        # Correlation heatmap
        st.markdown("#### Feature Correlation Matrix")
        _oc = ett_df[num_cols].corr()["OT"].drop("OT")
        _lc = ett_df[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]].corr().where(lambda m: ~np.eye(6, dtype=bool)).abs().max().max()
        st.markdown(
            '<div class="context-card" style="font-size:0.80rem;">'
            f"Pearson correlation between all 7 columns. Oil temperature moves only weakly with any single load column at the same hour "
            f"(strongest: {_oc.abs().idxmax()}, r = {_oc[_oc.abs().idxmax()]:.2f}), while some load columns move almost together "
            f"(up to |r| = {_lc:.2f}). This matrix measures same-hour links only, not delayed effects."
            "</div>",
            unsafe_allow_html=True,
        )
        corr = ett_df[num_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#FF6B6B"], [0.5, "#161B22"], [1, "#4C8BF5"]],
            zmin=-1, zmax=1,
            text=corr.round(2).values.astype(str),
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(t=20, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Feature definitions table
        st.markdown("#### Feature Definitions")
        feat_def_data = {
            "Column": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "Full Name": [
                "High Voltage Useful Load",
                "High Voltage Useless Load",
                "Mid Voltage Useful Load",
                "Mid Voltage Useless Load",
                "Low Voltage Useful Load",
                "Low Voltage Useless Load",
                "Oil Temperature",
            ],
            "Voltage Level": ["High", "High", "Mid", "Mid", "Low", "Low", "—"],
            "Load Type": ["Active (W)", "Reactive (VAR)", "Active (W)", "Reactive (VAR)", "Active (W)", "Reactive (VAR)", "—"],
            "Unit": ["MW", "MW", "MW", "MW", "MW", "MW", "°C"],
            "Role": ["Covariate", "Covariate", "Covariate", "Covariate", "Covariate", "Covariate", "Target (OT)"],
            "Notes": [
                "Real power drawn by high-voltage consumers",
                "Reactive power — phase lag losses at high voltage",
                "Real power at medium-voltage distribution level",
                "Reactive power at medium-voltage level",
                "Real power at low-voltage (residential) level",
                "Reactive power at low-voltage level",
                "Transformer core temperature — rises with load; forecasting target",
            ],
        }
        feat_def_df = pd.DataFrame(feat_def_data)
        st.dataframe(feat_def_df.set_index("Column"), use_container_width=True)
        st.markdown(
            '<div class="context-card" style="font-size:0.80rem;">'
            "<strong>Useful (active) load</strong> is real power consumed (kW/MW). "
            "<strong>Useless (reactive) load</strong> is the imaginary component of AC power — "
            "it does no useful work but heats conductors and stresses the transformer. "
            "The correlation matrix shows MUFL–HUFL ≈ 0.99 and MULL–HULL ≈ 0.93: high and mid-voltage "
            "useful loads are nearly identical, suggesting the same aggregate load measured at two points. "
            "OT's low correlation with all load columns (0.05–0.22) is because temperature "
            "accumulates thermally with a time lag — the Granger causality test (Statistical tests page) "
            "reveals the true temporal relationship."
            "</div>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 5 — Statistical Tests
# ===========================================================================

if page.key == "tests":
    _dm = json.loads((ROLLING_DIR / "summary.json").read_text())["gate"]["dm_hln"]
    F.header(st, PAGES[5], "Are the differences between models real, or noise?",
             f"The headline one is real: Chronos-2 vs Holt-Winters, Diebold–Mariano (HLN) p = {_dm['p']:.0e} over {_dm['n']} daily forecasts. "
             "Below: stationarity, Granger causality and autocorrelation of the target, and pairwise tests on the original split.")
    st.markdown(
        '<div class="context-card">'
        "Rigorous statistical analysis of the ETT dataset and benchmark results. "
        "Tests covered: (1) <strong>Stationarity</strong> — ADF and KPSS tests on OT; "
        "(2) <strong>Granger Causality</strong> — do load columns predict OT?; "
        "(3) <strong>ACF / PACF</strong> — autocorrelation structure of OT; "
        "(4) <strong>Diebold-Mariano</strong> — pairwise statistical significance of model differences."
        "</div>",
        unsafe_allow_html=True,
    )

    _ett_st = load_ett_data("h1")

    if _ett_st is None:
        st.warning("ETTh1.csv not found — download with `python scripts/download_data.py`.")
    else:
        ot_series = _ett_st["OT"].dropna().values

        # -----------------------------------------------------------------------
        # Section 1: Stationarity tests
        # -----------------------------------------------------------------------
        st.markdown("### 1. Stationarity of OT (Oil Temperature)")
        st.markdown(
            '<div class="context-card" style="font-size:0.82rem;">'
            "<strong>ADF (Augmented Dickey-Fuller)</strong>: H₀ = unit root (non-stationary). "
            "Reject H₀ (p &lt; 0.05) → OT is stationary. "
            "<strong>KPSS</strong>: H₀ = level-stationary. "
            "Fail to reject H₀ (p &gt; 0.05) → OT is stationary. "
            "If ADF rejects AND KPSS fails to reject, both agree the series is stationary — "
            "meaning statistical models can be applied without differencing."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            adf_result = adfuller(ot_series, autolag="AIC")
            adf_stat, adf_p, adf_lags = adf_result[0], adf_result[1], adf_result[2]
            adf_conclusion = "Stationary (reject H₀)" if adf_p < 0.05 else "Non-stationary (fail to reject H₀)"

            kpss_result = kpss(ot_series, regression="c", nlags="auto")
            kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
            kpss_conclusion = "Stationary (fail to reject H₀)" if kpss_p > 0.05 else "Non-stationary (reject H₀)"

            st_col1, st_col2 = st.columns(2)
            with st_col1:
                adf_color = "#56D364" if adf_p < 0.05 else "#FF6B6B"
                st.markdown(
                    f'<div class="context-card">'
                    f"<strong>ADF Test</strong><br>"
                    f"Statistic: {adf_stat:.4f}<br>"
                    f"p-value: <span style='color:{adf_color}; font-weight:600'>{adf_p:.4f}</span><br>"
                    f"Lags used: {adf_lags}<br>"
                    f"Conclusion: <strong>{adf_conclusion}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with st_col2:
                kpss_color = "#56D364" if kpss_p > 0.05 else "#FF6B6B"
                st.markdown(
                    f'<div class="context-card">'
                    f"<strong>KPSS Test</strong><br>"
                    f"Statistic: {kpss_stat:.4f}<br>"
                    f"p-value: <span style='color:{kpss_color}; font-weight:600'>{kpss_p:.4f}</span><br>"
                    f"(truncated at 0.01 / 0.10 by statsmodels)<br>"
                    f"Conclusion: <strong>{kpss_conclusion}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            # Interpretation card based on actual results
            adf_stationary = adf_p < 0.05
            kpss_stationary = kpss_p > 0.05
            if adf_stationary and kpss_stationary:
                stat_verdict = "✅ <strong>Both tests agree: OT is stationary.</strong>"
                stat_implication = (
                    "ARIMA and Holt-Winters can be applied directly without differencing. "
                    "ML models also benefit — lag features will capture autocorrelation reliably "
                    "without a drifting mean distorting the feature space."
                )
            elif not adf_stationary and not kpss_stationary:
                stat_verdict = "⚠️ <strong>Both tests agree: OT is non-stationary.</strong>"
                stat_implication = (
                    "A trend or drift is present. ARIMA should use d≥1 (differencing). "
                    "ML models need detrending or difference-based lag features to avoid spurious correlation."
                )
            elif adf_stationary and not kpss_stationary:
                stat_verdict = "⚠️ <strong>Conflicting results (ADF stationary, KPSS non-stationary).</strong>"
                stat_implication = (
                    "This is the most common pattern for long time series: the series is stationary around a slowly-changing mean "
                    "(trend-stationary). KPSS is more sensitive to structural shifts; ADF has lower power. "
                    "Treat OT as weakly stationary — ARIMA with d=0 or d=1 should both be tested. "
                    "ML models are robust to this ambiguity since they don't assume strict stationarity."
                )
            else:
                stat_verdict = "ℹ️ <strong>Conflicting results (ADF non-stationary, KPSS stationary).</strong>"
                stat_implication = "Rare case — may indicate a unit root with heteroskedasticity. Consider differencing as a precaution."

            st.markdown(
                f'<div class="context-card" style="font-size:0.82rem; border-left: 3px solid #4C8BF5;">'
                f"<strong>What this means for forecasting:</strong><br>"
                f"{stat_verdict}<br>{stat_implication}"
                f"</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Stationarity tests failed: {e}")

        st.divider()

        # -----------------------------------------------------------------------
        # Section 2: ACF / PACF
        # -----------------------------------------------------------------------
        st.markdown("### 2. Autocorrelation (ACF) and Partial Autocorrelation (PACF) of OT")
        st.markdown(
            '<div class="context-card" style="font-size:0.82rem;">'
            "<strong>ACF</strong> measures correlation between OT at time t and OT at time t-k. "
            "Strong peaks at lag 24 and 168 confirm daily and weekly seasonality. "
            "<strong>PACF</strong> shows the correlation after removing intermediate lags — "
            "PACF cutting off after lag p suggests an AR(p) model; slowly decaying PACF suggests MA."
            "</div>",
            unsafe_allow_html=True,
        )

        acf_vals, pacf_vals, conf_bound = _compute_acf_pacf(tuple(ot_series.tolist()))
        lags_ax = list(range(len(acf_vals)))

        fig_acf = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
        for col_idx, (vals, name) in enumerate([(acf_vals, "ACF"), (pacf_vals, "PACF")], 1):
            n = len(vals)
            lx = list(range(n))
            fig_acf.add_trace(go.Bar(x=lx, y=vals, name=name, marker_color="#4C8BF5", showlegend=False,
                                     hovertemplate=f"Lag %{{x}}: {name}=%{{y:.3f}}<extra></extra>"), row=1, col=col_idx)
            fig_acf.add_hline(y=conf_bound, line_dash="dash", line_color="#56D364", row=1, col=col_idx)
            fig_acf.add_hline(y=-conf_bound, line_dash="dash", line_color="#56D364", row=1, col=col_idx)
            fig_acf.add_hline(y=0, line_color="#30363D", row=1, col=col_idx)

        fig_acf.update_layout(
            template=PLOTLY_TEMPLATE, height=320,
            margin=dict(t=40, b=50, l=50, r=20),
            annotations=[dict(text="Green dashed = 95% confidence band (±1.96/√n)",
                              xref="paper", yref="paper", x=0, y=-0.15,
                              showarrow=False, font=dict(size=9, color="#8B949E"))],
        )
        fig_acf.update_xaxes(title_text="Lag (hours)")
        st.plotly_chart(fig_acf, use_container_width=True)

        # ACF/PACF interpretation based on computed values
        acf_lag24 = acf_vals[24] if len(acf_vals) > 24 else 0
        acf_lag48 = acf_vals[48] if len(acf_vals) > 48 else 0
        acf_lag1 = acf_vals[1] if len(acf_vals) > 1 else 0
        pacf_lag1 = pacf_vals[1] if len(pacf_vals) > 1 else 0

        acf_lines = []
        if abs(acf_lag1) > conf_bound:
            acf_lines.append(f"<strong>Lag 1 ACF = {acf_lag1:.3f}</strong> — strong short-term persistence. Past OT is highly predictive of the next hour.")
        if abs(acf_lag24) > conf_bound:
            acf_lines.append(f"<strong>Lag 24 ACF = {acf_lag24:.3f}</strong> — significant daily seasonality. Temperature 24 h ago predicts today's temperature.")
        if abs(acf_lag48) > conf_bound:
            acf_lines.append(f"<strong>Lag 48 ACF = {acf_lag48:.3f}</strong> — the daily cycle persists 48 h out.")

        # Check if ACF decays slowly (all values above conf bound through lag 24)
        slow_decay = all(abs(acf_vals[i]) > conf_bound for i in range(1, min(25, len(acf_vals))))
        if slow_decay:
            acf_lines.append("Slowly decaying ACF suggests <strong>strong autoregressive (AR) structure</strong> — this is why lag features matter so much for ML models.")

        pacf_lines = []
        if abs(pacf_lag1) > conf_bound * 3:
            pacf_lines.append(f"<strong>PACF cuts off sharply after lag 1</strong> ({pacf_lag1:.3f}) — most of the predictability is captured by a single AR(1) term. An AR(1) or ARIMA(1,0,0) is a reasonable baseline.")

        st.markdown(
            '<div class="context-card" style="font-size:0.82rem; border-left: 3px solid #4C8BF5;">'
            "<strong>Reading these charts:</strong><br>"
            + "<br>".join(acf_lines + pacf_lines or ["No significant lags detected above the confidence band."])
            + "<br><br><em>Practical implication:</em> The strong autocorrelation at lags 1 and 24 "
            "justifies the lag features (OT_lag_1, OT_lag_2, OT_lag_24) used in ML models. "
            "Models that can exploit this structure — ML with lag features, N-BEATS, TFT — "
            "should outperform pure autoregressive statistical models."
            "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # -----------------------------------------------------------------------
        # Section 3: Granger Causality
        # -----------------------------------------------------------------------
        st.markdown("### 3. Granger Causality — Do Load Columns Predict OT?")
        st.markdown(
            '<div class="context-card" style="font-size:0.82rem;">'
            "The Granger causality test asks: <em>does knowing past values of X improve "
            "our prediction of Y beyond knowing Y's own past?</em> "
            "H₀: X does not Granger-cause OT. p &lt; 0.05 → reject H₀ → X is informative for forecasting OT. "
            "This justifies including load columns as multivariate covariates for ML models."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            from statsmodels.tsa.stattools import grangercausalitytests as _granger

            load_cols_gc = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
            gc_lags = [1, 6, 24]
            gc_rows = []

            for lc in load_cols_gc:
                xy = _ett_st[[lc, "OT"]].dropna().values
                row = {"Feature": lc}
                try:
                    gc_res = _granger(xy, maxlag=max(gc_lags), verbose=False)
                    for lg in gc_lags:
                        p = gc_res[lg][0]["ssr_ftest"][1]
                        row[f"p (lag {lg}h)"] = round(p, 4)
                        row[f"sig {lg}h"] = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "—"))
                except Exception:
                    for lg in gc_lags:
                        row[f"p (lag {lg}h)"] = "err"
                        row[f"sig {lg}h"] = "err"
                gc_rows.append(row)

            gc_df = pd.DataFrame(gc_rows).set_index("Feature")
            cols_order = []
            for lg in gc_lags:
                cols_order += [f"p (lag {lg}h)", f"sig {lg}h"]
            st.dataframe(gc_df[cols_order], use_container_width=True)
            st.caption("* p<0.05  ** p<0.01  *** p<0.001  — = not significant")

            # Interpretation
            sig_features = [
                lc for lc in load_cols_gc
                if any(
                    isinstance(gc_df.loc[lc, f"p (lag {lg}h)"], float)
                    and gc_df.loc[lc, f"p (lag {lg}h)"] < 0.05
                    for lg in gc_lags
                )
            ]
            not_sig = [lc for lc in load_cols_gc if lc not in sig_features]

            feature_meanings = {
                "HUFL": "high-voltage useful load", "HULL": "high-voltage useless load",
                "MUFL": "mid-voltage useful load", "MULL": "mid-voltage useless load",
                "LUFL": "low-voltage useful load", "LULL": "low-voltage useless load",
            }
            sig_desc = ", ".join(f"<strong>{f}</strong> ({feature_meanings.get(f, f)})" for f in sig_features)
            not_desc = ", ".join(f"<strong>{f}</strong>" for f in not_sig) if not_sig else "none"

            gc_interp = []
            if sig_features:
                gc_interp.append(
                    f"{sig_desc} Granger-cause OT — meaning their past values contain predictive information about oil temperature "
                    f"<em>beyond what OT's own history already tells us</em>. "
                    "This statistically justifies using these load columns as covariates in multivariate ML models."
                )
            if not_sig:
                gc_interp.append(
                    f"{not_desc} did <em>not</em> show significant Granger causality — "
                    "their past values add little incremental predictive value once OT's own lags are accounted for."
                )
            gc_interp.append(
                "<em>Important caveat:</em> Granger causality is a statistical association, not physical causation. "
                "High load → more heat → higher OT is the physical mechanism, but the test only confirms predictive utility."
            )

            st.markdown(
                '<div class="context-card" style="font-size:0.82rem; border-left: 3px solid #4C8BF5;">'
                "<strong>What this means:</strong><br>"
                + "<br><br>".join(gc_interp)
                + "</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Granger causality test failed: {e}")

        st.divider()

        # -----------------------------------------------------------------------
        # Section 4: Diebold-Mariano Model Comparison
        # -----------------------------------------------------------------------
        st.markdown("### 4. Diebold-Mariano Test — Are Model Differences Statistically Significant?")
        st.markdown(
            '<div class="context-card" style="font-size:0.82rem;">'
            "The <strong>Diebold-Mariano (DM) test</strong> asks: is model A's forecast accuracy "
            "statistically significantly different from model B's? "
            "H₀: equal predictive accuracy. DM statistic ~ N(0,1) under H₀. "
            "p &lt; 0.05 → the difference in RMSE between these two models is unlikely due to chance."
            "</div>",
            unsafe_allow_html=True,
        )

        def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
            d = e1 ** 2 - e2 ** 2
            n = len(d)
            d_mean = d.mean()
            # Newey-West variance at lag h=1
            gamma0 = np.mean((d - d_mean) ** 2)
            gamma1 = np.mean((d[1:] - d_mean) * (d[:-1] - d_mean))
            nw_var = (gamma0 + 2 * gamma1) / n
            if nw_var <= 0:
                return 0.0, 1.0
            dm_stat = d_mean / np.sqrt(nw_var)
            import math as _math
            p_val = _math.erfc(abs(dm_stat) / _math.sqrt(2))
            return float(dm_stat), float(p_val)

        dm_models = [k for k in selected_keys if load_model_arrays(k).get("y_pred")]
        if len(dm_models) < 2:
            st.info("Select at least 2 models in the sidebar to run DM tests.")
        else:
            dm_col_a, dm_col_b = st.columns(2)
            dm_model_labels = {display_name(k, results[k].get("_model_key", k)): k for k in dm_models}
            model_a_label = dm_col_a.selectbox("Model A", list(dm_model_labels.keys()), key="dm_a")
            remaining = [l for l in dm_model_labels if l != model_a_label]
            model_b_label = dm_col_b.selectbox("Model B", remaining, key="dm_b")

            key_a = dm_model_labels[model_a_label]
            key_b = dm_model_labels[model_b_label]
            _arr_a = load_model_arrays(key_a)
            _arr_b = load_model_arrays(key_b)
            e1 = np.array(_arr_a["y_true"]) - np.array(_arr_a["y_pred"])
            e2 = np.array(_arr_b["y_true"]) - np.array(_arr_b["y_pred"])
            min_len = min(len(e1), len(e2))
            dm_stat, dm_p = diebold_mariano(e1[:min_len], e2[:min_len])
            rmse_a = results[key_a].get("metrics", {}).get("RMSE", float("nan"))
            rmse_b = results[key_b].get("metrics", {}).get("RMSE", float("nan"))
            winner = model_a_label if rmse_a < rmse_b else model_b_label

            sig_color = "#56D364" if dm_p < 0.05 else "#FF6B6B"
            conclusion = (
                f"<strong>Significant</strong> at p={dm_p:.4f} — {winner} is statistically better (p&lt;0.05)"
                if dm_p < 0.05
                else f"<strong>Not significant</strong> at p={dm_p:.4f} — cannot reject equal accuracy"
            )
            rmse_diff_pct = abs(rmse_a - rmse_b) / max(rmse_a, rmse_b) * 100
            st.markdown(
                f'<div class="context-card">'
                f"<strong>{model_a_label}</strong> RMSE = {rmse_a:.4f} &nbsp;|&nbsp; "
                f"<strong>{model_b_label}</strong> RMSE = {rmse_b:.4f}<br>"
                f"DM statistic = {dm_stat:.4f}<br>"
                f"p-value = <span style='color:{sig_color}; font-weight:600'>{dm_p:.4f}</span><br>"
                f"{conclusion}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Contextual interpretation
            if dm_p < 0.05:
                dm_interp = (
                    f"The {rmse_diff_pct:.1f}% RMSE gap between {model_a_label} and {model_b_label} is "
                    f"<strong>statistically real</strong> — not sampling noise. "
                    f"In a production deployment, choosing {winner} over the alternative is justified by evidence. "
                    "However, also consider inference cost and latency: if the worse model is 10× faster and the "
                    "gap is small in absolute terms (e.g., 0.1°C), the trade-off may still favour the simpler model."
                )
            else:
                dm_interp = (
                    f"The RMSE difference ({rmse_diff_pct:.1f}%) between {model_a_label} and {model_b_label} "
                    f"<strong>cannot be distinguished from random variation</strong>. "
                    "Both models are statistically equivalent on this test set. "
                    "In this case, prefer the simpler model — lower training cost, easier to explain, less risk of overfitting on a different dataset."
                )
            st.markdown(
                f'<div class="context-card" style="font-size:0.82rem; border-left: 3px solid #4C8BF5;">'
                f"<strong>What to do with this result:</strong><br>{dm_interp}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # DM heatmap for all selected pairs
            if st.checkbox("Show full pairwise DM p-value matrix", value=False):
                n_m = len(dm_models)
                dm_matrix = np.ones((n_m, n_m))
                labels_dm = [display_name(k, results[k].get("_model_key", k)) for k in dm_models]
                for i, ki in enumerate(dm_models):
                    for j, kj in enumerate(dm_models):
                        if i != j:
                            _ai = load_model_arrays(ki)
                            _aj = load_model_arrays(kj)
                            ei = np.array(_ai["y_true"]) - np.array(_ai["y_pred"])
                            ej = np.array(_aj["y_true"]) - np.array(_aj["y_pred"])
                            mlen = min(len(ei), len(ej))
                            _, pv = diebold_mariano(ei[:mlen], ej[:mlen])
                            dm_matrix[i, j] = pv

                fig_dm = go.Figure(go.Heatmap(
                    z=dm_matrix,
                    x=labels_dm, y=labels_dm,
                    colorscale=[[0, "#56D364"], [0.05, "#56D364"], [0.05, "#161B22"], [1, "#161B22"]],
                    zmin=0, zmax=1,
                    text=np.round(dm_matrix, 3).astype(str),
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    hovertemplate="Row=%{y}<br>Col=%{x}<br>p=%{z:.4f}<extra></extra>",
                ))
                fig_dm.update_layout(
                    template=PLOTLY_TEMPLATE, height=400,
                    margin=dict(t=30, b=80, l=120, r=20),
                    annotations=[dict(
                        text="Green = p<0.05 (significant difference). Diagonal = same model (p=1).",
                        xref="paper", yref="paper", x=0, y=-0.18,
                        showarrow=False, font=dict(size=9, color="#8B949E"),
                    )],
                )
                st.plotly_chart(fig_dm, use_container_width=True)

                # Count significant pairs
                n_sig = int(np.sum(dm_matrix < 0.05)) - n_m  # exclude diagonal
                n_total_pairs = n_m * (n_m - 1)
                pct_sig = n_sig / n_total_pairs * 100 if n_total_pairs > 0 else 0

                # Find non-significant pairs (p >= 0.05, off-diagonal)
                nonsig_pairs = []
                for i, ki in enumerate(dm_models):
                    for j, kj in enumerate(dm_models):
                        if i < j and dm_matrix[i, j] >= 0.05:
                            la = display_name(ki, results[ki].get("_model_key", ki))
                            lb = display_name(kj, results[kj].get("_model_key", kj))
                            nonsig_pairs.append(f"{la} vs {lb} (p={dm_matrix[i,j]:.3f})")

                st.markdown(
                    '<div class="context-card" style="font-size:0.82rem; border-left: 3px solid #4C8BF5;">'
                    "<strong>How to read this matrix:</strong><br>"
                    "Each cell (row=Model A, col=Model B) shows the DM p-value for whether A and B differ significantly. "
                    "<strong>Green = significant</strong> (p&lt;0.05, real difference). "
                    "<strong>Black = not significant</strong> (p≥0.05, models are statistically equivalent).<br><br>"
                    f"<strong>Summary:</strong> {n_sig} of {n_total_pairs} pairs ({pct_sig:.0f}%) are significantly different. "
                    + (
                        f"Statistically equivalent pairs: {'; '.join(nonsig_pairs)}. "
                        "For these pairs, prefer the simpler or faster model — the accuracy gain is within noise."
                        if nonsig_pairs else
                        "All model pairs are significantly different — each model occupies a distinct performance tier."
                    )
                    + "<br><br><strong>What to consider next:</strong><br>"
                    "1. <strong>Model selection:</strong> Pick the best model from the top tier (lowest RMSE) that is statistically separated from others.<br>"
                    "2. <strong>Ensemble opportunity:</strong> Models with significantly different errors may produce uncorrelated mistakes — blending them can reduce variance.<br>"
                    "3. <strong>Test set caveat:</strong> DM tests are only valid on the same held-out test set. If you retrain on more data, rerun the test."
                    "</div>",
                    unsafe_allow_html=True,
                )

# ===========================================================================
# TAB 6 — Ask AI (RAG)  [was tab5]
# ===========================================================================

if page.key == "agent":
    _rec = json.loads((Path(__file__).resolve().parent / "replays_ask_ai.json").read_text())["runs"]
    F.header(st, PAGES[8], "Can I ask about the benchmark in plain words?",
             f"Yes. The assistant answers from this project's committed results, fair benchmark first, and cites each source; "
             f"{sum(r['grounded'] for r in _rec)} of {len(_rec)} recorded answers pass the number check. Replay is free; with your own "
             "OpenAI key you can ask anything, 10 questions per session, and the key is never stored.")
    st.markdown(
        '<div class="context-card">'
        "Ask about the models, the fair benchmark or the dataset. The assistant retrieves context from this project's committed "
        "results (the fair rolling benchmark first; the original table is labelled not like-for-like) and answers with "
        "<strong>gpt-4o-mini</strong>, citing each source. Recorded answers by default; use your own OpenAI key to ask live."
        "</div>",
        unsafe_allow_html=True,
    )
    chat.render(st, Path(__file__).resolve().parent / "replays_ask_ai.json", ask_ai, prefix="ask_")

# ===========================================================================
# TAB 7 — RL Selector (LinUCB contextual bandit)
# ===========================================================================

if page.key == "rl":
    _rl_head = st.container()                        # filled once the bandit has run
    st.markdown(
        '<div class="context-card">'
        "A <strong>contextual bandit</strong> (LinUCB) learns which of the 11 forecasting models "
        "to use for each 24-hour window based on observable features of the preceding window — "
        "trend direction, volatility, autocorrelation, and more. "
        "Training is fully offline: it replays all 145 test windows using the pre-computed "
        "RMSE from every model as feedback, so no simulation or re-inference is needed."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("How LinUCB works", expanded=False):
        st.markdown(
            """
Each model is an **arm**. At every 24-hour window the bandit:
1. Extracts 7 features from the preceding window (level, volatility, trend, autocorrelation, max jump, range, skewness) plus a bias term.
2. Computes a **UCB score** for each arm: `θ_a · x + α √(x⊤ A_a⁻¹ x)` — expected reward plus an exploration bonus that shrinks as the arm accumulates data.
3. Selects the arm with the highest UCB score.
4. Receives all 11 models' true RMSE as feedback (full-feedback offline update).

**Reward**: normalised per window — 1.0 if this model had the lowest RMSE in this window, 0.0 if it had the highest, linearly interpolated in between.

**Baselines**:
- **Oracle**: always picks the best model for each window (unachievable in practice).
- **Static best**: always picks Random Forest (best average RMSE across all windows).
- **Random**: uniformly random arm selection every window.
"""
        )

    # -----------------------------------------------------------------------
    # Alpha slider + cached training
    # -----------------------------------------------------------------------

    rl_alpha = st.slider(
        "Exploration factor α (higher = more exploration in early windows)",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1,
        help="Controls the UCB exploration bonus. α=0 → pure exploitation; α=3.0 → heavy exploration.",
    )

    @st.cache_data(show_spinner="Training LinUCB bandit…")
    def _run_bandit(forecast_dir: str, alpha: float) -> dict:
        try:
            import sys as _sys
            _sys.path.insert(0, str(PROJECT_ROOT))
            from src.rl.model_selector import BanditModelSelector as _BMS
            _sel = _BMS(forecast_dir, alpha=alpha)
            _sel.train()
            return _sel.get_results()
        except Exception as _e:
            return {"error": str(_e)}

    rl = _run_bandit(str(FORECAST_DIR), rl_alpha)
    F.header(_rl_head, PAGES[7], "Could picking a model for each day beat one model for every day?",
             (f"On the original split (not like-for-like): the bandit averages RMSE {rl['bandit_avg_rmse']:.3f} °C per window, against "
              f"{rl['static_best_avg_rmse']:.3f} °C for always using {MODEL_DISPLAY.get(rl['static_best_name'], rl['static_best_name'])} and "
              f"{rl['oracle_avg_rmse']:.3f} °C for a perfect daily pick.") if "error" not in rl else f"The bandit could not run here ({rl['error']}).")

    if "error" in rl:
        st.error(f"Bandit training failed: {rl['error']}")
        st.stop()

    arm_names = rl["arm_names"]
    n_arms = len(arm_names)
    n_win = rl["n_windows"]

    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Oracle avg RMSE",
            f"{rl['oracle_avg_rmse']:.4f} °C",
            help="Best possible — always picks the lowest-RMSE model per window.",
        )
    with k2:
        st.metric(
            "Bandit avg RMSE",
            f"{rl['bandit_avg_rmse']:.4f} °C",
            delta=f"{rl['bandit_avg_rmse'] - rl['oracle_avg_rmse']:+.4f} vs oracle",
            delta_color="inverse",
        )
    with k3:
        st.metric(
            f"Static best ({MODEL_DISPLAY.get(rl['static_best_name'], rl['static_best_name'])})",
            f"{rl['static_best_avg_rmse']:.4f} °C",
            delta=f"{rl['static_best_avg_rmse'] - rl['oracle_avg_rmse']:+.4f} vs oracle",
            delta_color="inverse",
        )
    with k4:
        improvement_pct = 100 * (rl["random_avg_rmse"] - rl["bandit_avg_rmse"]) / rl["random_avg_rmse"]
        st.metric(
            "Bandit improvement vs random",
            f"{improvement_pct:.1f}%",
            help=f"Random baseline avg RMSE: {rl['random_avg_rmse']:.4f} °C",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 1: Cumulative Regret  +  Arm Selection Distribution
    # -----------------------------------------------------------------------
    col_regret, col_pie = st.columns([PHI, 1])

    with col_regret:
        st.markdown("#### Cumulative Regret vs Oracle")
        st.markdown(
            '<div class="context-card" style="font-size:0.82rem;">'
            "Regret = how much RMSE was wasted by not always picking the oracle best model. "
            "Lower is better. The bandit rapidly eliminates statistical and deep-learning models "
            "that consistently underperform, slashing regret vs the random baseline."
            "</div>",
            unsafe_allow_html=True,
        )
        windows = list(range(1, n_win + 1))
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=windows, y=rl["regret_bandit"],
            mode="lines", name="LinUCB Bandit",
            line=dict(color="#4C8BF5", width=2),
        ))
        fig_reg.add_trace(go.Scatter(
            x=windows, y=rl["regret_static"],
            mode="lines", name=f"Static Best ({MODEL_DISPLAY.get(rl['static_best_name'], rl['static_best_name'])})",
            line=dict(color="#56D364", width=2, dash="dot"),
        ))
        fig_reg.add_trace(go.Scatter(
            x=windows, y=rl["regret_random"],
            mode="lines", name="Random Baseline",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
        ))
        fig_reg.update_layout(
            xaxis_title="Window (24-hour step)",
            yaxis_title="Cumulative RMSE regret (°C)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_pie:
        st.markdown("#### Arm Selection Distribution")
        st.markdown(
            '<div class="context-card" style="font-size:0.82rem;">'
            "Which model the bandit selected across all 145 windows."
            "</div>",
            unsafe_allow_html=True,
        )
        counts = rl["arm_selection_counts"]
        pie_labels = [MODEL_DISPLAY.get(a, a) for a in arm_names]
        pie_colors = [FAMILY_COLORS.get(FAMILY_MAP.get(a, "ML"), "#888") for a in arm_names]
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels,
            values=counts,
            marker=dict(colors=pie_colors),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} windows (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------------------------------------------------
    # Section 2: Per-window RMSE comparison
    # -----------------------------------------------------------------------
    st.markdown("#### Per-Window RMSE — Bandit vs Baselines")
    st.markdown(
        '<div class="context-card" style="font-size:0.82rem;">'
        "RMSE of the selected model for each of the 145 24-hour windows. "
        "Oracle always picks the per-window winner. "
        "The bandit closely tracks oracle after the first ~20 exploration windows. "
        "Statistical models (arima, holt_winters) selected in early exploration "
        "cause the visible spikes."
        "</div>",
        unsafe_allow_html=True,
    )
    fig_win = go.Figure()
    fig_win.add_trace(go.Scatter(
        x=windows, y=rl["per_window_oracle_rmse"],
        mode="lines", name="Oracle (best per window)",
        line=dict(color="#56D364", width=1.5, dash="dot"),
        opacity=0.85,
    ))
    fig_win.add_trace(go.Scatter(
        x=windows, y=rl["per_window_bandit_rmse"],
        mode="lines", name="LinUCB Bandit",
        line=dict(color="#4C8BF5", width=2),
    ))
    fig_win.add_trace(go.Scatter(
        x=windows, y=rl["per_window_static_rmse"],
        mode="lines", name=f"Static Best ({MODEL_DISPLAY.get(rl['static_best_name'], rl['static_best_name'])})",
        line=dict(color="#F5A623", width=1.5, dash="dash"),
        opacity=0.75,
    ))
    fig_win.update_layout(
        xaxis_title="Window index",
        yaxis_title="RMSE (°C)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
    )
    st.plotly_chart(fig_win, use_container_width=True)

    # -----------------------------------------------------------------------
    # Section 3: Selection history scatter
    # -----------------------------------------------------------------------
    st.markdown("#### Model Selection History")
    st.markdown(
        '<div class="context-card" style="font-size:0.82rem;">'
        "Which arm was chosen at each window. "
        "Statistical models appear only in the first few exploration windows; "
        "the bandit quickly converges to the ML family."
        "</div>",
        unsafe_allow_html=True,
    )
    sel_hist = rl["per_window_selections"]
    sel_labels = [MODEL_DISPLAY.get(arm_names[i], arm_names[i]) for i in sel_hist]
    sel_families = [FAMILY_MAP.get(arm_names[i], "ML") for i in sel_hist]
    sel_colors_list = [FAMILY_COLORS.get(f, "#888") for f in sel_families]

    fig_hist = go.Figure(go.Scatter(
        x=windows,
        y=sel_labels,
        mode="markers",
        marker=dict(color=sel_colors_list, size=8, symbol="circle"),
        hovertemplate="Window %{x}: %{y}<extra></extra>",
    ))
    fig_hist.update_layout(
        xaxis_title="Window index",
        yaxis=dict(categoryorder="array", categoryarray=[
            MODEL_DISPLAY.get(a, a) for a in arm_names
        ]),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
    )
    # Add family-colour legend manually
    for fam, fcolor in FAMILY_COLORS.items():
        fig_hist.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=fcolor, size=10),
            name=fam, showlegend=True,
        ))
    fig_hist.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------------------------------
    # Section 4: Per-arm summary table
    # -----------------------------------------------------------------------
    st.markdown("#### Arm Summary")
    arm_avg_rmses = rl["arm_avg_rmses"]
    arm_rows = []
    for i, arm in enumerate(arm_names):
        arm_rows.append({
            "Model": MODEL_DISPLAY.get(arm, arm),
            "Family": FAMILY_MAP.get(arm, "—"),
            "Times Selected": int(counts[i]),
            "Selection %": f"{100 * counts[i] / n_win:.1f}%",
            "Avg RMSE (all windows)": f"{arm_avg_rmses[i]:.4f} °C",
        })
    arm_df = pd.DataFrame(arm_rows).sort_values("Times Selected", ascending=False)
    st.dataframe(arm_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Section 5: What-If Predictor
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### What-If Predictor")
    st.markdown(
        '<div class="context-card">'
        "Set the characteristics of a hypothetical time-series window. "
        "The bandit's learned linear model scores each arm and recommends "
        "the model most likely to have the lowest RMSE under those conditions."
        "</div>",
        unsafe_allow_html=True,
    )

    feat_min = rl["feat_min"]
    feat_max = rl["feat_max"]
    feat_mean = rl["feat_mean"]

    wif_col1, wif_col2 = st.columns(2)
    with wif_col1:
        wif_vol = st.slider(
            "Volatility (local std / global std)",
            min_value=float(feat_min[1]),
            max_value=float(feat_max[1]),
            value=float(feat_mean[1]),
            step=0.01,
            help="How much the series fluctuates in this window relative to the global scale.",
        )
        wif_trend = st.slider(
            "Trend (slope per step / global std)",
            min_value=float(feat_min[2]),
            max_value=float(feat_max[2]),
            value=float(feat_mean[2]),
            step=0.005,
            format="%.3f",
            help="Positive = series is rising, negative = falling, 0 = flat.",
        )
    with wif_col2:
        wif_autocorr = st.slider(
            "Autocorrelation (lag-1 Pearson r)",
            min_value=float(feat_min[3]),
            max_value=float(feat_max[3]),
            value=float(feat_mean[3]),
            step=0.01,
            help="How strongly each hour predicts the next. High = smooth/persistent.",
        )
        wif_skew = st.slider(
            "Skewness",
            min_value=float(feat_min[6]),
            max_value=float(feat_max[6]),
            value=float(feat_mean[6]),
            step=0.1,
            format="%.1f",
            help="Positive = right-tailed (rare large spikes up), negative = left-tailed.",
        )

    # Build synthetic context vector from sliders; derive other features proportionally
    wif_context = np.array([
        feat_mean[0],                      # level — neutral
        wif_vol,                           # volatility
        wif_trend,                         # trend
        wif_autocorr,                      # autocorr
        wif_vol * (feat_mean[4] / (feat_mean[1] + 1e-8)),   # max_jump proportional
        wif_vol * (feat_mean[5] / (feat_mean[1] + 1e-8)),   # range_ratio proportional
        wif_skew,                          # skewness
        1.0,                               # bias
    ], dtype=np.float32)

    # Score each arm using learned theta weights (no UCB bonus — pure exploitation estimate)
    weights = np.array(rl["weights"], dtype=np.float64)   # (n_arms, n_features)
    wif_scores = (weights @ wif_context.astype(np.float64)).tolist()
    best_arm_idx = int(np.argmax(wif_scores))
    best_arm_name = arm_names[best_arm_idx]
    best_family = FAMILY_MAP.get(best_arm_name, "ML")
    best_color = FAMILY_COLORS.get(best_family, "#888")

    st.markdown(
        f'<div class="context-card" style="border-left:4px solid {best_color}; font-size:1.05rem;">'
        f"<strong>Recommended model:</strong> "
        f"<span style='color:{best_color}; font-size:1.2rem; font-weight:700'>"
        f"{MODEL_DISPLAY.get(best_arm_name, best_arm_name)}</span> "
        f"<span style='font-size:0.85rem; color:#aaa'>({best_family})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Horizontal bar chart of scores
    score_labels = [MODEL_DISPLAY.get(a, a) for a in arm_names]
    score_colors = [FAMILY_COLORS.get(FAMILY_MAP.get(a, "ML"), "#888") for a in arm_names]
    sorted_idx = sorted(range(n_arms), key=lambda i: wif_scores[i])
    fig_wif = go.Figure(go.Bar(
        x=[wif_scores[i] for i in sorted_idx],
        y=[score_labels[i] for i in sorted_idx],
        orientation="h",
        marker_color=[score_colors[i] for i in sorted_idx],
        hovertemplate="%{y}: score %{x:.4f}<extra></extra>",
    ))
    fig_wif.update_layout(
        xaxis_title="Expected reward score (higher = predicted lower RMSE)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
    )
    st.plotly_chart(fig_wif, use_container_width=True)

    st.caption(
        "Scores are linear expected-reward estimates from the learned LinUCB weights "
        f"(α = {rl_alpha}). The bandit was trained on {n_win} × 24-hour windows "
        f"({n_arms} arms). Reliable within the observed feature ranges; "
        "extrapolation beyond those ranges is not guaranteed."
    )
