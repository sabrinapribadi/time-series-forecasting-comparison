"""
Time Series Forecasting Benchmark Dashboard

Reads pre-computed results from data/forecasts/ETT/*.json.
No model weights or raw data required at runtime (ETT CSVs used only in Data Explorer).

Tabs:
  1. Benchmark Results  -- KPI cards, metric table, ranked bar chart, radar chart
  2. Forecast Gallery   -- test-set overlay with zoom and family colouring
  3. Model Inspector    -- residuals, error distribution, metric deep-dive
  4. Data Explorer      -- ETT dataset visualisation, correlation, seasonality
  5. Ask AI             -- RAG assistant powered by Claude (Anthropic API)
"""

from __future__ import annotations

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
FAMILY_MAP: dict[str, str] = {
    "holt_winters": "Statistical",
    "arima": "Statistical",
    "prophet": "Statistical",
    "random_forest": "ML",
    "xgboost": "ML",
    "lightgbm": "ML",
    "catboost": "ML",
    "lstm": "Deep Learning",
    "transformer": "Deep Learning",
    "nbeats": "Deep Learning",
    "tft": "Deep Learning",
}

FAMILY_COLORS: dict[str, str] = {
    "Statistical": "#F5A623",
    "ML": "#4C8BF5",
    "Deep Learning": "#9B8FFF",
}

MODEL_DISPLAY: dict[str, str] = {
    "holt_winters": "Holt-Winters",
    "arima": "ARIMA",
    "prophet": "Prophet",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "nbeats": "N-BEATS",
    "tft": "TFT",
}

# Metric metadata: label, direction (lower/higher is better), formula, description, unit
METRIC_META: dict[str, dict] = {
    "RMSE": {
        "label": "RMSE",
        "full": "Root Mean Squared Error",
        "direction": "lower",
        "formula": "sqrt(mean((y - y_hat)^2))",
        "description": (
            "Penalises large errors more than small ones due to squaring. "
            "Use when large deviations are especially costly — e.g., extreme temperature spikes."
        ),
        "unit": "°C",
    },
    "MAE": {
        "label": "MAE",
        "full": "Mean Absolute Error",
        "direction": "lower",
        "formula": "mean(|y - y_hat|)",
        "description": (
            "Treats all error magnitudes equally — more robust to outliers than RMSE. "
            "Preferred when you want a straightforward average mistake in the original unit."
        ),
        "unit": "°C",
    },
    "MAPE": {
        "label": "MAPE",
        "full": "Mean Absolute Percentage Error",
        "direction": "lower",
        "formula": "mean(|y - y_hat| / |y|) * 100",
        "description": (
            "Scale-free percentage error. Note: undefined or inflated when actuals are "
            "near zero — ETTh1 OT can go negative in winter, so values below 0.1 are excluded."
        ),
        "unit": "%",
    },
    "SMAPE": {
        "label": "SMAPE",
        "full": "Symmetric MAPE",
        "direction": "lower",
        "formula": "mean(|y - y_hat| / ((|y| + |y_hat|)/2)) * 100",
        "description": (
            "Symmetric version of MAPE. Range [0, 200%]. Handles near-zero actuals better "
            "than MAPE, and treats over- and under-forecasting symmetrically."
        ),
        "unit": "%",
    },
    "MASE": {
        "label": "MASE",
        "full": "Mean Absolute Scaled Error",
        "direction": "lower",
        "formula": "MAE / mean(|y_train[t] - y_train[t-1]|)",
        "description": (
            "Scale-free metric relative to the in-sample naive (random walk) baseline. "
            "MASE < 1 means the model beats naive forecasting. Comparable across ETT variants."
        ),
        "unit": "—",
    },
    "R2": {
        "label": "R²",
        "full": "Coefficient of Determination",
        "direction": "higher",
        "formula": "1 - SS_res / SS_tot",
        "description": (
            "Proportion of variance in OT explained by the model. "
            "R² = 1 is perfect; R² = 0 means the model is no better than predicting the mean; "
            "R² < 0 means it is worse than predicting the mean."
        ),
        "unit": "—",
    },
}

# Detailed model context for sidebar and RAG knowledge base
MODEL_META: dict[str, dict] = {
    "holt_winters": {
        "label": "Holt-Winters",
        "family": "Statistical",
        "algorithm": "Exponential Smoothing (additive trend + seasonality)",
        "description": "Decomposes the series into level, trend, and seasonal components. Each component is updated with exponential smoothing coefficients (alpha, beta, gamma). Suitable for series with stable seasonality.",
        "strengths": "Fast, interpretable, no feature engineering, handles seasonality explicitly",
        "weaknesses": "Assumes fixed seasonality structure; degrades severely over long horizons (3,484-step test); not suitable for multivariate input",
        "formula": "y_hat(t+h) = l_t + h*b_t + s_{t+h-m(k+1)}",
    },
    "arima": {
        "label": "ARIMA",
        "family": "Statistical",
        "algorithm": "AutoRegressive Integrated Moving Average",
        "description": "Combines autoregression (AR), differencing for stationarity (I), and moving-average error correction (MA). Order (p,d,q) selected automatically via ADF stationarity test.",
        "strengths": "Theoretically grounded, handles non-stationarity, well-studied",
        "weaknesses": "Linear; assumes stationarity after differencing; no native seasonality without SARIMA; degrades over long horizons",
        "formula": "phi(B) * nabla^d * y_t = theta(B) * epsilon_t",
    },
    "prophet": {
        "label": "Prophet",
        "family": "Statistical",
        "algorithm": "Additive decomposition with piecewise linear trend",
        "description": "Facebook's decomposable time series model: y(t) = trend + seasonality + holidays + noise. Trend uses piecewise linear changepoints. Seasonality uses Fourier series.",
        "strengths": "Handles missing data and outliers, automatic changepoint detection, easy to tune",
        "weaknesses": "Designed for daily data with weekly/yearly patterns; hourly OT forecasting is outside its intended domain; long-horizon degradation",
        "formula": "y(t) = g(t) + s(t) + h(t) + epsilon_t",
    },
    "random_forest": {
        "label": "Random Forest",
        "family": "ML",
        "algorithm": "Bootstrap-aggregated regression trees",
        "description": "Ensemble of B decision trees trained on bootstrap samples with random feature subsets. Variance reduction through averaging. Uses oracle lag features (OT_lag_1/2/24) as input.",
        "strengths": "Robust to outliers, handles non-linearity, no scaling needed, parallelisable",
        "weaknesses": "Cannot extrapolate beyond training range, requires feature engineering, no explicit sequence modelling",
        "formula": "y_hat(x) = (1/B) * sum_b f_b(x)",
    },
    "xgboost": {
        "label": "XGBoost",
        "family": "ML",
        "algorithm": "Gradient boosting with second-order Taylor expansion",
        "description": "Sequential ensemble of regression trees fitted by minimising a regularised second-order approximation of the loss. Features L1/L2 regularisation, column/row subsampling.",
        "strengths": "High accuracy, built-in regularisation, efficient parallel tree construction",
        "weaknesses": "More hyperparameters than RF, can overfit without tuning, no sequence awareness",
        "formula": "F_K(x) = sum_k eta * f_k(x),  Omega(f) = gamma*T + lambda/2 * ||w||^2",
    },
    "lightgbm": {
        "label": "LightGBM",
        "family": "ML",
        "algorithm": "Histogram-based gradient boosting with leaf-wise growth",
        "description": "Same gradient boosting objective as XGBoost but uses histogram binning, leaf-wise (best-first) tree growth, GOSS sampling, and EFB bundling for 10-100x speedup on large data. Uses IOH AOP2026 production params (reg_alpha=0.1, reg_lambda=0.1).",
        "strengths": "Fastest gradient boosting, memory efficient, strong regularisation, large datasets",
        "weaknesses": "Leaf-wise growth can overfit on small datasets; less interpretable than level-wise",
        "formula": "Histogram split: O(#bins) vs O(n log n); leaf-wise: expand leaf with max gain",
    },
    "catboost": {
        "label": "CatBoost",
        "family": "ML",
        "algorithm": "Ordered boosting with oblivious trees",
        "description": "Uses ordered boosting (prevents target leakage) and symmetric oblivious trees (same split at each depth level). Particularly strong on datasets with categorical features.",
        "strengths": "Ordered boosting reduces overfitting, fast inference via lookup tables, handles categoricals",
        "weaknesses": "Slower training than LightGBM on purely numerical data",
        "formula": "Oblivious tree: O(2^depth) inference lookup; ordered boosting uses sigma_r(x_i) permutation",
    },
    "lstm": {
        "label": "LSTM",
        "family": "Deep Learning",
        "algorithm": "Long Short-Term Memory (PyTorch)",
        "description": "Recurrent neural network with input, forget, and output gates to mitigate vanishing gradients. Trained with 96-step input window, 24-step direct output head. Evaluated with rolling-window oracle (context refreshed with true OT every 24 steps).",
        "strengths": "Captures long-range temporal dependencies, no manual feature engineering",
        "weaknesses": "Requires significant training time (100+ epochs for competitive results); here only 30 epochs — undertrained",
        "formula": "c_t = f_t * c_{t-1} + i_t * c_tilde_t,  h_t = o_t * tanh(c_t)",
    },
    "transformer": {
        "label": "Transformer",
        "family": "Deep Learning",
        "algorithm": "Multi-head self-attention (Darts)",
        "description": "Replaces recurrence with scaled dot-product attention, enabling parallel computation over the full input sequence. Captures global dependencies without distance bias.",
        "strengths": "Parallelisable training, captures global patterns, strong on long sequences",
        "weaknesses": "O(n^2) attention cost; undertrained here (5 epochs); needs GPU for competitive results",
        "formula": "Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V",
    },
    "nbeats": {
        "label": "N-BEATS",
        "family": "Deep Learning",
        "algorithm": "Neural Basis Expansion (Darts)",
        "description": "Pure MLP stack without convolution or attention. Each block produces a backcast (residual) and forecast (contribution). Generic or interpretable (trend/seasonality) basis functions.",
        "strengths": "No inductive bias, purely data-driven, interpretable decomposition option",
        "weaknesses": "Needs many training epochs; undertrained here (30 epochs); no native multivariate support",
        "formula": "y_hat = sum_l f_l; x_{l+1} = x_l - b_l  (block-wise residual connections)",
    },
    "tft": {
        "label": "TFT",
        "family": "Deep Learning",
        "algorithm": "Temporal Fusion Transformer (Darts)",
        "description": "Combines variable selection networks, gated residual networks, and multi-head self-attention. Designed for multi-horizon with interpretable attention. Requires future covariates — approximated here with a relative time index.",
        "strengths": "Multi-horizon, variable importance, interpretable attention over past steps",
        "weaknesses": "Needs real future covariates (load schedule) for best performance; relative index is a poor substitute; undertrained (30 epochs)",
        "formula": "GLU(a,b) = a * sigmoid(b); GRN + VSN + InterpretableMultiHeadAttn",
    },
}

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
    results = {}
    for path in sorted(FORECAST_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        # Derive display name from JSON or stem
        key = path.stem
        model_key = data.get("model", key.split("_h")[0].split("_m")[0])
        results[key] = {**data, "_model_key": model_key}
    return results


@st.cache_data
def load_ett_data(variant: str = "h1") -> pd.DataFrame | None:
    path = RAW_DIR / f"ETT{variant[0].upper()}{variant[1]}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" stroke="#4C8BF5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <div style="font-size:1rem; font-weight:700; color:#E6EDF3; line-height:1.2;">ETT Forecasting</div>
        <div style="font-size:0.72rem; color:#8B949E; letter-spacing:0.05em;">BENCHMARK DASHBOARD</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

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
            max_value=min(1000, len(results[list(results.keys())[0]].get("y_true", []))),
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

    st.divider()
    st.caption("v2.0 · July 2026 · Sabrina Pribadi")

# ---------------------------------------------------------------------------
# Filter DataFrames
# ---------------------------------------------------------------------------

sel_df = metrics_df.loc[metrics_df.index.isin(selected_keys)].copy()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Benchmark Results",
    "Forecast Gallery",
    "Model Inspector",
    "Data Explorer",
    "Ask AI",
])

# ===========================================================================
# TAB 1 — Benchmark Results
# ===========================================================================

with tab1:
    # Header
    st.markdown("## Benchmark Results")
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

with tab2:
    st.markdown("## Forecast Gallery")
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
    first_data = results[selected_keys[0]]
    y_true_full = first_data.get("y_true", [])
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
        if "y_pred" not in data:
            continue
        y_pred = data["y_pred"][:steps]
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
        height=int(240 * PHI),  # ≈ 388px
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
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

with tab3:
    st.markdown("## Model Inspector")

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

    with detail_col:
        if "y_true" in sel_data and "y_pred" in sel_data:
            y_true = np.array(sel_data["y_true"][:n_steps])
            y_pred = np.array(sel_data["y_pred"][:n_steps])
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
                    margin=dict(t=30, b=40, l=50, r=10),
                )
                st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("No predictions available for this model.")

# ===========================================================================
# TAB 4 — Data Explorer
# ===========================================================================

with tab4:
    st.markdown("## Data Explorer")
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
        st.warning(f"ETT{ett_variant[0].upper()}{ett_variant[1]}.csv not found in data/raw/ETT/.")
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
                height=int(200 * PHI) + 100,  # ≈ 424px
                xaxis2_title="Date",
                yaxis_title="Load (MW)",
                yaxis2_title="OT (°C)",
                legend=dict(orientation="h", y=-0.15),
                margin=dict(t=30, b=60, l=60, r=20),
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
        st.markdown(
            '<div class="context-card" style="font-size:0.80rem;">'
            "Pearson correlation between all 7 features. Strong positive correlation (>0.7) "
            "between load columns and OT confirms that load features are informative covariates "
            "for the multivariate forecasting mode."
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

# ===========================================================================
# TAB 5 — Ask AI (RAG)
# ===========================================================================

with tab5:
    st.markdown("## Ask AI")
    st.markdown(
        '<div class="context-card">'
        "Ask natural language questions about the forecasting models, benchmark results, or ETT dataset. "
        "The assistant retrieves relevant context from the benchmark data and uses "
        "<strong>Claude (Anthropic)</strong> to generate grounded, cited answers.<br><br>"
        "Requires an <code>ANTHROPIC_API_KEY</code> in environment or "
        "<code>.streamlit/secrets.toml</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # RAG knowledge base — build once from loaded results
    # ---------------------------------------------------------------------------

    @st.cache_data
    def build_knowledge_base(results_json: str) -> list[dict]:
        """
        Build a list of retrieval chunks from benchmark data and model metadata.
        Each chunk has: id, text, tags.
        """
        import json as _json
        results_data = _json.loads(results_json)

        chunks = []

        # Chunk 1: Dataset overview
        chunks.append({
            "id": "dataset_overview",
            "tags": ["dataset", "ett", "overview", "columns", "target", "split"],
            "text": (
                "ETT (Electricity Transformer Temperature) dataset. "
                "4 variants: ETTh1/ETTh2 (hourly, 17,420 rows), ETTm1/ETTm2 (15-min, 69,680 rows). "
                "8 columns: date, HUFL (high-voltage useful load), HULL (high-voltage useless load), "
                "MUFL (mid-voltage useful load), MULL (mid-voltage useless load), "
                "LUFL (low-voltage useful load), LULL (low-voltage useless load), "
                "OT (oil temperature — the forecasting target, in Celsius). "
                "Split: 70% train / 10% val / 20% test — chronological, no shuffling. "
                "Horizon: 24 steps. Source: github.com/zhouhaoyi/ETDataset (MIT License). "
                "Citation: Zhou et al., 2021, AAAI Informer paper."
            ),
        })

        # Chunk 2: Benchmark summary table
        rows_text = []
        for run_key, data in results_data.items():
            mk = data.get("_model_key", run_key)
            name = MODEL_DISPLAY.get(mk, mk)
            fam = FAMILY_MAP.get(mk, "Unknown")
            m = data.get("metrics", {})
            rmse = m.get("RMSE", "N/A")
            r2 = m.get("R2", "N/A")
            mae = m.get("MAE", "N/A")
            rows_text.append(
                f"  {name} ({fam}): RMSE={rmse:.4f if isinstance(rmse, float) else rmse}, "
                f"MAE={mae:.4f if isinstance(mae, float) else mae}, "
                f"R2={r2:.4f if isinstance(r2, float) else r2}"
            )
        chunks.append({
            "id": "benchmark_results",
            "tags": ["results", "rmse", "mae", "r2", "benchmark", "comparison", "performance", "best", "worst"],
            "text": (
                "ETTh1 benchmark results (horizon=24, univariate, default hyperparameters):\n"
                + "\n".join(rows_text)
                + "\n\nML models use oracle lag features (true OT from previous steps as inputs), "
                "giving them an advantage. Deep learning models ran only 5–30 epochs (undertrained)."
            ),
        })

        # Chunk 3: ML oracle explanation
        chunks.append({
            "id": "oracle_explanation",
            "tags": ["oracle", "lag", "ml", "feature", "advantage", "fair", "bias"],
            "text": (
                "Why do ML models (Random Forest, XGBoost, LightGBM, CatBoost) dominate the benchmark? "
                "They use oracle lag features: OT_lag_1 = true OT at the previous step, OT_lag_2, OT_lag_24. "
                "On the test set, these contain ground-truth values — effectively giving the model the "
                "true previous observation as input. This is equivalent to a near-1-step-ahead problem "
                "for 24-step horizon forecasting. Statistical models forecast all 3,484 test steps in one "
                "shot (no oracle refresh), so their autoregressive error compounds. "
                "This explains the RMSE gap: ML (0.74–0.85) vs Statistical (4.0–7.9)."
            ),
        })

        # Chunk 4: DL undertraining
        chunks.append({
            "id": "dl_undertraining",
            "tags": ["deep learning", "lstm", "transformer", "nbeats", "tft", "epochs", "undertraining", "gpu"],
            "text": (
                "Why do deep learning models (LSTM, Transformer, N-BEATS, TFT) underperform? "
                "They ran for only 5–30 epochs due to CPU/Apple MPS compute limits. "
                "A proper GPU run with 100+ epochs is expected to bring LSTM and Transformer "
                "into the R² > 0.8 range. TFT additionally requires real future covariates "
                "(load schedule for the next 24 hours) — the relative time index used here "
                "is a poor substitute, explaining TFT's worst-in-class RMSE of 9.10."
            ),
        })

        # Chunk per model family
        for fam, models_in_fam in [
            ("Statistical", ["holt_winters", "arima", "prophet"]),
            ("ML", ["random_forest", "xgboost", "lightgbm", "catboost"]),
            ("Deep Learning", ["lstm", "transformer", "nbeats", "tft"]),
        ]:
            parts = []
            for mk in models_in_fam:
                meta = MODEL_META.get(mk, {})
                parts.append(
                    f"{meta.get('label', mk)}: {meta.get('description', '')} "
                    f"Strengths: {meta.get('strengths', '')}. "
                    f"Weaknesses: {meta.get('weaknesses', '')}."
                )
            chunks.append({
                "id": f"family_{fam.lower().replace(' ', '_')}",
                "tags": [fam.lower(), "algorithm", "model", "description"] + models_in_fam,
                "text": f"{fam} model family:\n" + "\n".join(parts),
            })

        # Chunk per metric
        for mk, mm in METRIC_META.items():
            chunks.append({
                "id": f"metric_{mk.lower()}",
                "tags": [mk.lower(), "metric", "evaluation", mm.get("direction", "")],
                "text": (
                    f"{mm['full']} ({mk}): {mm['description']} "
                    f"Formula: {mm['formula']}. Unit: {mm['unit']}. "
                    f"Direction: {'lower is better' if mm['direction'] == 'lower' else 'higher is better'}."
                ),
            })

        # Chunk: Feature engineering
        chunks.append({
            "id": "feature_engineering",
            "tags": ["feature", "lag", "cyclical", "rolling", "preprocessing", "encoding", "hour", "seasonality"],
            "text": (
                "Feature engineering for ML models: "
                "Cyclical time features — hour_sin/cos (period 24h), dow_sin/cos (7 days), "
                "month_sin/cos (12 months). Cyclical encoding avoids discontinuity at period boundaries. "
                "Lag features: OT_lag_1, OT_lag_2, OT_lag_24 (previous 1/2/24 hour OT values). "
                "Rolling features (from IOH AOP2026 pipeline): OT_rolling_mean_3 (3-step rolling mean, "
                "shift 1), OT_rolling_std_3 (volatility), OT_growth_rate (pct_change), OT_trend_3 "
                "(linear slope over last 3 steps). All rolling features shifted by 1 to prevent leakage. "
                "Multivariate mode adds HUFL, HULL, MUFL, MULL, LUFL, LULL as ML covariates."
            ),
        })

        return chunks

    # Serialise results for cache key
    results_json_str = json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_pred")}
         for k, v in results.items()},
        default=str,
    )
    knowledge_chunks = build_knowledge_base(results_json_str)

    def retrieve_chunks(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
        """
        Simple keyword-based retrieval (BM25-style without external deps).
        Scores each chunk by how many query words appear in its tags or text.
        """
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            tag_hits = sum(1 for t in chunk["tags"] if any(w in t for w in query_words))
            text_hits = sum(chunk["text"].lower().count(w) for w in query_words)
            score = tag_hits * 3 + text_hits
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Always include benchmark_results as baseline context
        top = [c for _, c in scored[:top_k]]
        if not any(c["id"] == "benchmark_results" for c in top):
            br = next((c for c in chunks if c["id"] == "benchmark_results"), None)
            if br:
                top[-1] = br  # replace last with benchmark context
        return top

    # ---------------------------------------------------------------------------
    # API key check
    # ---------------------------------------------------------------------------

    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        st.warning(
            "**Anthropic API key not found.** "
            "To enable the AI assistant, add your key to `.streamlit/secrets.toml`:\n\n"
            "```toml\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```\n\n"
            "or set the environment variable `ANTHROPIC_API_KEY` before launching Streamlit."
        )

    # ---------------------------------------------------------------------------
    # Quick-question buttons
    # ---------------------------------------------------------------------------

    st.markdown("#### Quick Questions")
    quick_q_col = st.columns(3)
    quick_questions = [
        "Which model should I use for production?",
        "Why do ML models outperform statistical models?",
        "How does LSTM compare to Transformer?",
        "What does RMSE mean and when should I use it?",
        "What is the oracle lag issue in this benchmark?",
        "How would I run multivariate mode for XGBoost?",
    ]

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    for i, qq in enumerate(quick_questions):
        col_idx = i % 3
        if quick_q_col[col_idx].button(qq, key=f"qq_{i}", use_container_width=True):
            st.session_state.chat_input = qq

    st.divider()

    # ---------------------------------------------------------------------------
    # Chat history
    # ---------------------------------------------------------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong>You</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai"><strong>AI Assistant</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_query = st.text_area(
        "Your question",
        value=st.session_state.chat_input,
        height=80,
        placeholder="e.g. Which model is best for production deployment? Why does XGBoost outperform Holt-Winters?",
        key="chat_textarea",
    )
    st.session_state.chat_input = ""  # reset after rendering

    send_col, clear_col = st.columns([PHI, 1])
    send_clicked = send_col.button("Send", type="primary", use_container_width=True, disabled=(not api_key))
    clear_col.button(
        "Clear history",
        on_click=lambda: st.session_state.__setitem__("chat_history", []),
        use_container_width=True,
    )

    if send_clicked and user_query.strip() and api_key:
        # Retrieve relevant chunks
        retrieved = retrieve_chunks(user_query, knowledge_chunks, top_k=4)
        context_text = "\n\n---\n\n".join(f"[Source: {c['id']}]\n{c['text']}" for c in retrieved)

        system_prompt = (
            "You are a helpful data science assistant specialising in time series forecasting. "
            "You are embedded in a Streamlit dashboard for the ETT Forecasting Benchmark. "
            "Answer the user's question accurately and concisely using the context below. "
            "Cite the source ID when you use specific data. "
            "If the answer is not in the context, say so clearly and give a brief general explanation. "
            "Use clear, professional language — no bullet spam, no excessive headers. "
            "Format numbers to 4 decimal places when referencing benchmark metrics.\n\n"
            f"CONTEXT:\n{context_text}"
        )

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Generating response..."):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ],
                )
                answer = response.content[0].text
                # Show retrieved sources
                source_ids = [c["id"] for c in retrieved]
                answer += f"\n\n*Sources retrieved: {', '.join(source_ids)}*"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"API error: {e}")

    elif send_clicked and not api_key:
        st.error("API key not configured. See instructions above.")

    # Sources viewer
    if user_query.strip():
        with st.expander("Retrieved context chunks (RAG sources)", expanded=False):
            retrieved_preview = retrieve_chunks(user_query, knowledge_chunks, top_k=4)
            for c in retrieved_preview:
                st.markdown(f"**`{c['id']}`** — tags: {', '.join(c['tags'][:6])}")
                st.caption(c["text"][:400] + "..." if len(c["text"]) > 400 else c["text"])
                st.divider()
