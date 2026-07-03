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
        "description": "Same gradient boosting objective as XGBoost but uses histogram binning, leaf-wise (best-first) tree growth, GOSS sampling, and EFB bundling for 10-100x speedup on large data. Fixed regularized defaults (reg_alpha=0.1, reg_lambda=0.1).",
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

    st.divider()
    st.caption("v2.0 · July 2026 · Sabrina Pribadi")

# ---------------------------------------------------------------------------
# Filter DataFrames
# ---------------------------------------------------------------------------

sel_df = metrics_df.loc[metrics_df.index.isin(selected_keys)].copy()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Benchmark Results",
    "Forecast Gallery",
    "Model Inspector",
    "Data Explorer",
    "Statistical Tests",
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
            _ai_key = None
            try:
                _ai_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass
            if not _ai_key:
                _ai_key = os.environ.get("OPENAI_API_KEY")

            if _show_ai_btn and not _ai_key:
                st.caption("Add OPENAI_API_KEY to .streamlit/secrets.toml to enable AI explanations.")

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
                        st.error(f"OpenAI API error: {e}")

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
            "accumulates thermally with a time lag — the Granger causality test (Statistical Tests tab) "
            "reveals the true temporal relationship."
            "</div>",
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 5 — Statistical Tests
# ===========================================================================

with tab5:
    st.markdown("## Statistical Tests")
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
            from scipy import stats as _scipy_stats
            p_val = 2 * (1 - _scipy_stats.norm.cdf(abs(dm_stat)))
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

with tab6:
    st.markdown("## Ask AI")
    st.markdown(
        '<div class="context-card">'
        "Ask natural language questions about the forecasting models, benchmark results, or ETT dataset. "
        "The assistant retrieves relevant context from the benchmark data and uses "
        "<strong>GPT-4o mini (OpenAI)</strong> to generate grounded, cited answers.<br><br>"
        "Requires an <code>OPENAI_API_KEY</code> in environment or "
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
            rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else str(rmse)
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
            r2_str = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
            rows_text.append(
                f"  {name} ({fam}): RMSE={rmse_str}, MAE={mae_str}, R2={r2_str}"
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
                "Rolling features: OT_rolling_mean_3 (3-step rolling mean, "
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
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        st.warning(
            "**OpenAI API key not found.** "
            "To enable the AI assistant, add your key to `.streamlit/secrets.toml`:\n\n"
            "```toml\nOPENAI_API_KEY = \"sk-...\"\n```\n\n"
            "or set the environment variable `OPENAI_API_KEY` before launching Streamlit."
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

    for i, qq in enumerate(quick_questions):
        col_idx = i % 3
        if quick_q_col[col_idx].button(qq, key=f"qq_{i}", use_container_width=True):
            st.session_state.chat_textarea = qq   # directly override the widget state
            st.session_state._auto_send = True

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
        height=80,
        placeholder="e.g. Which model is best for production deployment? Why does XGBoost outperform Holt-Winters?",
        key="chat_textarea",
    )
    _auto_send = st.session_state.get("_auto_send", False)
    st.session_state._auto_send = False  # consume the flag

    send_col, clear_col = st.columns([PHI, 1])
    send_clicked = send_col.button("Send", type="primary", use_container_width=True, disabled=(not api_key))
    clear_col.button(
        "Clear history",
        on_click=lambda: st.session_state.__setitem__("chat_history", []),
        use_container_width=True,
    )

    if (send_clicked or _auto_send) and user_query.strip() and api_key:
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
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history
                        ],
                    ],
                )
                answer = response.choices[0].message.content
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
