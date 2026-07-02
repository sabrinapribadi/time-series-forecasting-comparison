"""
Streamlit dashboard — reads pre-computed forecast results from
data/forecasts/ETT/*.json (no model weights needed at runtime).

Tabs:
  1. Benchmark Results  — KPI row + metric comparison table + bar chart
  2. Forecast Gallery   — per-model overlay plots on the test set
  3. Model Inspector    — residuals and error distribution for a selected model
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FORECAST_DIR = Path(__file__).parent.parent.parent / "data" / "forecasts" / "ETT"

st.set_page_config(
    page_title="Time Series Forecasting Benchmark",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

@st.cache_data
def load_results() -> dict:
    results = {}
    for path in sorted(FORECAST_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        results[path.stem] = data
    return results


results = load_results()

if not results:
    st.warning(
        "No forecast results found in `data/forecasts/ETT/`. "
        "Run `python scripts/run_inference.py` first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Build metrics DataFrame
# ---------------------------------------------------------------------------

rows = []
for model_name, data in results.items():
    row = {"Model": model_name}
    row.update(data.get("metrics", {}))
    rows.append(row)

metrics_df = pd.DataFrame(rows).set_index("Model")
metric_cols = [c for c in ["RMSE", "MAE", "MAPE", "SMAPE", "MASE", "R2"] if c in metrics_df.columns]
best_model = metrics_df["RMSE"].idxmin() if "RMSE" in metrics_df.columns else "—"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")
selected_models = st.sidebar.multiselect(
    "Models", options=list(results.keys()), default=list(results.keys())
)
selected_metric = st.sidebar.selectbox("Primary metric", metric_cols, index=0)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Benchmark Results", "Forecast Gallery", "Model Inspector"])

# ---- Tab 1: Benchmark Results ----
with tab1:
    st.title("Time Series Forecasting Benchmark — ETT Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models evaluated", len(results))
    col2.metric("Best model (RMSE)", best_model)
    if best_model in metrics_df.index and "RMSE" in metrics_df.columns:
        col3.metric("Best RMSE", f"{metrics_df.loc[best_model, 'RMSE']:.4f}")
    if best_model in metrics_df.index and "MAE" in metrics_df.columns:
        col4.metric("Best MAE", f"{metrics_df.loc[best_model, 'MAE']:.4f}")

    st.subheader("Metric Comparison")
    st.dataframe(
        metrics_df[metric_cols].loc[selected_models].style.highlight_min(
            subset=[c for c in metric_cols if c != "R2"], color="lightgreen"
        ).highlight_max(
            subset=["R2"] if "R2" in metric_cols else [], color="lightgreen"
        ).format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader(f"{selected_metric} by Model")
    fig = go.Figure(
        go.Bar(
            x=metrics_df.loc[selected_models].index.tolist(),
            y=metrics_df.loc[selected_models, selected_metric].tolist(),
            marker_color="steelblue",
        )
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=selected_metric, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Forecast Gallery ----
with tab2:
    st.title("Forecast Gallery — Test Set Overlay")
    fig = go.Figure()
    first = True
    for model_name in selected_models:
        data = results[model_name]
        if "y_true" not in data or "y_pred" not in data:
            continue
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        x = list(range(len(y_true)))
        if first:
            fig.add_trace(go.Scatter(x=x, y=y_true, name="Ground Truth", line=dict(color="white", width=2)))
            first = False
        fig.add_trace(go.Scatter(x=x, y=y_pred, name=model_name, mode="lines"))

    fig.update_layout(
        xaxis_title="Time step",
        yaxis_title="Oil Temperature (°C)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Model Inspector ----
with tab3:
    st.title("Model Inspector — Residuals")
    selected = st.selectbox("Select model", selected_models)
    data = results.get(selected, {})

    if "y_true" in data and "y_pred" in data:
        y_true = pd.Series(data["y_true"])
        y_pred = pd.Series(data["y_pred"])
        residuals = y_true - y_pred

        col_a, col_b = st.columns(2)
        with col_a:
            fig_res = go.Figure(go.Scatter(y=residuals.tolist(), mode="lines", name="Residual"))
            fig_res.update_layout(title="Residuals over time", xaxis_title="Step", yaxis_title="Error", height=350)
            st.plotly_chart(fig_res, use_container_width=True)
        with col_b:
            fig_hist = go.Figure(go.Histogram(x=residuals.tolist(), nbinsx=40, name="Residual dist"))
            fig_hist.update_layout(title="Residual distribution", xaxis_title="Error", yaxis_title="Count", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Metrics")
        st.json(data.get("metrics", {}))
    else:
        st.info("No predictions found — run scripts/run_inference.py first.")
