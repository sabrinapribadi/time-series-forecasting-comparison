"""Model, family and metric metadata shared by the dashboard, its knowledge base and the replay recorder."""
from __future__ import annotations

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
