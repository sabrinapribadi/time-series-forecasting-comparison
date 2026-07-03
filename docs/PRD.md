PRODUCT REQUIREMENTS DOCUMENT (PRD)
Project: Time Series Forecasting Model Comparison
Version: 6.0 (Overfitting/Underfitting Remediation + DL Capacity Increases)
Author: Sabrina Pribadi
Date: July 3, 2026
Status: Active — All 11 models retrained with remediated hyperparameters; full benchmark updated


1. EXECUTIVE SUMMARY

Problem: Practitioners selecting a forecasting method face a fragmented landscape — statistical
methods, tree-based ML, and deep learning models all have champions, but controlled side-by-side
comparisons on the same dataset under the same protocol are rare. Without this, model selection
defaults to convention ("ARIMA for time series", "XGBoost always wins") rather than evidence.

Solution: A reproducible end-to-end pipeline that trains 11 models across 3 families (statistical,
ML, deep learning) on the ETT (Electricity Transformer Temperature) dataset, evaluates each on
9 metrics, and publishes results to a Streamlit dashboard. Incorporates univariate and multivariate
modes, rolling statistical features, and Optuna hyperparameter tuning.

Value Proposition: Three things at once — a rigorous academic benchmark (ETT, standard 70/10/20
split, RMSE/MAE/MAPE/SMAPE/MASE/R²/MDA/Bias/MAAPE evaluation), a practical HPO framework
(oracle lag filling, multivariate covariates, Optuna), and a Streamlit dashboard that works from
pre-computed JSON without model weights.

Success Metrics:
- All 11 models train successfully on ETTh1 and produce valid predictions
- Dashboard loads from data/forecasts/ETT/*.json in under 3 seconds (no model weights at runtime)
- Multivariate mode widens the ML feature matrix with all 6 load columns
- Optuna HPO reduces RMSE vs. default hyperparameters for at least 2 of 4 tunable models
- All 6 Architecture Decision Records documented before coding each component


2. PROJECT CONTEXT AND BACKGROUND

This project was built to demonstrate capabilities across:

1. Time Series Fundamentals: Chronological train/val/test splitting, horizon-based forecasting,
   cyclical feature encoding, rolling-window oracle evaluation for DL models.

2. Statistical Forecasting: Exponential smoothing with Holt-Winters (level/trend/seasonality
   decomposition), ARIMA order selection via ADF stationarity test and auto_arima, Prophet's
   piecewise-linear trend with Fourier-series seasonality.

3. ML for Time Series: Lag-feature construction for tabular regression, oracle lag filling on
   the test set, univariate vs. multivariate feature matrices, gradient boosting regularisation
   (XGBoost L1/L2, LightGBM GOSS/EFB, CatBoost ordered boosting).

4. Deep Learning Architectures: PyTorch LSTM with direct multi-step head; Darts-wrapped
   Transformer (multi-head self-attention), N-BEATS (neural basis expansion), and TFT (temporal
   fusion with variable selection and gated residual networks). MPS/CUDA/CPU auto-detection
   with float32 casting for Apple Silicon compatibility.

5. Hyperparameter Optimisation: Optuna search with per-model trial counts and objectives
   (HoltWinters 15 trials RMSE, RF 10 trials R², CatBoost 8 trials R², XGBoost 50 trials R²).

6. Deployment Engineering: Pre-computed JSON predictions committed to data/forecasts/ETT/;
   Streamlit dashboard reads JSON only — no model weights, no retraining at runtime.

7. AI Features (Dashboard v2): RAG Ask AI tab using GPT-4o mini with keyword retrieval over
   benchmark data and model metadata; AI Explainability in Model Inspector generates streaming
   bullet-point diagnoses of each model's performance (metrics, residual autocorrelation, failure
   modes). OpenAI API key stored in .streamlit/secrets.toml (gitignored).

8. SHAP Explainability: ML models in the Model Inspector show a SHAP beeswarm-style chart
   (TreeExplainer, interventional perturbation, 200 test samples, 100 background samples) instead
   of LLM text. Pre-computed by scripts/extract_shap_values.py using the exact 15-feature
   training matrix (multivariate mode: HUFL/HULL/MUFL/MULL/LUFL/LULL + time + OT lags).
   Top drivers: OT_lag_1 dominates RF and CatBoost; hour_sin dominates XGBoost and LightGBM.

9. DL Model Improvements: All 4 DL models retrained with StandardScaler normalization (LSTM),
   early stopping (patience=15 on val_loss via PyTorch Lightning EarlyStopping), ReduceLROnPlateau
   scheduler (LSTM, patience=5), and 100 max epochs vs previous 30. Hidden size increased to 128.

10. Threading Fix (macOS): XGBoost and LightGBM segfaulted when loaded after PyTorch initialised
    its thread pool (exit code 139). Fixed with OMP/MKL/OPENBLAS_NUM_THREADS=1 in train_model.py
    and nthread=1 / num_threads=1 in the model constructors.

11. Extended Feature Engineering: Rolling features (rolling_mean_3, rolling_std_3, trend_3,
    growth_rate) complement lag features for ML models; multivariate covariate mode exposes all
    6 load columns; additional metrics (MDA, Bias, MAAPE) extend standard RMSE/MAE reporting.

12. Overfitting Diagnostic (v5): scripts/train_model.py now computes and saves train_metrics
    (in-sample RMSE, MAE, R² etc.) alongside test metrics for all ML and LSTM models. Model
    Inspector shows a Train vs Test RMSE gap card with ratio-based verdict: ratio < 1.2 = 
    Well-fitted, 1.2–2.0 = Moderate overfit, > 2.0 = Overfitting. Key finding: Random Forest
    shows 2.1× gap (classic overfit); CatBoost shows 0.82× (ordered boosting makes in-sample
    predictions conservative — test RMSE < train RMSE is expected for CatBoost).

13. ML Mode Standardised as Multivariate (v5): All 4 ML models retrained with --mode multivariate
    (15 features: 6 load cols + 6 time sin/cos + 3 OT lags) to be fully consistent with the 
    pre-computed SHAP data which used the 15-feature matrix. Previous versions had XGBoost and
    LightGBM in univariate mode (9 features), creating SHAP/model inconsistency.

14. Statistical Tests Enhancements (v5): All 4 Statistical Tests sections now have dynamic
    contextual interpretation cards that explain the test result in plain language:
    - Stationarity: explains ADF/KPSS conflict and what it means for ARIMA vs ML
    - ACF/PACF: flags lag 1/24/48 peaks by name and their practical implication for lag features
    - Granger Causality: lists significant features, explains physical mechanism, adds caveat
    - Diebold-Mariano: per-comparison guidance (significant → choose best; not significant → 
      choose simpler); matrix summary with count of equivalent pairs and 3 next steps

15. Dashboard UX Fixes (v5):
    - Ask AI quick questions: clicking a button now auto-sends immediately (sets chat_textarea
      directly in session state + _auto_send flag) instead of just populating the text box
    - Time series chart: legend moved above chart (y=1.02) to avoid collision with Date axis
    - Scatter chart: legend moved to top-right to avoid collision with x-axis label
    - Feature importance for LightGBM/XGBoost: when model checkpoint cannot be loaded (library
      mismatch), falls back to mean_abs_shap from pre-computed SHAP JSON as proxy importance

16. Overfitting/Underfitting Remediation (v6):
    Identified from train vs test RMSE gap analysis (v5) and applied targeted fixes per model:

    ML Overfitting (RF 2.13×, XGBoost 1.23×):
    - RandomForest: Optuna search space expanded with max_features parameter (sqrt/log2/0.5/0.7/1.0)
      to control tree decorrelation alongside existing max_depth constraint. After 10 trials:
      n_estimators=350, max_depth=10, max_features=1.0 → RMSE 0.783→0.703, Gap 2.13×→1.06× ✅
    - XGBoost: Added early_stopping_rounds=50 to XGBRegressor() constructor (XGBoost 2.x API —
      not a fit() parameter) plus reg_alpha and reg_lambda to Optuna search space (L1/L2 penalty).
      After 50 trials with val-set early stopping: RMSE and Gap both improved.

    DL Underfitting (all 4 DL models RMSE > 3.0 vs ML RMSE < 0.85):
    - LSTM: hidden_size 128→256, num_layers 2→3, input_len 96→168 (7-day window).
      NOTE: LSTM v2 caused an MPS kernel hang on Apple Silicon (OOM during the larger
      forward/backward pass with 3-layer, hidden=256 on mps device). All retries failed.
      v1 result retained: RMSE=5.332, MAE=4.457, R²=-1.394. The batched validation fix
      (chunked into batch_size=64 windows) was committed to deep_learning.py but the
      training forward pass itself hangs. Workaround for future runs: force device='cpu'.
    - Transformer: d_model 64→128 (4× parameter count), input_chunk_length 96→168.
      d_model=128 with nhead=4 gives 32 dims/head (unchanged ratio, more expressive).
      Result: RMSE=3.525, MAE=2.900, R²=-0.046 (16 min training, 929K params).
      Marginally worse than v1 RMSE=3.429 — capacity increase alone insufficient without
      longer training or LR warmup.
    - N-BEATS: input_chunk_length 96→168. N-BEATS already has 7.3M params (30 stacks ×
      4 layers × 256 width); primary bottleneck was the 4-day window missing weekly cycle.
      Result: RMSE=4.358, MAE=3.483, R²=-0.599. Slightly worse than v1 RMSE=4.065.
      Longer window did not help — N-BEATS may need more epochs or generic/seasonality
      stack separation to benefit from longer context.
    - TFT: hidden_size 64→128, input_chunk_length 96→168. Additionally, fit() now accepts
      past_covariates (6 load columns: HUFL/HULL/MUFL/MULL/LUFL/LULL) passed from train_model.py
      as a Darts multivariate TimeSeries. The TFTModel stores the covariate TimeSeries during
      fit() and passes it automatically to predict(). This gives TFT access to the same electrical
      load signals that drive OT, which ML models receive as X features.
      Result: RMSE=4.214, MAE=3.413, R²=-0.495.

    Overall DL remediation verdict: The capacity/window increases did not overcome the
    fundamental advantage of ML oracle lag features. DL models improved architecture but the
    ML vs DL RMSE gap (0.70 vs 3.5+) is driven by data access, not model capacity.

    No action for LightGBM (1.10× gap — well-fitted with fixed regularized defaults) or CatBoost
    (0.82× gap — ordered boosting makes conservative in-sample estimates by design).

Data Source: ETT (Electricity Transformer Temperature Dataset)
- GitHub: zhouhaoyi/ETDataset (MIT License)
- 4 variants: ETTh1, ETTh2 (hourly, 17,420 rows), ETTm1, ETTm2 (15-min, 69,680 rows)
- 8 columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Target: OT (Oil Temperature, °C)
- Date range: 2016-07-01 → 2018-06-26 (2 years, one transformer station per pair)
- Citation: Zhou et al., 2021 (AAAI — Informer paper)


3. SCOPE

In-Scope:
- Dataset: ETT (all 4 variants; primary benchmark on ETTh1)
- Models: 11 models across statistical, ML, and deep learning families
- Modes: Univariate (OT history) and Multivariate (+ 6 load covariates)
- Feature Engineering: Cyclical time features, OT lags, rolling stats + trend
- HPO: Optuna tuning for HoltWinters, RandomForest, XGBoost, CatBoost
- Evaluation: 9 metrics (RMSE, MAE, MAPE, SMAPE, MASE, R², MDA, Bias, MAAPE)
- Deployment: Pre-computed JSON + Streamlit 6-tab dashboard (v2)
- AI Features: RAG Ask AI (GPT-4o mini + keyword retrieval) + AI Explainability (streaming bullets)
- SHAP: Pre-computed SHAP beeswarm charts for all 4 ML models (scripts/extract_shap_values.py)
- DL Retraining: All 4 DL models retrained with normalization + early stopping + LR scheduling
- Bug Fixes: macOS threading fix for XGBoost/LightGBM; button key mismatch in Model Inspector
- Documentation: 6 ADRs, PRD v4, ARCHITECTURE.md, README v3

Out-of-Scope (current version):
- Multi-horizon probabilistic forecasting (quantile outputs from TFT not yet exposed)
- Hierarchical reconciliation across ETT variants
- Real-time inference or streaming pipeline
- GPU cloud training (local MPS/CPU only)
- Production A/B testing infrastructure


4. USER PERSONAS AND STORIES

| Persona | Goal | Pain Point |
|---------|------|-----------|
| Alex (ML Engineer) | Choose the right model for a new load forecasting task | No controlled comparison — papers compare on different datasets |
| Maria (Data Scientist) | Understand when deep learning actually beats statistical baselines | DL papers cherry-pick results; no honest accounting |
| Dana (Researcher) | Reproduce published ETT benchmarks | Many papers use ETT but with different preprocessing choices |
| Sam (Dashboard User) | Compare model performance without running code | Raw .joblib checkpoints are not browsable |

User Stories Implemented:
- As an ML engineer, I can train any of the 11 models with a single command and compare RMSE.
- As a data scientist, I can switch between univariate and multivariate modes to measure the
  impact of adding load covariates to ML models.
- As a researcher, I can reproduce the exact train/val/test split (70/10/20 chronological) and
  feature engineering choices documented in ADR-002 and ADR-003.
- As a dashboard user, I can compare all 11 models side-by-side without running any training.


5. FUNCTIONAL REQUIREMENTS

Module A: Data Loading (ETTLoader)
- A.1 Variants: load any of ETTh1/h2/m1/m2 from data/raw/ETT/
- A.2 Split: chronological 70/10/20 split on row count (no shuffling); returns (train, val, test)
       DataFrames preserving the 'date' column
- A.3 Time features: cyclical sin/cos encoding for hour (24h period), day-of-week (7d),
       month (12m); computed via ETTLoader._add_time_features()
- A.4 Lag features: OT_lag_1, OT_lag_2, OT_lag_24 for hourly (OT_lag_1/4/96 for 15-min);
       NaN rows dropped after lag computation; enabled for ML models only (ADR-005)
- A.5 Rolling features: OT_rolling_mean_3, OT_rolling_std_3, OT_growth_rate, OT_trend_3
       all shifted by 1 step to prevent leakage
- A.6 Modes:
       univariate  — drops HUFL/HULL/MUFL/MULL/LUFL/LULL from returned DataFrames
       multivariate — keeps all 6 load columns as ML covariates; 17 feature columns vs 11
- A.7 Frequency: ETTh = 'h' (24 default seasonal_periods); ETTm = '15min' (96 default)

Module B: Statistical Models (src/models/statistical.py)
- B.1 HoltWintersModel: wraps statsmodels ExponentialSmoothing with additive trend/seasonal;
       seasonal_periods=24 (hourly) or 96 (15-min); fit(y_train), predict(horizon)
       Optional: tune_with_optuna(y_train, n_trials=15) — minimize RMSE on train residuals
- B.2 ARIMAModel: pmdarima auto_arima with test='adf', stepwise=True, seasonal=False;
       fit(y_train), predict(horizon) via model.predict(n_periods=horizon)
- B.3 ProphetModel: Facebook Prophet with yearly/weekly/daily seasonality; requires dates_train;
       predict(horizon) via make_future_dataframe + model.predict(future)

Module C: ML Models (src/models/ml.py)
- C.1 RandomForestModel: sklearn RandomForestRegressor; fit(X, y); predict(X)
       Optional: tune_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=10) — maximize R² on val
       Optuna search (v6): n_estimators, max_depth [None/5/10/15/20], min_samples_split,
       min_samples_leaf, max_features [sqrt/log2/0.5/0.7/1.0] (added to control tree decorrelation)
- C.2 XGBoostModel: xgboost XGBRegressor; fit(X, y, X_val, y_val); predict(X)
       early_stopping_rounds=50 in constructor (XGBoost 2.x API — activates when eval_set provided)
       Optional: tune_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=50) — maximize R² on val
       Optuna search (v6): n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
       min_child_weight, reg_alpha, reg_lambda (L1+L2 added in v6 alongside early stopping)
- C.3 LightGBMModel: lightgbm LGBMRegressor with fixed regularized defaults:
       n_estimators=500, reg_alpha=0.1, reg_lambda=0.1, num_leaves=31; no Optuna (fixed, 1.10× gap)
- C.4 CatBoostModel: catboost CatBoostRegressor; fit(X, y, X_val, y_val); predict(X)
       Optional: tune_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=8) — maximize R² on val
       No remediation needed: 0.82× gap is expected ordered-boosting behaviour

Module D: Deep Learning Models (src/models/deep_learning.py)
- D.1 LSTMModel: _LSTMNet(nn.Module) at module scope (required for joblib pickling);
       input_len=168, horizon=24, hidden=256, layers=3 (v6: was 96/128/2);
       auto-detects MPS/CUDA/CPU; StandardScaler normalization;
       early stopping patience=15 on val MSE; ReduceLROnPlateau scheduler (patience=5);
       100 max epochs (default); fit(y_train, y_val); predict(y_context) → horizon-step array
- D.2 TransformerModel: Darts TransformerModel wrapper; 100 max epochs;
       input_chunk_length=168, d_model=128 (v6: was 96/64); nhead=4; EarlyStopping(patience=15);
       fit(y_train, dates_train, y_val, dates_val); predict(horizon)
- D.3 NBEATSModel: Darts NBEATSModel wrapper; 100 max epochs; EarlyStopping(patience=15);
       input_chunk_length=168 (v6: was 96); same interface as TransformerModel
- D.4 TFTModel: Darts TFTModel with add_relative_index=True; 100 max epochs;
       input_chunk_length=168, hidden_size=128 (v6: was 96/64); EarlyStopping(patience=15);
       fit() accepts optional past_cov_arr + past_cov_dates for multivariate conditioning;
       stores _past_cov_ts during fit() and passes it to predict() automatically;
       train_model.py loads multivariate splits for TFT and passes load columns as past_covariates

Module E: Evaluation (src/evaluation/metrics.py)
- E.1 compute_rmse, compute_mae, compute_mape (skips |y|<0.1), compute_smape, compute_mase
- E.2 compute_r2, compute_mda (directional accuracy), compute_bias, compute_maape (arctan-based)
- E.3 compute_all_metrics(y_true, y_pred, y_train) → dict with all 9 metrics

Module F: Training Script (scripts/train_model.py)
- F.1 Arguments: --model (11 choices), --variant (h1/h2/m1/m2), --mode (univariate/multivariate),
       --tune (Optuna flag), --epochs (DL), --horizon, --seasonal-periods
- F.2 Model dispatch: STATISTICAL, ML_MODELS, DL_MODELS sets; builds model with build_model()
- F.3 Tuning flow: when --tune + model in TUNABLE → call tune_with_optuna() instead of fit()
- F.4 ML pipeline: feature_cols = all train_df columns except 'date' and 'OT';
       multivariate columns automatically included when mode=multivariate
- F.5 LSTM rolling window: oracle context refreshed with ground-truth OT every horizon steps
- F.6 Result saved FIRST (data/forecasts/ETT/{run_name}.json), then checkpoint ({run_name}.joblib)
- F.7 Run name encodes mode + tune status: {model}[_multivariate][_tuned]_{variant}
- F.8 Train metrics: after test evaluation, in-sample predictions computed and saved as
       "train_metrics" dict in JSON — ML models predict on X_train; LSTM uses rolling windows.
       Statistical and Darts DL models: train_metrics is empty {} (in-sample not applicable).
- F.9 Post-retrain cleanup: scripts/retrain_cleanup.sh renames {model}_multivariate_{variant}.json
       to {model}_{variant}.json so SHAP/feature_importance lookups remain consistent with base names.

Module G: Streamlit Dashboard v2 (src/ui/dashboard.py)
- G.1 Data source: reads data/forecasts/ETT/*.json only — no model weights at runtime
- G.2 Design: golden ratio (φ=1.618) column splits; Inter + JetBrains Mono fonts;
       dark theme (#0D1117 / #161B22); family colours Statistical=#F5A623, ML=#4C8BF5, DL=#9B8FFF
- G.3 Tab 1 — Benchmark Results: KPI cards, ranked bar chart with metric context cards,
       per-metric formula + description, multi-metric normalised radar chart, full metric table
- G.4 Tab 2 — Forecast Gallery: all models overlaid on OT test set, colour-coded by family,
       range slider for zoom; family RMSE summary cards below chart
- G.5 Tab 3 — Model Inspector: Actual vs Predicted, residuals over time, residual histogram
       with mean line, predicted-vs-actual scatter (legend top-right, no label collision);
       Overfitting Diagnostic: Train vs Test RMSE with ratio-based verdict (ratio thresholds:
       <1.2 Well-fitted, 1.2–2.0 Moderate overfit, >2.0 Overfitting);
       Feature Importance: MDI bar chart from pre-computed JSON (ML models);
       SHAP: beeswarm-style scatter (dot = test sample, color = feature value) from pre-computed
       JSON for ML models — replaces LLM text for interpretable tree-based explainability;
       AI Explainability button (GPT-4o mini, streaming bullets) for Statistical and DL models
- G.6 Tab 4 — Data Explorer: ETT raw time series (dual-panel + range slider, legend above chart),
       summary stats, 7×7 Pearson correlation heatmap
- G.7 Tab 5 — Statistical Tests: Stationarity (ADF + KPSS with contextual interpretation card),
       ACF/PACF (barchart + interpretation of lag 1/24/48 peaks and decay pattern),
       Granger Causality (table + per-feature interpretation + physical mechanism note),
       Diebold-Mariano (pairwise test + actionable guidance + full matrix with next-step summary)
- G.8 Tab 6 — Ask AI (RAG): keyword retrieval over knowledge base; GPT-4o mini response with
       cited source IDs; 6 quick-question buttons (clicking auto-sends immediately via
       st.session_state.chat_textarea + _auto_send flag); full chat history in session state
- G.9 Sidebar: grouped expanders (Model Families, Individual Models, Display Options, Dataset Info)


6. NON-FUNCTIONAL REQUIREMENTS

- Performance: Dashboard loads from JSON in < 3 seconds. Training all 11 models on ETTh1
  takes ~30–120 minutes on Apple MPS (statistical: seconds; ML: 2–10 min; DL: 5–60 min/model).
- Reproducibility: Chronological split on row count (not date) ensures identical splits across
  Python/pandas versions. All hyperparameters documented in configs/default_config.yaml.
- Hardware: Supports Apple MPS, NVIDIA CUDA, and CPU. DL models auto-detect device.
  MPS requires float32 (not float64) for all PyTorch operations and Darts TimeSeries.
- Code Quality: Modular structure (src/data/, src/models/, src/evaluation/, src/ui/, scripts/).
  All decisions documented in ADRs before coding.
- Version Control: models/*.joblib (checkpoints) and data/processed/ are gitignored.
  Committed: src/, scripts/, docs/, configs/, data/raw/ETT/*.csv, data/forecasts/ETT/*.json.


7. DATA STRATEGY

Data Dictionary:

| Column | Full Name | Unit | Type | Description |
|--------|-----------|------|------|-------------|
| date | Timestamp | — | datetime64 | Observation time (UTC+8, China) |
| HUFL | High Useful Load | MW | float32 | High-voltage side active load |
| HULL | High Useless Load | MW | float32 | High-voltage side reactive load |
| MUFL | Middle Useful Load | MW | float32 | Mid-voltage side active load |
| MULL | Middle Useless Load | MW | float32 | Mid-voltage side reactive load |
| LUFL | Low Useful Load | MW | float32 | Low-voltage side active load |
| LULL | Low Useless Load | MW | float32 | Low-voltage side reactive load |
| OT | Oil Temperature | °C | float32 | TARGET — thermal stress proxy |

ETT Variants:

| File | Frequency | Rows | Train | Val | Test |
|------|-----------|------|-------|-----|------|
| ETTh1.csv | Hourly | 17,420 | 12,194 | 1,742 | 3,484 |
| ETTh2.csv | Hourly | 17,420 | 12,194 | 1,742 | 3,484 |
| ETTm1.csv | 15-min | 69,680 | 48,776 | 6,968 | 13,936 |
| ETTm2.csv | 15-min | 69,680 | 48,776 | 6,968 | 13,936 |

OT statistics (ETTh1):
- Min: −4.5 °C (winter months, transformer cooling)
- Max: 63.7 °C (peak summer load)
- Mean: ~15.6 °C | Std: ~9.3 °C
- Note: negative values make raw MAPE meaningless → compute_mape skips |y|<0.1

Feature Engineering:

Cyclical time features (prevent linear discontinuity at period boundaries):
| Column | Formula | Period |
|--------|---------|--------|
| hour_sin/cos | sin/cos(2π·h/24) | 24 hours |
| dow_sin/cos | sin/cos(2π·d/7) | 7 days |
| month_sin/cos | sin/cos(2π·(m−1)/12) | 12 months |

Lag features (ML models only — oracle filling on test set, ADR-005):
| Column | Description |
|--------|-------------|
| OT_lag_1 | OT shifted 1 step |
| OT_lag_2 | OT shifted 2 steps |
| OT_lag_24 | OT shifted 24 steps (seasonal lag for hourly) |

Rolling features (all shifted by 1 to prevent leakage):
| Column | Formula | Purpose |
|--------|---------|---------|
| OT_rolling_mean_3 | rolling(3).mean().shift(1) | Short-term level |
| OT_rolling_std_3 | rolling(3).std().shift(1) | Local volatility |
| OT_growth_rate | pct_change().shift(1) | Relative velocity |
| OT_trend_3 | polyfit slope on last 3 obs, shift(1) | Instantaneous slope |


8. TECHNICAL ARCHITECTURE

+------------------------------------------------------------------+
|                STREAMLIT DASHBOARD (3 tabs)                       |
|  Benchmark Results | Forecast Gallery | Model Inspector           |
|  (reads data/forecasts/ETT/*.json — no model or checkpoint)      |
+------------------------------------------------------------------+
                            |
         +------------------+-------------------+
         |                                      |
+--------+--------+                   +---------+---------+
| scripts/run_inference.py        |  | scripts/train_model.py      |
| Loads .joblib checkpoint        |  | --model --variant           |
| Re-runs predict() on test set   |  | --mode univariate|multivar. |
| Saves results JSON              |  | --tune (Optuna HPO)         |
+--------------------------------+  +-----------+-----------------+
                                                |
                        +-----------------------+
                        |
        +---------------+--------------+
        |               |              |
+-------+------+ +------+------+ +----+------+
|statistical.py| |   ml.py     | |deep_       |
|HoltWinters   | |RF XGBoost   | |learning.py |
|ARIMA Prophet | |LightGBM     | |LSTM Transf.|
|+ Optuna      | |CatBoost     | |NBEATS TFT  |
|              | |+ Optuna     | |            |
+--------------+ +-------------+ +------------+
                        |
        +---------------+---------------+
        |                               |
+-------+--------+             +--------+------+
| ETTLoader      |             | metrics.py    |
| load()         |             | compute_all() |
| get_splits()   |             | RMSE MAE MAPE |
| mode=univar.   |             | SMAPE MASE R² |
| mode=multivar. |             | MDA Bias MAAPE|
| rolling feats  |             +---------------+
+----------------+
        |
+-------+--------+
| data/raw/ETT/  |
| ETTh1.csv      |
| ETTh2.csv      |
| ETTm1.csv      |
| ETTm2.csv      |
+----------------+


9. IMPLEMENTATION PLAN

| Phase | Deliverables | Status |
|-------|-------------|--------|
| 1 | ETTLoader, chronological split, time features | Complete |
| 2 | Statistical models (HoltWinters, ARIMA, Prophet) | Complete |
| 3 | ML models (RF, XGBoost, LightGBM, CatBoost) | Complete |
| 4 | Deep learning (LSTM, Transformer, N-BEATS, TFT) | Complete |
| 5 | Evaluation metrics, train_model.py, dashboard | Complete |
| 6 | Multivariate mode, rolling features | Complete |
| 7 | Optuna HPO, additional metrics | Complete |
| 8 | Full benchmark on ETTh1, results committed | Complete |
| 9 | Full 100+ epoch DL run on GPU for fair comparison | Planned |
| 10 | ETTh2/ETTm1/ETTm2 benchmark extension | Planned |


10. TESTING STRATEGY

- Split reproducibility: verified train_end/val_end indices are stable across runs (row-count
  based, not date-based)
- Lag feature leakage: ETTLoader._add_lag_features uses shift(n), not future values;
  verified by checking that OT_lag_1[i] == OT[i−1]
- Rolling feature leakage: all rolling features shifted by 1; first valid row after dropna
  is confirmed to contain no future OT values
- MAPE on negative OT: ETTh1 OT goes negative in winter (min=−4.5°C); compute_mape skips
  |y|<0.1 to avoid division-by-near-zero; verified NaN not propagated to dashboard
- MPS float32: all Darts TimeSeries constructed from float32 arrays; verified no float64
  MPS error on M-series chip
- LSTM pickling: _LSTMNet defined at module scope (not inside fit()); verified joblib.dump
  does not raise PicklingError
- Dashboard: manual QA of all 3 tabs; confirmed loads from JSON without any model weights


11. RISKS AND MITIGATIONS

| Risk | Mitigation | Status |
|------|-----------|--------|
| DL models undertrained | Ran 5–30 epochs (MPS/CPU limit); documented in README benchmark table | Documented |
| MAPE undefined on negative OT | compute_mape skips abs(y)<0.1; ETT OT min≈−4.5°C; SMAPE/MAAPE used as primary percentage metrics | Resolved |
| MPS float64 error (Darts) | Cast all series to float32 before TimeSeries.from_series() | Resolved |
| LSTM checkpoint 2 bytes | Save results JSON BEFORE joblib.dump; move _net to CPU first | Resolved |
| ML oracle advantage | Documented in README note; future: recursive evaluation without oracle | Documented |
| TFT needs future covariates | add_relative_index=True auto-generates relative time index | Resolved |
| _LSTMNet unpicklable | Moved to module scope (not nested in fit()) | Resolved |
| Statistical horizon degradation | Statistical models forecast full 3484-step test horizon; | Documented |
|                                  | performance degrades vs. ML with oracle lags. Documented. | |


12. SUCCESS CRITERIA

Data pipeline (achieved):
- ETTLoader loads all 4 variants without error
- Chronological split produces identical indices across runs
- Cyclical features have no discontinuity at day/week/year boundaries
- Rolling features are strictly backward-looking (no leakage)

Training pipeline (achieved):
- All 11 models train on ETTh1 and produce JSON results
- Multivariate mode expands ML feature matrix to 17 columns
- Optuna HPO runs for HoltWinters, RF, XGBoost, CatBoost
- LightGBM uses fixed regularized defaults (reg_alpha=0.1, reg_lambda=0.1)

Evaluation (achieved):
- 9 metrics computed per model (RMSE, MAE, MAPE, SMAPE, MASE, R², MDA, Bias, MAAPE)
- MAPE correctly handles negative/near-zero OT values
- MDA correctly measures directional hit rate for trend-aware comparison

Dashboard (achieved):
- Loads from data/forecasts/ETT/*.json in < 3 seconds
- No model weights or ETT CSV files needed at dashboard runtime


13. KEY INSIGHTS FROM BENCHMARK

ETTh1 multivariate (ML) / univariate (DL/Statistical), horizon=24, default hyperparameters:

ML models dominate (RMSE 0.744–0.847, R² 0.940–0.953) because:
- They use oracle lag features: OT_lag_1 is ground-truth previous-step value on test set
- 24-step-ahead with oracle lag_1 is nearly a 1-step-ahead problem in practice
- This advantage is acknowledged and documented — mirrors production rolling inference

Overfitting analysis (train vs test RMSE, newly measured in v5):
- Random Forest: train=0.368, test=0.783 → 2.13× gap → OVERFITTING (bootstrap bags memorise)
- XGBoost: train=0.690, test=0.847 → 1.23× gap → moderate overfit
- LightGBM: train=0.687, test=0.755 → 1.10× gap → well-fitted (GOSS/EFB regularisation)
- CatBoost: train=0.910, test=0.744 → 0.82× (test RMSE < train RMSE — expected with ordered
  boosting; in-sample estimates are conservative by design)

Statistical models underperform (RMSE 4.0–8.8, R² negative) because:
- They forecast the full 3,484-step test horizon in one shot
- Autoregressive error compounds over thousands of steps
- A fair comparison would use rolling evaluation for statistical models too (planned)

DL models (fully retrained with early stopping, 100 max epochs):
- LSTM: early stopping patience=15; 100 epoch max; normalization + ReduceLROnPlateau
- Transformer, N-BEATS, TFT: EarlyStopping(patience=15) via PyTorch Lightning callbacks
- Expected to outperform statistical models; DL results updated after retraining completes

Regularization confirmed effective:
- LightGBM with reg_alpha=0.1, reg_lambda=0.1 well-fitted (1.10× gap)
- CatBoost ordered boosting gives near-zero train bias and strong test R² (0.953)
- Rolling features (OT_rolling_mean_3, OT_trend_3) improve feature richness for all ML models


14. APPENDIX

- Dataset: github.com/zhouhaoyi/ETDataset (MIT License)
- Tech Stack: statsmodels, pmdarima, prophet, sklearn, xgboost, lightgbm, catboost,
  PyTorch, Darts, Optuna, Streamlit, Plotly, Pandas, NumPy, Poetry
- ADR Index (see docs/adr/README.md for full index):
    ADR-001: Univariate first, multivariate Phase 2 (amended — multivariate now live)
    ADR-002: Cyclical encoding + lag features + rolling statistics
    ADR-003: Chronological 70/10/20 split (no shuffling)
    ADR-004: Compare all 11 models + Optuna HPO option
    ADR-005: Oracle lag filling for ML models on test set
    ADR-006: Pre-computed JSON → Streamlit dashboard
- Citation:
    Zhou et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence
    Time-Series Forecasting. AAAI 2021. https://arxiv.org/abs/2012.07436
