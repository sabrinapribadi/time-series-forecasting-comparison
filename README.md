# Time Series Forecasting Model Comparison

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Darts](https://img.shields.io/badge/Darts-0.27-0099CC?style=for-the-badge)](https://unit8co.github.io/darts/)
[![Dataset](https://img.shields.io/badge/Dataset-ETT-00BFFF?style=for-the-badge)](https://github.com/zhouhaoyi/ETDataset)

## Why This Project

Practitioners choosing a forecasting model face a confusing landscape: statistical methods are fast and interpretable, ML models handle non-linearity well, and deep learning models promise the highest accuracy — but come with steep compute costs. This project runs all three families side-by-side on the same dataset under the same evaluation protocol, so the accuracy/complexity trade-offs are visible and reproducible.

A secondary goal is to mirror two IOH internal production pipelines (AOP2026 monthly data traffic forecasting and the IMPACT congestion forecasting project), bridging academic benchmarks and operational telecom forecasting practice.

## System Architecture

```mermaid
flowchart LR
    A[("ETT Dataset\ndata/raw/ETT/\nETTh1/h2/m1/m2\n17,420–69,680 rows")] --> B

    subgraph B ["Data Loading (src/data/ett_loader.py)"]
        B1["ETTLoader.load()\nparse_dates, sort"] --> B2["get_splits()\n70/10/20 chronological"]
        B2 --> B3["Feature Engineering\ntime sin/cos + lags\n+ rolling stats"]
    end

    B3 --> C

    subgraph C ["Training (scripts/train_model.py)"]
        C1["--mode univariate\n(OT history only)"]
        C2["--mode multivariate\n(+ HUFL/HULL/MUFL\n/MULL/LUFL/LULL)"]
        C3["--tune\nOptuna HPO\n8–50 trials"]
    end

    C --> D["src/models/\nstatistical.py · ml.py · deep_learning.py"]
    D --> E["src/evaluation/metrics.py\nRMSE MAE MAPE SMAPE R² MASE MDA Bias"]
    E --> F["data/forecasts/ETT/*.json\n(pre-computed — committed)"]
    F --> G["Streamlit Dashboard\nsrc/ui/dashboard.py\nBenchmark · Gallery · Inspector"]
```

## Models and Algorithms

### Statistical Family

These models work directly on the time series values. They are fast to fit, require no feature engineering, and are fully interpretable, but have limited capacity for complex non-linear patterns.

#### Holt-Winters (Exponential Smoothing)

Decomposes a series into level, trend, and seasonality components and updates each with exponential smoothing.

```
Level:      l_t = α(y_t − s_{t−m}) + (1−α)(l_{t−1} + b_{t−1})
Trend:      b_t = β(l_t − l_{t−1}) + (1−β) b_{t−1}
Seasonality: s_t = γ(y_t − l_{t−1} − b_{t−1}) + (1−γ) s_{t−m}
Forecast:   ŷ_{t+h} = l_t + h·b_t + s_{t+h−m(k+1)}
```

Where α ∈ [0,1] controls level smoothing, β trend smoothing, γ seasonality smoothing, m = seasonal period, k = ⌊(h−1)/m⌋.

#### ARIMA (AutoRegressive Integrated Moving Average)

Combines autoregression (AR), differencing (I) for stationarity, and a moving-average error model (MA). Order (p,d,q) is selected automatically by `auto_arima` with ADF stationarity test.

```
φ(B) ∇^d y_t = θ(B) ε_t

AR polynomial:  φ(B) = 1 − φ₁B − φ₂B² − ⋯ − φ_pB^p
MA polynomial:  θ(B) = 1 + θ₁B + θ₂B² + ⋯ + θ_qB^q
Differencing:   ∇^d = (1−B)^d   (d=1 removes linear trend)
```

Where B is the backshift operator (By_t = y_{t−1}) and ε_t ~ N(0, σ²).

#### Prophet (Facebook)

Additive decomposition with automatic changepoint detection for piecewise linear/logistic trend and Fourier-series seasonality. Handles holidays and missing data natively.

```
y(t) = g(t) + s(t) + h(t) + ε_t

Trend:      g(t) = k + (a(t)ᵀδ)t + (m + a(t)ᵀγ)   [piecewise linear]
Seasonality: s(t) = Σ_{n=1}^{N} (a_n cos(2πnt/P) + b_n sin(2πnt/P))   [Fourier]
Holidays:   h(t) = Z(t)κ   [indicator × effect]
```

Where P = period, N = Fourier order, δ = changepoint slopes, a(t) = indicator vector.

---

### Machine Learning Family

These models treat forecasting as supervised regression on tabular lag features. They benefit from exogenous covariates (multivariate mode) and handle non-linear relationships without explicit model specification.

#### Random Forest

Bootstrap-aggregated ensemble of regression trees. Each tree is trained on a bootstrap sample and a random subset of features; averaging B trees reduces variance.

```
ŷ(x) = (1/B) Σ_{b=1}^{B} f_b(x)

Var(ŷ) = σ²/B + (1 − 1/B) ρ σ²

where ρ = pairwise tree correlation (reduced by feature subsampling)
```

Higher B and lower ρ (more feature randomness) → lower prediction variance.

#### XGBoost (eXtreme Gradient Boosting)

Additive ensemble of regression trees fitted sequentially by minimising a regularised second-order Taylor approximation of the loss.

```
F_K(x) = Σ_{k=1}^{K} η f_k(x)

Objective: L^(K) = Σ_i [g_i f_K(x_i) + (1/2) h_i f_K(x_i)²] + Ω(f_K)

Regulariser: Ω(f) = γT + (λ/2)||w||²

g_i = ∂l(y_i, ŷ_i^(K−1))/∂ŷ   (first-order gradient)
h_i = ∂²l(y_i, ŷ_i^(K−1))/∂ŷ²  (second-order hessian)
```

Where T = number of leaves, w = leaf weights, η = learning rate, γ/λ = regularisation.

#### LightGBM (Light Gradient Boosting Machine)

Same gradient boosting objective as XGBoost, but uses **histogram-based** binning for split finding and **leaf-wise** (best-first) tree growth. Also uses GOSS and EFB for scalability.

```
Same objective as XGBoost; key differences:

Histogram split: bin features → O(#bins) vs O(n log n) exact
Leaf-wise growth: expand leaf with max loss reduction (not level-wise)
GOSS: keep large-gradient instances + random sample small-gradient ones
EFB: bundle mutually exclusive sparse features into single feature
```

Uses IOH AOP2026 production hyperparameters: `reg_alpha=0.1, reg_lambda=0.1, n_estimators=500`.

#### CatBoost

Gradient boosting with **ordered boosting** (prevents target leakage) and **oblivious trees** (symmetric structure for fast inference). Natively handles categorical features via target statistics.

```
Same gradient boosting objective.

Ordered boosting: for sample i, use only a permutation-prior subset σ_r(x_i)
                  to compute target statistics — prevents overfitting on training data

Oblivious tree: all nodes at depth d use the same split (feature, threshold)
               → O(2^depth) lookup table for inference
```

---

### Deep Learning Family

These models learn sequence representations directly from raw time series without manual feature engineering. They excel at long-range dependencies but require significantly more training time.

#### LSTM (Long Short-Term Memory)

Extends the vanilla RNN with gating mechanisms that learn to selectively retain or forget information over long sequences, mitigating the vanishing gradient problem.

```
Forget gate:  f_t = σ(W_f · [h_{t−1}, x_t] + b_f)
Input gate:   i_t = σ(W_i · [h_{t−1}, x_t] + b_i)
Candidate:    c̃_t = tanh(W_c · [h_{t−1}, x_t] + b_c)
Cell state:   c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t
Output gate:  o_t = σ(W_o · [h_{t−1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(c_t)
```

Implementation: `_LSTMNet(nn.Module)` with direct multi-step head `Linear(hidden, horizon)`. Evaluated with rolling-window oracle strategy (ground-truth context refreshed every `horizon` steps).

#### Transformer (Attention-based)

Replaces recurrence with scaled dot-product self-attention, enabling parallel computation over the entire input sequence and capturing global dependencies without distance bias.

```
Attention:      Attn(Q, K, V) = softmax(QKᵀ / √d_k) V
Multi-head:     MH(Q, K, V) = Concat(head₁, …, head_h) Wᴼ
                where head_i = Attn(Q W_i^Q, K W_i^K, V W_i^V)
FFN:            FFN(x) = max(0, xW₁ + b₁) W₂ + b₂
```

Each layer: `LayerNorm(x + Attn(x)) → LayerNorm(x + FFN(x))`. Implemented via Darts `TransformerModel`.

#### N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)

A pure MLP-based architecture organised as a stack of blocks, each producing a **backcast** (residual explanation) and **forecast** (contribution to final output). No recurrence, no convolution.

```
Block l output:
  θ_b^l, θ_f^l = FC(residual input x_l)
  backcast:  b_l = Σ_k θ_b^l · g_k^b    (basis expansion)
  forecast:  f_l = Σ_k θ_f^l · g_k^f

Stack: x_{l+1} = x_l − b_l
Final: ŷ = Σ_{l=1}^{L} f_l
```

With generic (data-driven) basis functions, or interpretable (trend/seasonality) bases via `GENERIC` and `SEASONALITY` stacks.

#### TFT (Temporal Fusion Transformer)

A multi-horizon architecture combining variable selection networks (VSN), gated residual networks (GRN), and multi-head self-attention. Produces interpretable attention weights over past time steps.

```
Variable Selection:  VSN → weighted feature embeddings per time step
Sequence encoding:   LSTM encoder over past obs → context vector c_s, c_e, c_h, c_c
Enrichment:          GRN(past) with c_e as context
Attention:           InterpretableMultiHead(Q=enriched, K=V=encoder output)
Output:              GLU → Add+Norm → Dense → quantile outputs
```

Where GLU = Gated Linear Unit: `GLU(a,b) = a ⊙ σ(b)`. Implemented via Darts `TFTModel` with `add_relative_index=True`.

---

## Benchmark Results — ETTh1 (Horizon = 24, Univariate, Default Hyperparameters)

> Note: ML models use oracle lag features (ground-truth OT lags as inputs on test set), giving them an advantage over statistical models that forecast the full 3,484-step test horizon in one shot. DL models ran for only 5–30 epochs (undertrained). See [docs/adr/ADR-005](docs/adr/ADR-005-feature-engineering.md) for the oracle lag rationale.

| Model | Family | RMSE ↓ | MAE ↓ | SMAPE (%) ↓ | R² ↑ |
|-------|--------|--------|-------|-------------|------|
| **LightGBM** | ML | **0.743** | 0.547 | 11.08 | **0.954** |
| **CatBoost** | ML | 0.744 | **0.531** | **11.11** | 0.953 |
| Random Forest | ML | 0.783 | 0.582 | 11.71 | 0.948 |
| XGBoost | ML | 0.847 | 0.628 | 12.45 | 0.940 |
| LSTM | Deep | 3.257 | 2.595 | 36.97 | 0.107 |
| Transformer | Deep | 3.912 | 3.097 | 42.39 | −0.289 |
| N-BEATS | Deep | 3.727 | 3.080 | 43.66 | −0.170 |
| ARIMA | Statistical | 4.138 | 3.446 | 50.74 | −0.442 |
| Prophet | Statistical | 4.030 | 3.244 | 47.76 | −0.368 |
| TFT | Deep | 9.097 | 7.980 | 72.32 | −5.969 |
| Holt-Winters | Statistical | 7.851 | 6.614 | 125.41 | −4.190 |

**Key findings:**
- **ML models dominate** because they use oracle lag features (OT_lag_1, OT_lag_2, OT_lag_24) — ground-truth past OT values — which are highly predictive for 24-step-ahead forecasting.
- **Statistical models underperform** because they forecast the full 3,484-step test horizon in one shot; autoregressive degradation compounds over thousands of steps.
- **DL models are undertrained** (5–30 epochs due to MPS/CPU compute limits); a proper 100+ epoch run on GPU is expected to bring LSTM/Transformer into the R² > 0.8 range.
- TFT underperforms because it requires future covariates which we approximate with a relative time index — insufficient without real future load data.

## Feature Engineering

Two modes controlled by `--mode` flag:

### Univariate (default)
Forecast OT from its own history. Features: cyclical time encodings + autoregressive lags.

| Feature | Formula | Purpose |
|---------|---------|---------|
| `hour_sin/cos` | sin/cos(2π·h/24) | Cyclical hour encoding |
| `dow_sin/cos` | sin/cos(2π·d/7) | Day-of-week seasonality |
| `month_sin/cos` | sin/cos(2π·(m−1)/12) | Annual seasonality |
| `OT_lag_1/2/24` | OT shifted 1,2,24 steps | Recent history for ML |
| `OT_rolling_mean_3` | rolling(3).mean().shift(1) | Short-term trend level |
| `OT_rolling_std_3` | rolling(3).std().shift(1) | Local volatility |
| `OT_growth_rate` | pct_change().shift(1) | Relative change |
| `OT_trend_3` | polyfit slope on last 3 points, shift(1) | Instantaneous slope |

### Multivariate (`--mode multivariate`)
Adds all 6 load columns as covariates for ML models — analogous to the `Volume_WD_GB` chained regressor in the IOH congestion pipeline. Expands feature matrix from 11 to 17 columns.

## Hyperparameter Tuning (Optuna)

IOH AOP2026 production trial counts, enabled via `--tune`:

| Model | Trials | Objective | Direction |
|-------|--------|-----------|-----------|
| Holt-Winters | 15 | RMSE on train residuals | Minimize |
| Random Forest | 10 | R² on validation | Maximize |
| CatBoost | 8 | R² on validation | Maximize |
| XGBoost | 50 | R² on validation | Maximize |
| LightGBM | — | Fixed IOH production params | — |

## Evaluation Metrics

| Metric | Formula | Best | Notes |
|--------|---------|------|-------|
| RMSE | √(mean((y−ŷ)²)) | Lower | Penalises large errors |
| MAE | mean(\|y−ŷ\|) | Lower | Robust to outliers |
| MAPE | mean(\|y−ŷ\|/\|y\|)×100 | Lower | Skips \|y\|<0.1 (ETT has negative OT) |
| SMAPE | mean(\|y−ŷ\|/((|y|+|ŷ|)/2))×100 | Lower | Symmetric; range [0,200] |
| MASE | MAE / MAE(naive) | <1 beats naive | Scale-free across variants |
| R² | 1−SS_res/SS_tot | Higher | 1.0 = perfect |
| MDA | mean(sign(Δy)==sign(Δŷ))×100 | Higher | Directional accuracy % |
| Bias | mean(ŷ−y) | ~0 | Systematic over/under-forecast |
| MAAPE | mean(arctan(\|y−ŷ\|/(\|y\|+ε)))×100 | Lower | Handles near-zero actuals |

## Architecture Decision Records

See [docs/adr/](docs/adr/README.md) for full decision log.

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](docs/adr/ADR-001-task-framing.md) | Task framing: Univariate first, multivariate Phase 2 | Accepted — Amended (multivariate now live) |
| [ADR-002](docs/adr/ADR-002-preprocessing.md) | Preprocessing: Cyclical encoding + lag + rolling features | Accepted |
| [ADR-003](docs/adr/ADR-003-validation-strategy.md) | Validation: Chronological 70/10/20 split | Accepted |
| [ADR-004](docs/adr/ADR-004-model-selection.md) | Model selection: Compare all 11 + Optuna HPO | Accepted |
| [ADR-005](docs/adr/ADR-005-feature-engineering.md) | Feature engineering: Oracle lag filling for ML models | Accepted |
| [ADR-006](docs/adr/ADR-006-deployment.md) | Deployment: Pre-computed JSON → Streamlit dashboard | Accepted |

## Project Structure

```
time-series-forecasting-comparison/
├── src/
│   ├── data/
│   │   └── ett_loader.py          # ETTLoader — load, split, feature engineering
│   │                              # mode='univariate'|'multivariate', rolling features
│   ├── models/
│   │   ├── statistical.py         # HoltWinters (+ Optuna), ARIMA, Prophet
│   │   ├── ml.py                  # RF, XGBoost, LightGBM (IOH params), CatBoost (+ Optuna)
│   │   └── deep_learning.py       # LSTM (PyTorch), Transformer/N-BEATS/TFT (Darts)
│   ├── evaluation/
│   │   └── metrics.py             # RMSE, MAE, MAPE, SMAPE, MASE, R², MDA, Bias, MAAPE
│   ├── ui/
│   │   └── dashboard.py           # Streamlit 3-tab dashboard (reads JSON only)
│   └── utils.py                   # setup_logging, load_config, save_results
├── scripts/
│   ├── download_data.py           # Download ETT CSVs from GitHub
│   ├── validate_data.py           # Schema + missing value checks
│   ├── train_model.py             # Train one model (--model, --variant, --mode, --tune)
│   └── run_inference.py           # Load checkpoint, re-run inference, save JSON
├── data/
│   ├── raw/ETT/                   # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
│   ├── processed/ETT/             # (gitignored)
│   └── forecasts/ETT/             # Per-model JSON results (committed — dashboard source)
├── configs/
│   ├── default_config.yaml        # Split ratios, horizon, lag_steps
│   └── optuna_configs.yaml        # Per-model Optuna search spaces + IOH trial counts
├── docs/
│   ├── PRD.md                     # Full product requirements + data definition
│   ├── ARCHITECTURE.md            # Mermaid diagrams + module descriptions
│   └── adr/                       # Architecture Decision Records (6 ADRs)
├── models/                        # Saved .joblib checkpoints (gitignored)
├── notebooks/
├── tests/
└── pyproject.toml                 # Poetry dependency spec
```

## Setup

**Prerequisites:** Python 3.10+, [Poetry](https://python-poetry.org/)

```bash
git clone https://github.com/sabrinapribadi/time-series-forecasting-comparison.git
cd time-series-forecasting-comparison
poetry install
```

## Data

The ETT dataset is already included at `data/raw/ETT/`. To validate:

```bash
poetry run python scripts/validate_data.py
```

## Training

```bash
# Train a single model (univariate, default)
poetry run python scripts/train_model.py --model xgboost --variant h1

# Multivariate mode — ML models receive all 6 load columns as covariates
poetry run python scripts/train_model.py --model catboost --variant h1 --mode multivariate

# With Optuna HPO (IOH production trial counts)
poetry run python scripts/train_model.py --model catboost --variant h1 --tune
poetry run python scripts/train_model.py --model holt_winters --variant h1 --tune

# Train all 11 models
make train-all

# Train ML models in multivariate mode
make train-multivariate

# Train tunable models with Optuna
make train-tuned
```

## Dashboard

Pre-computed JSON results in `data/forecasts/ETT/` power the dashboard — no model weights or retraining needed:

```bash
PYTHONPATH=. poetry run streamlit run src/ui/dashboard.py
```

Dashboard opens at `http://localhost:8501`.

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Statistical models | statsmodels, pmdarima, prophet |
| ML models | scikit-learn, xgboost, lightgbm, catboost |
| Deep learning | PyTorch 2.x (LSTM), Darts 0.27 (Transformer, N-BEATS, TFT) |
| HPO | Optuna 3.x |
| Data | NumPy, Pandas |
| Frontend | Streamlit 1.35+, Plotly |
| Metrics | scikit-learn, NumPy |
| Language | Python 3.10+ |
| Hardware | Apple MPS / CUDA / CPU (auto-detected) |
| Dependency mgmt | Poetry |

## Citation

```bibtex
@inproceedings{zhou2021informer,
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author    = {Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai
               and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2021}
}

@inproceedings{Oreshkin2020nbeats,
  title     = {N-BEATS: Neural basis expansion analysis for interpretable time series forecasting},
  author    = {Oreshkin, Boris N and Carpov, Dmitri and Chapados, Nicolas and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations},
  year      = {2020}
}

@inproceedings{lim2021temporal,
  title     = {Temporal Fusion Transformers for interpretable multi-horizon time series forecasting},
  author    = {Lim, Bryan and Arik, Sercan O and Loeff, Nicolas and Pfister, Tomas},
  journal   = {International Journal of Forecasting},
  year      = {2021}
}
```
