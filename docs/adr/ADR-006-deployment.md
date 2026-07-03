# ADR-006: Deployment — Pre-Computed JSON + Streamlit Dashboard v3 + RAG / AI Explainability

**Status:** Accepted (updated 2026-07-03 — Dashboard v3, Remediation v6)
**Date:** 2026-06-25 | **Last Updated:** 2026-07-03  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

After training 11 models, the results need to be accessible to a non-technical audience without requiring them to install a full ML environment, re-train models, or interact with raw `.joblib` checkpoint files.

The deployment design must answer:

1. **What format to store predictions in**: raw checkpoint (.joblib) vs. pre-computed predictions (JSON or parquet)
2. **How to serve the dashboard**: live inference from checkpoints vs. static pre-computed results
3. **What the dashboard should show**: raw metrics table vs. interactive comparison

A live inference approach would require loading ~11 checkpoint files (some with PyTorch/Darts dependencies) into the Streamlit process — adding ~2–5 GB of dependencies and 60–300 seconds of inference time to every dashboard load.

## Decision Drivers

- Dashboard must load in < 3 seconds with minimal dependencies
- Reviewers without PyTorch, Darts, or catboost installed should be able to run the dashboard
- Pre-computed results are the standard approach for portfolio projects (see multimodal-damage-assessment: inference_results.parquet committed to git)
- JSON is human-readable and can be inspected without any tooling; parquet requires pyarrow
- Each model's JSON file is ~500KB–2MB (y_true + y_pred arrays + metrics dict)
- Total storage: 11 models × ~1MB = ~11MB (safe to commit to git)

## Considered Options

**Option A — Live inference from .joblib checkpoints**
- Pros: Always uses latest model; single source of truth
- Cons: Requires all ML/DL dependencies at dashboard runtime; 60–300s startup; .joblib files are 50–500MB (too large for git)
- Why considered: Most "complete" solution

**Option B — Pre-computed JSON per model (chosen)**
- Pros: < 3s dashboard load; only numpy/pandas/streamlit/plotly needed; JSON files are ~1MB each (committable); human-readable
- Cons: Dashboard is a snapshot — must re-run training + inference to update
- Why considered: Matches multimodal-damage-assessment's pre-computed parquet pattern

**Option C — Pre-computed parquet (single file)**
- Pros: More compact than multiple JSONs; single file to commit
- Cons: Requires pyarrow; harder to inspect individual model results; all-or-nothing (must regenerate all models if one changes)
- Why considered: Used in the multimodal-damage-assessment project

## Decision Outcome

**Chosen: Option B — Pre-computed JSON per model, committed to `data/forecasts/ETT/`.**

Each model produces a JSON file with:
```json
{
  "model": "lightgbm",
  "variant": "h1",
  "mode": "multivariate",
  "tuned": false,
  "horizon": 24,
  "metrics": {"RMSE": 0.755, "MAE": 0.556, "R2": 0.952, ...},
  "train_metrics": {"RMSE": 0.687, "MAE": 0.477, "R2": 0.993, ...},
  "y_true": [...],
  "y_pred": [...]
}
```

`train_metrics` added in v5: in-sample RMSE/MAE/R² computed on training data for ML and LSTM models; empty dict `{}` for statistical and Darts DL models where in-sample prediction is not straightforward.

JSON over parquet because: human-readable, no pyarrow dependency, individual model files can be updated independently.

JSON saved **before** joblib checkpoint in `scripts/train_model.py` — so results persist even if the checkpoint write fails (e.g. LSTM pickle error).

Run name encoding: `{model}[_multivariate][_tuned]_{variant}.json` — allows univariate and multivariate results to coexist in the same directory.

Dashboard (`src/ui/dashboard.py`) v3 reads `data/forecasts/ETT/*.json` at startup, builds a DataFrame of metrics, and renders **6 tabs**:
- **Benchmark Results**: KPI cards, ranked bar chart with per-metric formula cards, normalised radar chart, full metric table
- **Forecast Gallery**: all models overlaid on OT test set, family colour-coded, zoomable with range slider
- **Model Inspector**: Actual vs Predicted, residuals, histogram, scatter + **Overfitting Diagnostic** (Train vs Test RMSE gap with ratio-based verdict) + Feature Importance (MDI from JSON) + SHAP beeswarm (ML models) + **AI Explainability** (GPT-4o mini streaming bullets, Statistical/DL models)
- **Data Explorer**: ETT raw signal (dual-panel time series + range slider), summary stats, Pearson heatmap
- **Statistical Tests**: Stationarity (ADF+KPSS + contextual interpretation), ACF/PACF (with peak detection and model implications), Granger Causality (table + per-feature interpretation), Diebold-Mariano (pairwise test + actionable guidance + full p-value matrix with next-step summary)
- **Ask AI (RAG)**: keyword retrieval over knowledge base → GPT-4o mini response with cited source IDs; 6 quick-question buttons that auto-send on click

**Overfitting Diagnostic (Model Inspector, v3):** Shows Train RMSE, Test RMSE, ratio, and verdict:
- ratio < 1.2 → "Well-fitted" (green)
- ratio 1.2–2.0 → "Moderate overfit" (orange)
- ratio > 2.0 → "Overfitting" (red)
- ratio < 1.0 (test < train) → characteristic of CatBoost ordered boosting

**AI Explainability (Model Inspector):** For any selected model, a button calls GPT-4o mini with:
- Model algorithm, family, benchmark metrics, rank vs. average
- Residual mean (bias), residual std, residual lag-1 autocorrelation (pattern detection)
- Outputs 3 bullet points: RMSE rationale, residual failure modes, production recommendation

**RAG Ask AI (Tab 6):** Keyword-based retrieval (no vector DB) over pre-built chunks (dataset overview, benchmark table, oracle explanation, model descriptions, metric formulas, feature engineering). Top-4 chunks sent to GPT-4o mini as context. Quick-question buttons write directly to `st.session_state.chat_textarea` and set `_auto_send=True` so the query fires without requiring a manual Send click.

**Statistical Tests (Tab 5, v3):** Each section now includes a contextual interpretation card:
- Stationarity: detects ADF/KPSS agreement or conflict and explains implications for ARIMA vs ML
- ACF/PACF: reads computed values at lags 1, 24, 48; flags peaks by value; detects slow decay pattern
- Granger Causality: lists significant features, explains physical mechanism (load → heat → OT), adds correlation ≠ causation caveat
- Diebold-Mariano: per-comparison guidance (significant = justified model choice; not significant = prefer simpler); matrix summary with count of equivalent pairs and 3 next steps

**API Key management:** `OPENAI_API_KEY` stored in `.streamlit/secrets.toml` (gitignored by `.gitignore`). Dashboard gracefully degrades — AI features show a warning if key is absent, all other tabs remain functional.

**Streamlit Cloud deployment:** `requirements-streamlit.txt` lists only 5 deps (numpy, pandas, plotly, streamlit, openai). No torch/darts/catboost at runtime.  
**Live URL:** https://time-series-forecasting-comparison-q74ppg2ggcaqxywkqwqxsw.streamlit.app/

## Positive Consequences

- Dashboard loads from pre-committed JSON in < 3 seconds
- No PyTorch, Darts, catboost, or other heavy ML dependencies at runtime
- Individual model results can be regenerated without re-running all 11 models
- JSON files are diff-able in git — metric changes are visible in PR diffs

## Negative Consequences

- Dashboard is a snapshot — not real-time with model weights
- y_true and y_pred arrays (3,484 floats each) make JSON files ~1–2MB; total ~11MB committed
- Must explicitly re-run `train_model.py` and commit new JSON after any model change

## Implementation Notes

- `src/utils.save_results()`: saves JSON to `data/forecasts/ETT/{name}.json`; creates directory if needed
- `scripts/train_model.py`: calls `save_results()` BEFORE `joblib.dump()` — results persist even if checkpoint write fails
- `.gitignore`: includes `/models/` (checkpoints), excludes `data/forecasts/ETT/*.json` (committed); `.streamlit/secrets.toml` explicitly gitignored
- `src/ui/dashboard.py`: `Path('data/forecasts/ETT').glob('*.json')` at startup; `st.stop()` with error if no files found
- `.streamlit/config.toml`: dark theme (backgroundColor=#0D1117), Inter font, XSRF protection
- `requirements-streamlit.txt`: minimal 5-dep file for Streamlit Cloud (numpy, pandas, plotly, streamlit, openai)
- `pyproject.toml`: `openai>=2.44.0` added for local development

## Overfitting / Underfitting Remediation (v6)

After measuring train vs test RMSE gaps, targeted fixes were applied and models retrained:

**ML Overfitting:**
| Model | Problem | Fix | Before | After |
|-------|---------|-----|--------|-------|
| Random Forest | 2.13× gap | Optuna max_features search (sqrt/log2/0.5/0.7/1.0); max_depth≤10 found | RMSE=0.783 | RMSE=0.703, gap=1.06× ✅ |
| XGBoost | 1.23× gap | early_stopping_rounds=50 in XGBRegressor() constructor (XGBoost 2.x API); reg_alpha/reg_lambda in Optuna | RMSE=0.847 | RMSE updated post-retrain |
| LightGBM | 1.10× (OK) | No change — fixed regularized defaults already well-fitted | RMSE=0.755 | unchanged |
| CatBoost | 0.82× (OK) | No change — ordered boosting gives conservative in-sample by design | RMSE=0.744 | unchanged |

**DL Underfitting (all models RMSE > 3.0 vs ML < 0.85) — final results:**
| Model | Fix | Param change | Before | After | Note |
|-------|-----|-------------|--------|-------|------|
| LSTM | More capacity + longer window | hidden_size 128→256, num_layers 2→3, input_len 96→168 | RMSE=5.332 | RMSE=5.332 (v1 kept) | v2 causes MPS kernel hang on Apple Silicon; all retries failed |
| Transformer | More capacity + longer window | d_model 64→128, input_chunk_length 96→168 | RMSE=3.429 | RMSE=3.525 | Marginally worse — capacity alone insufficient without LR warmup |
| N-BEATS | Longer window (already large) | input_chunk_length 96→168 | RMSE=4.065 | RMSE=4.358 | Slightly worse — needs more epochs or stack separation for longer context |
| TFT | More capacity + longer window + covariates | hidden_size 64→128, input_chunk_length 96→168, past_covariates=6 load cols | — | RMSE=4.214 | New architecture; past_covariates add electrical load signals |

**Verdict:** DL capacity increases did not close the ML vs DL RMSE gap (0.70 vs 3.5+). The gap is driven by oracle lag access (ML) vs univariate input (DL), not model architecture.

**TFT past_covariates design decision:** TFT.fit() accepts optional `past_cov_arr` (numpy array of shape [n_train, 6]) and `past_cov_dates`. These are converted to a Darts multivariate TimeSeries and stored as `self._past_cov_ts` for use in predict(). train_model.py loads a separate multivariate split for TFT and passes the load columns (HUFL/HULL/MUFL/MULL/LUFL/LULL). This is the same load-to-OT physical relationship that ML models exploit via X features.

**XGBoost 2.x API note:** `early_stopping_rounds` was moved from `fit()` to the `XGBRegressor()` constructor in XGBoost 2.0. When `eval_set` is provided to `fit()`, early stopping activates automatically. The `callbacks` parameter was also removed from `fit()`. This is a breaking change from XGBoost 1.x.

## Related Decisions

- ADR-004: Model selection (all 11 models must produce JSON for complete dashboard)
- ADR-005: Oracle lag filling (reflected in dashboard benchmark table note)

## References

- multimodal-damage-assessment: inference_results.parquet committed to git — same pre-computed pattern
- Streamlit deployment best practice: committed static data file, minimal runtime dependencies
