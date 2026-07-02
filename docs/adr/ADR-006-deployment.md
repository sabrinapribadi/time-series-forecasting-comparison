# ADR-006: Deployment — Pre-Computed JSON + Streamlit Dashboard v2 + RAG / AI Explainability

**Status:** Accepted (updated 2026-07-02 — Dashboard v2)
**Date:** 2026-06-25 | **Last Updated:** 2026-07-02  
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
  "mode": "univariate",
  "tuned": false,
  "horizon": 24,
  "metrics": {"RMSE": 0.743, "MAE": 0.547, "MDA": 72.4, ...},
  "y_true": [...],
  "y_pred": [...]
}
```

JSON over parquet because: human-readable, no pyarrow dependency, individual model files can be updated independently.

JSON saved **before** joblib checkpoint in `scripts/train_model.py` — so results persist even if the checkpoint write fails (e.g. LSTM pickle error).

Run name encoding: `{model}[_multivariate][_tuned]_{variant}.json` — allows univariate and multivariate results to coexist in the same directory.

Dashboard (`src/ui/dashboard.py`) v2 reads `data/forecasts/ETT/*.json` at startup, builds a DataFrame of metrics, and renders **5 tabs**:
- **Benchmark Results**: KPI cards, ranked bar chart with per-metric formula cards, normalised radar chart, full metric table
- **Forecast Gallery**: all models overlaid on OT test set, family colour-coded, zoomable with range slider
- **Model Inspector**: Actual vs Predicted, residuals, histogram, scatter + **AI Explainability** (GPT-4o mini)
- **Data Explorer**: ETT raw signal (dual-panel time series + range slider), summary stats, Pearson heatmap
- **Ask AI (RAG)**: keyword retrieval over knowledge base → GPT-4o mini response with cited source IDs

**AI Explainability (Model Inspector):** For any selected model, a button calls GPT-4o mini with:
- Model algorithm, family, benchmark metrics, rank vs. average
- Residual mean (bias), residual std, residual lag-1 autocorrelation (pattern detection)
- Outputs 3–4 paragraphs: why the RMSE, what residuals reveal, when to use in production

**RAG Ask AI (Tab 5):** Keyword-based retrieval (no vector DB) over pre-built chunks (dataset overview, benchmark table, oracle explanation, model descriptions, metric formulas, feature engineering). Top-4 chunks sent to GPT-4o mini as context.

**API Key management:** `OPENAI_API_KEY` stored in `.streamlit/secrets.toml` (gitignored by `.gitignore`). Dashboard gracefully degrades — AI features show a warning if key is absent, all other tabs remain functional.

**Streamlit Cloud deployment:** `requirements-streamlit.txt` lists only 5 deps (numpy, pandas, plotly, streamlit, openai). No torch/darts/catboost at runtime.

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

## Related Decisions

- ADR-004: Model selection (all 11 models must produce JSON for complete dashboard)
- ADR-005: Oracle lag filling (reflected in dashboard benchmark table note)

## References

- multimodal-damage-assessment: inference_results.parquet committed to git — same pre-computed pattern
- Streamlit deployment best practice: committed static data file, minimal runtime dependencies
