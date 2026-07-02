# ADR-005: Feature Engineering — Oracle Lag Filling for ML Models on Test Set

**Status:** Accepted  
**Date:** 2026-06-25  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

Tree-based ML models need lag features (e.g. OT_lag_1, OT_lag_24) to see the recent history of the series. These features are constructed from the target column by shifting it backwards.

A critical question arises when evaluating on the test set: **what values should the lag features contain?**

- **Oracle evaluation**: use ground-truth OT values as lag inputs (as if previous steps are always known)
- **Recursive evaluation**: use the model's own predictions as lag inputs (simulating real deployment where future ground truth is unknown)

The two approaches give very different results on a 3,484-step test horizon:
- Oracle gives best-case ML performance; lag_1 is essentially a perfect 1-step-ahead signal
- Recursive compounds prediction error over thousands of steps and matches statistical model constraints

This choice also determines whether the benchmark comparison between ML and statistical families is on equal footing.

## Decision Drivers

- The project goal is benchmarking, not production deployment simulation
- Oracle evaluation is standard in academic ML-for-time-series papers (Informer, PatchTST, etc.)
- Oracle gives an upper bound on ML model capability — useful for understanding the value of each feature
- IOH production IMPACT congestion forecasting also uses oracle lag evaluation in its holdout analysis
- Recursive evaluation for 3,484 steps would require re-architecture of the predict() API (expose rolling prediction loop with state), adding significant complexity
- The oracle assumption is explicitly documented so readers understand the benchmark conditions

## Considered Options

**Option A — Oracle lag filling (chosen)**
- Pros: Standard in academic literature; shows best-case ML performance; simple implementation (precomputed lags from full ETT dataframe)
- Cons: Overstates ML advantage vs. statistical models; not directly deployable (ground truth not available in real deployment)
- Why considered: De facto standard for tabular-ML benchmarks on time series

**Option B — Recursive (multi-step rollout) evaluation**
- Pros: Matches real deployment; more honest comparison with statistical models
- Cons: Error compounds over 3,484 steps; needs significantly different predict() API; ML models would likely perform worse than statistical models (which also compound but have built-in uncertainty handling)
- Why considered: More realistic for production use case

**Option C — Rolling-window oracle (hybrid)**
- Pros: Uses ground truth every `horizon` steps to refresh context; matches how LSTM is evaluated; intermediate between A and B
- Cons: More complex; still oracle at each refresh point
- Why considered: This is what the LSTM uses (rolling-window oracle)

## Decision Outcome

**Chosen: Option A — Oracle lag filling for all ML models; Option C — Rolling-window oracle for LSTM.**

**ML models (RandomForest, XGBoost, LightGBM, CatBoost):**
Lags are precomputed from the complete `df` before splitting:
```python
df["OT_lag_1"]  = df["OT"].shift(1)
df["OT_lag_2"]  = df["OT"].shift(2)
df["OT_lag_24"] = df["OT"].shift(24)
```
When `test_df["OT_lag_1"][i]` is accessed at test step `i`, it contains the ground-truth OT value from step `i−1` — this is oracle data on the test set (the model is given the true previous value).

**LSTM model (rolling-window oracle):**
Predict `horizon=24` steps, extend context with true OT, advance `horizon` steps:
```python
context = concat([y_train, y_val])
for i in range(0, len(y_test), horizon):
    chunk = model.predict(context)   # predict 24 steps
    preds.extend(chunk)
    context = append(context, y_test[i:i+horizon])  # refresh with true OT
```

**Statistical models (Holt-Winters, ARIMA, Prophet):**
Forecast all 3,484 test steps in one shot from training data alone — no oracle. This is the standard statistical forecasting evaluation.

## Positive Consequences

- ML model performance represents the upper bound achievable with complete past data
- Implementation is simple (lags precomputed from full df before splitting)
- Consistent with academic benchmarks using ETT dataset
- Oracle lag advantage is explicitly documented in README and benchmark table note

## Negative Consequences

- ML models have an inherent advantage over statistical models in the benchmark — the comparison is not on equal footing
- Results are not directly comparable to production deployment performance
- MASE (which uses MAE(y_train) as baseline) does not fully account for the oracle advantage

## Implementation Notes

- Lags are computed in `ETTLoader._add_lag_features()` on the **full dataframe before splitting** — so test rows correctly reference train/val ground truth values
- `scripts/train_model.py` feature_cols: `[c for c in train_df.columns if c not in ('date', 'OT')]` — includes lag columns automatically
- LSTM rolling-window: context initialised from `concat([y_train, y_val])`; true `y_test` values appended after each horizon block
- A future planned enhancement (Phase 9): implement recursive evaluation for ML models to enable fair comparison with statistical models

## Related Decisions

- ADR-002: Preprocessing (lag feature construction details)
- ADR-003: Validation strategy (train/val/test split indices)
- ADR-004: Model selection (which models use lag features)

## References

- IOH IMPACT Congestion Forecasting: oracle lag evaluation in holdout analysis
- Zhou et al. (2021). Informer: uses precomputed look-back window as input — equivalent to oracle on evaluation set
