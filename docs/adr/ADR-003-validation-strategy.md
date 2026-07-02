# ADR-003: Validation Strategy — Chronological 70/10/20 Split

**Status:** Accepted  
**Date:** 2026-06-25  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

Time series data has temporal ordering — the past predicts the future, not the other way around. Choosing a validation strategy requires deciding:

1. **How to split**: chronological vs. random shuffle vs. k-fold cross-validation
2. **What proportions**: 80/20 vs. 70/15/15 vs. 70/10/20
3. **What metric gates the benchmark**: RMSE vs. relative improvement vs. multiple metrics

A random shuffle would cause leakage: training on future data and evaluating on the past. Standard cross-validation is not directly applicable to time series without careful ordering (time-series cross-validation / walk-forward validation).

## Decision Drivers

- Temporal ordering must be preserved: no shuffling
- Three-way split is needed: `train` for fitting, `val` for early stopping / HPO, `test` for final evaluation
- ETT is a standard academic benchmark — the 70/10/20 split matches the Informer paper's split for ETTh1/ETTh2
- Proportions should leave enough test data to be statistically meaningful (3,484 steps for hourly = 145 days)
- A single fixed split is more reproducible than rolling windows for an initial benchmark

## Considered Options

**Option A — Random 80/20 split**
- Pros: Simple
- Cons: Leakage from future to past; invalid for time series
- Why considered: Default for sklearn pipelines

**Option B — Walk-forward cross-validation (k-fold time series)**
- Pros: Reduces variance of estimates; tests model on multiple windows
- Cons: k times more training runs; incompatible with DL models that need fixed train/val sets for early stopping; complex to implement consistently across 11 models
- Why considered: More statistically rigorous for production systems

**Option C — Chronological 70/10/20 split (chosen)**
- Pros: No leakage; matches Informer paper split; three distinct sets for fit/tune/evaluate; reproducible
- Cons: Single evaluation window; results depend on whether the test period is typical or anomalous
- Why considered: Standard for academic time series benchmarks

## Decision Outcome

**Chosen: Option C — Chronological 70/10/20 split.**

Row-count based (not date-based) so proportions are identical across all 4 ETT variants regardless of timestamps:

```
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.80)   # 0.70 + 0.10

ETTh1/h2: train=12,194 | val=1,742 | test=3,484 rows
ETTm1/m2: train=48,776 | val=6,968 | test=13,936 rows
```

Evaluation metrics: RMSE (primary), MAE, MAPE, SMAPE, MASE, R², MDA, Bias, MAAPE. No relative improvement gate (this is an exploratory benchmark, not an ablation with a falsification criterion).

## Positive Consequences

- No temporal leakage at any stage
- Identical indices across Python/pandas versions (row-count based, not date-based)
- Sufficient test set for meaningful evaluation (145 days at hourly, 97 days at 15-min)
- Matches Informer paper for comparability

## Negative Consequences

- Single test window — results may not generalise to other time periods
- Walk-forward validation would give more robust estimates but is 5–10× more compute
- Val set (10%) is relatively small — HPO results may be noisy on 1,742 points

## Implementation Notes

- `ETTLoader.get_splits()`: split indices computed once from `len(df)` after feature engineering (after `dropna()` if lags/rolling features enabled — so effective train size is ~12,177 not 12,194 for hourly with lag_24)
- Ordering: `df.sort_values('date').reset_index(drop=True)` before any split
- No `random_state` used in splits (deterministic row-count arithmetic)

## Related Decisions

- ADR-001: Task framing determines which columns are in the feature matrix
- ADR-002: Feature engineering determines when to drop NaN rows (which adjusts effective split sizes)
- ADR-005: Oracle lag filling on test set — validation set ground truth values are NOT used in test set lag construction

## References

- Zhou et al. (2021). Informer: Table 3, ETTh1/ETTh2 split sizes. AAAI. https://arxiv.org/abs/2012.07436
- Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice. Chapter 5 (Train/Test Split)
