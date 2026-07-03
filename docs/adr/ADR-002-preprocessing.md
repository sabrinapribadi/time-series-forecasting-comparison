# ADR-002: Preprocessing — Cyclical Encoding, Lag Features, and Rolling Statistics

**Status:** Accepted  
**Date:** 2026-06-25  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

Tree-based ML models and neural networks need numerical features — they cannot consume raw timestamps or string-encoded time information. Two encoding design choices must be made:

1. **How to encode time-of-day, day-of-week, and month-of-year**: linear integers vs. cyclical sin/cos vs. one-hot
2. **How to capture temporal dependencies for ML models**: lag features vs. rolling statistics vs. nothing (leave sequence modelling to the model)

Rolling statistics (rolling_mean_3, rolling_std_3, trend_3, growth_rate) capture short-term trend and volatility without requiring the model to select the relevant lag window.

## Decision Drivers

- Hour=23 and hour=0 are adjacent, but linear encoding places them 23 apart — this misrepresents periodicity
- One-hot encoding requires 24+7+12=43 additional columns; cyclical requires only 6
- Lag features are essential for tree-based models, which have no built-in sequence memory
- Rolling statistics provide summary statistics of recent history without requiring the model to select the relevant lag window
- Rolling features chosen: `rolling_mean_3`, `rolling_std_3`, `trend_3` (linear slope), `growth_rate` (pct_change) — all shifted by 1 step to prevent leakage
- All derived features must be strictly backward-looking: no future information in training data

## Considered Options

**Option A — Linear integer features (hour, dayofweek, month)**
- Pros: Simple; directly interpretable
- Cons: Discontinuity at period boundaries (hour=23→hour=0 appears as a large jump); tree models may learn spurious splits near the boundary
- Why considered: Default for many quick implementations

**Option B — One-hot encoding of time features**
- Pros: No spurious discontinuities; model learns a separate coefficient per bin
- Cons: 43 columns for just calendar features; high cardinality; sparse for 15-minute data
- Why considered: Standard for categorical variables

**Option C — Cyclical sin/cos encoding (chosen)**
- Pros: Continuity at period boundaries guaranteed; only 6 columns for 3 periods; distance in feature space matches temporal distance
- Cons: Non-trivial for models to recover exact hour from sin+cos; not directly interpretable
- Why considered: Standard for time series with periodic structure

## Decision Outcome

**Chosen: Option C — Cyclical sin/cos encoding for all calendar features.**

Three period pairs computed by `ETTLoader._add_time_features()`:

| Feature | Formula | Period |
|---------|---------|--------|
| hour_sin / hour_cos | sin/cos(2π · (hour+minute/60) / 24) | 24 h |
| dow_sin / dow_cos | sin/cos(2π · dayofweek / 7) | 7 days |
| month_sin / month_cos | sin/cos(2π · (month−1) / 12) | 12 months |

Lag features for ML models (ADR-005):

| Feature | Shift | Rationale |
|---------|-------|-----------|
| OT_lag_1 | 1 step | Most recent observation |
| OT_lag_2 | 2 steps | Second most recent |
| OT_lag_24 | 24 steps | Previous day same hour (hourly); OT_lag_4/OT_lag_96 for 15-min |

Rolling statistics (all shifted by 1 to prevent leakage):

| Feature | Formula | Purpose |
|---------|---------|---------|
| OT_rolling_mean_3 | rolling(3).mean().shift(1) | Short-term level estimate |
| OT_rolling_std_3 | rolling(3).std().shift(1) | Recent local volatility |
| OT_growth_rate | pct_change().shift(1) | Relative velocity |
| OT_trend_3 | polyfit slope on last 3 obs, shift(1) | Instantaneous slope direction |

## Positive Consequences

- No spurious discontinuities at day/week/year boundaries
- Compact feature representation (6 columns for 3 cyclical periods)
- Rolling features summarise recent history without excessive lag columns
- Shift-by-1 ensures strict backward-lookingness on all derived features

## Negative Consequences

- Rolling features + lag features introduce NaN in the first few rows — dropped via `dropna()` after feature engineering
- Cyclical encoding is less directly interpretable than raw integers
- `OT_trend_3` uses `rolling().apply(np.polyfit)` which is slower than vectorised operations for large 15-min datasets

## Implementation Notes

- `_add_time_features(df)`: always applied; uses `dt.hour + dt.minute/60` for sub-hourly fractional hour
- `_add_lag_features(df, lag_steps)`: hourly default [1, 2, 24]; 15-min default [1, 4, 96]
- `_add_rolling_features(df)`: all 4 columns use `.shift(1)` to prevent any future leakage; enabled via `add_rolling_features=True`
- NaN handling: `df.dropna().reset_index(drop=True)` after all lag and rolling feature computation

## Related Decisions

- ADR-001: Task framing (univariate vs. multivariate — which raw columns are included)
- ADR-005: Lag filling strategy on test set

## References

- Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice. Chapter 7
