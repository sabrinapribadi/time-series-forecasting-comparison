# ADR-001: Task Framing — Univariate First, Multivariate Phase 2

**Status:** Accepted — Amended  
**Date:** 2026-06-25 (Amended 2026-07-02)  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

The ETT dataset has 8 columns: a `date` column, 6 load features (HUFL, HULL, MUFL, MULL, LUFL, LULL), and the target Oil Temperature (`OT`). There are two fundamentally different ways to use the input features:

1. **Univariate** — forecast OT using only its own history
2. **Multivariate** — forecast OT conditioned on all 6 load features + its own history
3. **Hierarchical** — separately forecast each series, then reconcile

The choice determines which model families are applicable and directly affects how the benchmark comparison is structured. A multivariate comparison would benefit ML and DL models — which can consume covariates — and disadvantage statistical models (ARIMA, Holt-Winters) that have no native covariate mechanism.

## Decision Drivers

- Goal of Phase 1 is benchmarking fairness: all 11 models should compete on the same input information
- Statistical models (ARIMA, Holt-Winters) cannot consume exogenous features without significant modification
- Deep learning models (TFT, Transformer) are specifically designed for multivariate time series
- A univariate comparison isolates model architecture differences from feature richness differences
- IOH IMPACT congestion project uses a multivariate approach: `Volume_WD_GB` as a chained regressor for Prophet and XGBoost — this pattern should be available for ML models in Phase 2

## Considered Options

**Option A — Univariate only**
- Pros: Fair comparison across all 11 models; isolates architecture from feature engineering
- Cons: Understates ML/DL advantage; load features are strongly predictive of OT
- Why considered: Simplest and most comparable approach

**Option B — Multivariate only**
- Pros: More realistic (production forecasters use all available signals)
- Cons: Cannot run statistical models as direct baselines; comparison is confounded
- Why considered: IOH production pipelines use multivariate inputs

**Option C — Univariate first, multivariate as optional mode (chosen)**
- Pros: Phase 1 is a fair cross-family benchmark; Phase 2 shows the covariate uplift separately; statistical models remain as honest baselines
- Cons: Requires two code paths (--mode flag)
- Why considered: Best of both; mirrors IOH AOP2026 approach of starting simple

## Decision Outcome

**Chosen: Option C — Univariate first, multivariate as optional mode (--mode flag).**

Phase 1 (this project): all models forecast OT from OT history only in `mode='univariate'`.  
Phase 2 (Amendment, 2026-07-02): `mode='multivariate'` added to ETTLoader and train_model.py.  
Statistical models remain as univariate baselines regardless of mode.  
Hierarchical forecasting is out of scope — it applies when individual series must be consistent with aggregated totals, which is not a requirement for OT prediction.

## Positive Consequences

- All models are evaluated on the same target (OT) with comparable input windows
- The benchmark is reproducible: same train/val/test indices, same target
- Multivariate uplift can be measured cleanly by comparing `--mode univariate` vs `--mode multivariate` results for ML models

## Negative Consequences

- Phase 1 results understate ML/DL advantage (load features are informative covariates)
- Two code paths in train_model.py and ETTLoader increase complexity

## Implementation Notes

- `ETTLoader.get_splits(mode='univariate')` drops LOAD_COLS from returned DataFrames
- `ETTLoader.get_splits(mode='multivariate')` retains all 6 load columns — 17 feature columns vs 11
- `scripts/train_model.py --mode multivariate` passes effective_mode='multivariate' only for ML models; statistical and DL models always use univariate (they have their own interfaces for covariates)
- `make train-multivariate` runs all 4 ML models in multivariate mode on ETTh1

## Related Decisions

- ADR-002: Feature preprocessing choices (which features are computed)
- ADR-004: Model selection (which 11 models to compare)
- ADR-005: Feature engineering for ML models

## References

- Zhou et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI. https://arxiv.org/abs/2012.07436
- IOH IMPACT Congestion Forecasting: multivariate Prophet/XGBoost with Volume_WD_GB as chained regressor
