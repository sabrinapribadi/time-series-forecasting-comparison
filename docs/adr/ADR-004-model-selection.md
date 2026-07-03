# ADR-004: Model Selection — Compare All 11 Models with Optuna HPO

**Status:** Accepted  
**Date:** 2026-06-25  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

With 11 models across 3 families, the project must decide:

1. **How many models to train**: compare all vs. rule-based pre-selection
2. **Whether to tune hyperparameters**: default configs vs. Optuna HPO
3. **How many epochs for DL models**: under-compute is a known risk on CPU/MPS

Rule-based per-site model selection (based on series characteristics such as volatility, trend_strength, seasonality_strength) is useful for production pipelines with hundreds of series but less relevant for a single-target academic benchmark.

## Decision Drivers

- The explicit goal is benchmarking: all models must run to produce a comparison table
- Rule-based selection is a production pattern, not a benchmarking pattern
- Optuna HPO (HoltWinters 15 trials, RF 10 trials, CatBoost 8 trials) should be available as an optional `--tune` flag
- DL models need 100+ epochs for fair comparison; MPS/CPU limits make this slow
- Optuna must be optional (default hyperparameters for the primary benchmark; HPO as --tune flag)

## Considered Options

**Option A — Train all 11 models with default hyperparameters**
- Pros: Fast; reproducible; fair comparison with no extra information (val set not used for HPO)
- Cons: Default hyperparameters may heavily favour some models; DL models need many epochs
- Why considered: Simplest baseline comparison

**Option B — Rule-based pre-selection**
- Pros: Mimics production deployment; only trains the "right" model per series
- Cons: Precludes side-by-side comparison; series-characteristic metrics need to be computed first
- Why considered: Practical for pipelines with hundreds of series

**Option C — Compare all 11 with optional Optuna HPO (chosen)**
- Pros: All models run in primary benchmark; HPO available for fair tuned comparison
- Cons: More compute time; requires two run modes (default vs. --tune)
- Why considered: Best of both — benchmark completeness + HPO for tunable models

## Decision Outcome

**Chosen: Option C — Compare all 11 models; Optuna HPO optional via `--tune` flag.**

Primary benchmark: default hyperparameters from `configs/default_config.yaml`.  
HPO benchmark: `--tune` flag enables Optuna with per-model trial counts.

Model families and expected trade-offs:

| Family | Strengths | Weaknesses |
|--------|-----------|-----------|
| Statistical | Interpretable, fast, no feature engineering | Limited capacity; long-horizon degradation |
| ML (tree-based) | Handles non-linearity + covariates; fast inference | Requires lag features; no explicit sequence memory |
| Deep Learning | Long-range dependencies; architecture flexibility | Slow to train; needs GPU for fair evaluation |

Tunable models and Optuna configuration:

| Model | Trials | Objective |
|-------|--------|-----------|
| HoltWinters | 15 | Minimize RMSE (train residuals) |
| RandomForest | 10 | Maximize R² (val) |
| CatBoost | 8 | Maximize R² (val) |
| XGBoost | 50 | Maximize R² (val) |
| LightGBM | — | Fixed regularized defaults (no tuning) |

DL models are not tuned via `--tune` in Phase 1 (compute cost too high on MPS/CPU).

## Positive Consequences

- All 11 model checkpoints and JSON results are produced
- Dashboard shows complete comparison table with no missing rows
- Optuna configs in `configs/optuna_configs.yaml` can be extended to any model
- Training is parallelisable (models are independent)

## Negative Consequences

- DL model results are undertrained (5–30 epochs vs. 100+ needed for fair comparison)
- Total benchmark run time on MPS: ~30–120 minutes
- Optuna search on val set breaks strict separation if val is small (1,742 hourly points)

## Implementation Notes

- `scripts/train_model.py`: `TUNABLE = {"holt_winters", "random_forest", "xgboost", "catboost"}`
- When `--tune`: calls `model.tune_with_optuna(X_tr, y_tr, X_val, y_val, n_trials=n)` which includes a refit on concat([train, val]) after finding best params
- When no `--tune`: standard `model.fit(X_train, y_train)` with defaults from `build_model()`
- `make train-tuned`: runs all 4 tunable models with Optuna on ETTh1
- Result JSON filename: `{model}[_multivariate][_tuned]_{variant}.json` — encodes HPO status

## Related Decisions

- ADR-001: Task framing (univariate vs. multivariate determines feature matrix)
- ADR-002: Preprocessing (features fed to ML models)
- ADR-005: Oracle lag filling (affects ML model advantage in benchmark)

## References

- Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD. https://arxiv.org/abs/1907.10902
