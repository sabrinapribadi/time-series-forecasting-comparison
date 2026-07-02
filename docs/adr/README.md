# Architecture Decision Records

All major design decisions are documented here before implementation.

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](ADR-001-task-framing.md) | Task framing: Univariate first, multivariate Phase 2 | Accepted — Amended (multivariate now live) |
| [ADR-002](ADR-002-preprocessing.md) | Preprocessing: Cyclical encoding + lag features + rolling statistics | Accepted |
| [ADR-003](ADR-003-validation-strategy.md) | Validation: Chronological 70/10/20 split, no data leakage | Accepted |
| [ADR-004](ADR-004-model-selection.md) | Model selection: Compare all 11 + Optuna HPO option | Accepted |
| [ADR-005](ADR-005-feature-engineering.md) | Feature engineering: Oracle lag filling for ML models on test set | Accepted |
| [ADR-006](ADR-006-deployment.md) | Deployment: Pre-computed JSON results → Streamlit dashboard | Accepted |

## Decision Dependency Graph

```
ADR-001 (task framing)
    ├── ADR-002 (preprocessing) → ADR-005 (lag filling)
    ├── ADR-003 (validation)
    ├── ADR-004 (model selection)
    └── ADR-006 (deployment)
```
