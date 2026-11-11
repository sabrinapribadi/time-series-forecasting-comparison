# ADR-008: Fair Rolling Benchmark and Zero-Shot Foundation Models

**Status:** Accepted — written and pushed before any code, download or run for this decision
**Date:** 2026-11-11
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

- **The benchmark is not like-for-like.** ML models get true past values at every test step (ADR-005, "oracle lags"), so a 24-hour forecast becomes a 1-step forecast.
- Deep and statistical models forecast all 3,484 test hours in one go. The ranking (ML RMSE 0.70 vs Transformer 3.53) mostly measures that difference.
- **New option:** pretrained time-series foundation models forecast with no training on our data (zero-shot) and run on a CPU.
- **Question:** when every model forecasts the next 24 hours from the same history, does a zero-shot foundation model beat the best model we trained ourselves?

## Decision Drivers

- Same information for every model: only values observed up to the forecast start.
- Fixed before any run: protocol, models, champion choice, the winning margin and the leakage handling.
- Free to run (R20): open weights on a laptop CPU; no paid API.
- ETTh1 is a widely published dataset, so some models may have seen it in pretraining.

## Decision (pre-registered)

### 1. Rolling protocol

- **Forecast starts (origins):** every 24 hours from the first test hour (row 13,936): 145 origins, each forecasting OT for the next 24 hours (3,480 hours).
- **Validation origins:** the same scheme over the validation period: 71 origins from row 12,218 (see corrections), used only to choose champions.
- **History:** every model sees all OT values up to the origin, never after it. Foundation models use the last 2,048 hours.
- **Trained models:** settings and training data as in ADR-004 (no new tuning).
  - ML: recursive. Each step's prediction becomes the next step's lag; calendar features are known in advance.
  - Deep models: Darts historical forecasts, no retraining.
  - Statistical: refitted at each origin with ADR-004 settings (ARIMA keeps its selected order).
- **Primary track:** OT only (univariate). **Secondary track (reported):** plus the six load series as past-only inputs, for models that accept them.

### 2. Models

| Group | Models | Licence |
|---|---|---|
| Baselines | Same hour yesterday (seasonal naive, 24 h); same hour last week | — |
| Trained (ADR-004) | ARIMA, Holt-Winters, Prophet, Random Forest, XGBoost, LightGBM, CatBoost, LSTM, N-BEATS, Transformer, TFT | — |
| Zero-shot | Chronos-2 (120M), TimesFM-2.5 (200M), Moirai-2.0-R-small (11M) | Apache-2.0; Apache-2.0; CC BY-NC 4.0 (non-commercial: allowed for this portfolio) |

**Not run:**

| Model | Rating | Why |
|---|---|---|
| Timestral (Mistral, Oct 2026) | Hold | Its licence withdraws all rights from anyone whose company or employer earns over $20 million a month; its real training data is unnamed. Its benchmark lead is the vendor's claim, under leaderboard review. |
| TimeGPT (Nixtla) | Hold | Paid API (R20); training data not public |
| MOMENT (CMU) | Not included | Built mainly for classification, anomaly detection and imputation; its forecasting results use a trained head |
| Lag-Llama | Not included | Self-declares test-data leakage on GIFT-Eval |

### 3. Leakage handling

| Model | ETTh1 in documented training data? | Tier |
|---|---|---|
| TimesFM-2.5 | Uses GIFT-Eval's pretraining set, which excludes evaluation datasets (ETTh1 is one) | Clean (documented) |
| Chronos-2 | "Select" GIFT-Eval training portions; test portions excluded; which datasets is not listed | Possible exposure before the last 1,008 hours |
| Moirai-2.0 | A subset of GIFT-Eval pretraining and training data | Possible exposure before the last 1,008 hours |

- GIFT-Eval keeps the end of each series for testing; for hourly ETTh1 no training portion reaches the last 1,008 hours (`TEST_SPLIT` and window rule in its code).
- **Clean tail:** the 41 test origins whose 24 hours fall in those last 1,008 hours. Every result is reported on all 145 origins and on the clean tail.

### 4. Scoring and the gate

- **Primary metric:** MASE, i.e. MAE divided by the in-sample MAE of "same hour yesterday" on the training set (below 1 beats that baseline). Also MAE, RMSE; weighted quantile loss for models with quantiles.
- **Champions:** the trained model and the foundation model with the lowest validation MASE (univariate track). Chosen before any test score is read.
- **Foundation model wins if** all three hold on the test origins:
  - its MASE is at least 5% lower than the trained champion's;
  - a Diebold–Mariano test with the Harvey–Leybourne–Newbold correction on per-origin MAE gives p < 0.05 (two-sided);
  - its MASE is also lower on the clean tail.
- **Trained model wins** under the mirror conditions. **Otherwise:** no clear winner, stated plainly.
- **Also reported:** how much the oracle lags flattered ML (old vs new RMSE); whether each champion beats both baselines; CPU latency per forecast (p50, p95) and model size. Cost is $0.

### 5. What changes downstream

- README, dashboard and pages lead with the rolling benchmark. The old table stays, labelled "oracle lags (ADR-005): not like-for-like".
- The bandit (ADR-007) is rerun on rolling forecasts only under a new decision.

## Corrections before any run (2026-11-11)

Found while reading the training code; pushed before any forecast was made.

- **Validation origins start at row 12,218 (71 origins), not 12,194 (72).** ML lag features drop the first 24 rows, so ML training ends at raw row 12,200. Starting later keeps every validation target out of every model's training rows.
- **TFT runs only in the covariate track.** It was trained with the six load series and cannot forecast from OT alone. The univariate trained group has 10 models.
- **ML uses the univariate checkpoints** (`random_forest_h1`, `xgboost_h1`, `lightgbm_h1`, `catboost_h1`). The headline multivariate variants need same-hour load values, which are unknown when forecasting ahead.
- **Covariate track:** Chronos-2 (past covariates) and TFT. TimesFM-2.5 and Moirai-2.0 run univariate only.
- **Known bias, stated:** deep and boosted models used the validation period for early stopping, so their validation scores are slightly optimistic. This affects only the choice within the trained group.

## Deviation before any test score was read (2026-11-11, 17:20)

- **Found during the runs:** the saved Transformer, N-BEATS and TFT checkpoints were untrained shells (Darts' network was lost when they were pickled), and the "univariate" Random Forest and CatBoost checkpoints were 15-feature models. All five were retrained with their ADR-004 settings on CPU.
- **Dropped by decision:** the Transformer and TFT, after 2 hours of CPU retraining without finishing. N-BEATS (47 minutes), Random Forest and CatBoost finished and stay in.
- **Effect:** the univariate trained group has 9 models; the covariate track has no trained comparator, so Chronos-2 with load inputs is reported against its own univariate score only.
- **State when decided:** validation scores of the 15 finished models had been read (the champion step); no test score had been read by anyone.

## Considered Options

| Option | Status | Why |
|---|---|---|
| **Rolling 24-hour origins, same history for all, champions by validation** | **Chosen** | Like-for-like; answers the question; no test-set selection |
| Add foundation models to the current protocol | Rejected | Inherits the oracle-lag unfairness |
| Pick the best test score per group | Rejected | Selection on test flatters the winner |
| Fine-tune the foundation models | Not now | A different question (zero-shot vs trained); would need its own ADR |
| Add a newer dataset free of pretraining exposure | Not now | Changes the project's dataset; the clean tail covers leakage for ETTh1 |

## Falsification

| Result | Diagnosis | Next action |
|---|---|---|
| Foundation model wins, also on the clean tail | Pretraining transfers to this transformer-temperature series | Recommend zero-shot as the default first model |
| Wins on all origins, not on the clean tail | Possible pretraining exposure | Report as unproven; prefer the clean-tier model |
| Trained champion wins | Twelve months of local history beats general pretraining here | Keep trained models; foundation models as a quick baseline |
| ML falls far behind once oracle lags go | The old ranking measured information, not skill | Retire the old headline |

## Related Decisions

- ADR-003 (split), ADR-004 (models), ADR-005 (oracle lags, now superseded as the headline), ADR-007 (bandit)

## Evidence

| Reference | Venue | Peer-reviewed | Year | Citations (OpenAlex, 2026-11-11) | Used for |
|---|---|---|---|---|---|
| Ansari et al., Chronos: Learning the Language of Time Series | TMLR (arXiv 2403.07815) | Yes | 2024 | 936 | Foundation-model approach |
| Ansari et al., Chronos-2: From Univariate to Universal Forecasting (arXiv 2510.15821) | Technical report | No | 2025 | Not indexed | Chronos-2 training data and covariates |
| Das et al., A decoder-only foundation model for time-series forecasting | ICML 2024 | Yes | 2024 | 491 | TimesFM |
| Woo et al., Unified Training of Universal Time Series Forecasting Transformers | ICML 2024 | Yes | 2024 | 309 | Moirai |
| Liu et al., Moirai 2.0: When Less Is More for Time Series Forecasting (arXiv 2511.11698) | Preprint | No | 2025 | 124 | Moirai 2.0 training data |
| Aksu et al., GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation | NeurIPS 2024 workshop (arXiv 2410.10393) | Workshop | 2024 | 223 | Test split, leakage tracking |
| Zhou et al., Informer | AAAI 2021 | Yes | 2021 | 8,633 | ETTh1 and its split |
| Zeng et al., Are Transformers Effective for Time Series Forecasting? | AAAI 2023 | Yes | 2023 | 4,297 | Simple baselines and fair protocols |
| Tashman, Out-of-sample tests of forecasting accuracy | Int. J. Forecasting | Yes | 2000 | 1,189 | Rolling-origin evaluation |
| Hyndman & Koehler, Another look at measures of forecast accuracy | Int. J. Forecasting | Yes | 2006 | 6,419 | MASE |
| Diebold & Mariano, Comparing Predictive Accuracy | J. Business & Economic Statistics | Yes | 1995 | 5,716 | Accuracy test |
| Harvey, Leybourne & Newbold, Testing the equality of prediction mean squared errors | Int. J. Forecasting | Yes | 1997 | 2,205 | Small-sample correction |
| Goswami et al., MOMENT | ICML 2024 | Yes | 2024 | 362 | Why not included |
| Garza & Mergenthaler-Canseco, TimeGPT-1 (arXiv 2310.03589) | Preprint | No | 2023 | 493 | Why on hold |
| Timestral announcement and model licence (15 Oct 2026) | Mistral AI | No | 2026 | — | Why on hold |
