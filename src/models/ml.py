"""
Tree-based ML forecasting models: Random Forest, XGBoost, LightGBM, CatBoost.

All models follow a tabular regression interface:
    fit(X_train, y_train)
    predict(X) -> np.ndarray

Optional Optuna HPO (from IOH AOP2026 + congestion projects):
    tune_with_optuna(X_train, y_train, X_val, y_val, n_trials)

Trial counts per IOH production practice:
    RandomForest  — 10 trials, maximize R² on validation
    XGBoost       — 50 trials, maximize R² on validation
    CatBoost      —  8 trials, maximize R² on validation
    LightGBM      — fixed params (no Optuna); uses IOH production hyperparameters

Feature engineering (lag columns, cyclical time features) is handled upstream
by ETTLoader.get_splits(add_lag_features=True).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest regressor for multi-step tabular forecasting."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs: Any,
    ):
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestModel":
        self._model.fit(X_train, y_train)
        logger.info(f"RandomForest fitted on {X_train.shape}")
        return self

    def tune_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 10,
    ) -> "RandomForestModel":
        """Optuna HPO: 10 trials, maximize R² on validation (IOH AOP2026 pattern)."""
        import optuna
        from sklearn.ensemble import RandomForestRegressor
        from src.evaluation.metrics import compute_r2

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                # max_features controls decorrelation between trees — key lever against overfit
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
            }
            m = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
            m.fit(X_train, y_train)
            return compute_r2(y_val, m.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(f"RF Optuna best R²={study.best_value:.4f}, params={best}")

        self._model = RandomForestRegressor(**best, n_jobs=-1, random_state=42)
        self._model.fit(
            np.vstack([X_train, X_val]),
            np.concatenate([y_train, y_val]),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class XGBoostModel:
    """XGBoost regressor."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        random_state: int = 42,
        **kwargs: Any,
    ):
        from xgboost import XGBRegressor

        # XGBoost 2.x: early_stopping_rounds lives in the constructor, not fit().
        # Passing it here activates it automatically when fit() receives eval_set.
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            early_stopping_rounds=50,
            random_state=random_state,
            verbosity=0,
            nthread=1,  # prevents segfault on macOS with XGBoost 2.x multi-threading
            **kwargs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostModel":
        eval_set = [(X_val, y_val)] if X_val is not None else None
        # early_stopping_rounds is set in the constructor (XGBoost 2.x API).
        # It activates automatically when eval_set is provided here.
        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info(f"XGBoost fitted on {X_train.shape}")
        return self

    def tune_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> "XGBoostModel":
        """Optuna HPO: 50 trials, maximize R² on validation."""
        import optuna
        from xgboost import XGBRegressor
        from src.evaluation.metrics import compute_r2

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                # L1/L2 regularisation reduces 1.23× overfit gap alongside early stopping
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            m = XGBRegressor(**params, random_state=42, verbosity=0, early_stopping_rounds=50)
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return compute_r2(y_val, m.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(f"XGBoost Optuna best R²={study.best_value:.4f}, params={best}")

        self._model = XGBRegressor(**best, random_state=42, verbosity=0)
        self._model.fit(
            np.vstack([X_train, X_val]),
            np.concatenate([y_train, y_val]),
            verbose=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class LightGBMModel:
    """
    LightGBM regressor — uses IOH AOP2026 production hyperparameters (fixed, no Optuna).

    IOH params: n_estimators=500, num_leaves=31, learning_rate=0.05,
                reg_alpha=0.1, reg_lambda=0.1 (L1/L2 regularization from AOP2026).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ):
        from lightgbm import LGBMRegressor

        self._model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=-1,
            num_threads=1,  # prevents segfault on macOS with LightGBM multi-threading
            **kwargs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LightGBMModel":
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(X_train, y_train, eval_set=eval_set)
        logger.info(f"LightGBM fitted on {X_train.shape}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class CatBoostModel:
    """CatBoost regressor — gradient boosting with categorical feature support."""

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        **kwargs: Any,
    ):
        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            verbose=0,
            **kwargs,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CatBoostModel":
        eval_set = (X_val, y_val) if X_val is not None else None
        self._model.fit(X_train, y_train, eval_set=eval_set)
        logger.info(f"CatBoost fitted on {X_train.shape}")
        return self

    def tune_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 8,
    ) -> "CatBoostModel":
        """Optuna HPO: 8 trials, maximize R² on validation (IOH AOP2026 pattern)."""
        import optuna
        from catboost import CatBoostRegressor
        from src.evaluation.metrics import compute_r2

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
            m = CatBoostRegressor(**params, random_seed=42, verbose=0)
            m.fit(X_train, y_train, eval_set=(X_val, y_val))
            return compute_r2(y_val, m.predict(X_val))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(f"CatBoost Optuna best R²={study.best_value:.4f}, params={best}")

        self._model = CatBoostRegressor(**best, random_seed=42, verbose=0)
        self._model.fit(
            np.vstack([X_train, X_val]),
            np.concatenate([y_train, y_val]),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
