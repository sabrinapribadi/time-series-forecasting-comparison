"""
Statistical forecasting models: Holt-Winters, ARIMA/SARIMA, Prophet.

Each model exposes a minimal sklearn-style interface:
    fit(y_train, dates_train=None)
    predict(horizon, dates_future=None) -> np.ndarray

Optional Optuna HPO for HoltWinters:
    tune_with_optuna(y_train, n_trials=15) — minimize RMSE on train-fit residuals
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HoltWintersModel:
    """
    Exponential Smoothing with Trend + Seasonality (Holt-Winters).
    Wraps statsmodels ExponentialSmoothing.
    """

    def __init__(
        self,
        trend: str = "add",
        seasonal: str = "add",
        seasonal_periods: int = 24,
        damped_trend: bool = False,
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self._model = None
        self._result = None

    def fit(self, y_train: np.ndarray, **kwargs) -> "HoltWintersModel":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
        )
        self._result = self._model.fit(optimized=True, **kwargs)
        logger.info(f"HoltWinters fitted (AIC={self._result.aic:.2f})")
        return self

    def tune_with_optuna(
        self, y_train: np.ndarray, n_trials: int = 15
    ) -> "HoltWintersModel":
        """
        Optuna HPO: 15 trials, minimize RMSE on train-fit residuals.
        Updates self.trend/seasonal/damped_trend and calls fit() with the best config.
        """
        import optuna
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from src.evaluation.metrics import compute_rmse

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            trend = trial.suggest_categorical("trend", ["add", "mul", None])
            seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
            damped = trial.suggest_categorical("damped_trend", [True, False])

            # mul seasonal requires positive data; skip invalid combos
            if trend is None and damped:
                return float("inf")

            try:
                m = ExponentialSmoothing(
                    y_train,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=self.seasonal_periods,
                    damped_trend=damped,
                )
                r = m.fit(optimized=True)
                return compute_rmse(y_train, r.fittedvalues)
            except Exception:
                return float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        logger.info(f"HoltWinters Optuna best RMSE={study.best_value:.4f}, params={best}")
        self.trend = best["trend"]
        self.seasonal = best["seasonal"]
        self.damped_trend = best["damped_trend"]
        self.fit(y_train)
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict()")
        return self._result.forecast(horizon)


class ARIMAModel:
    """
    ARIMA / SARIMA model.
    Uses pmdarima auto_arima for order selection if order is None.
    """

    def __init__(
        self,
        order: Optional[tuple[int, int, int]] = None,
        seasonal_order: Optional[tuple[int, int, int, int]] = None,
        auto: bool = True,
        seasonal: bool = False,
        m: int = 1,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto = auto
        self.seasonal = seasonal
        self.m = m
        self._model = None

    def fit(self, y_train: np.ndarray, **kwargs) -> "ARIMAModel":
        if self.auto or self.order is None:
            import pmdarima as pm

            self._model = pm.auto_arima(
                y_train,
                seasonal=self.seasonal,
                m=self.m,
                test="adf",
                stepwise=True,
                suppress_warnings=True,
                **kwargs,
            )
            logger.info(f"Auto-ARIMA selected order={self._model.order}")
        else:
            from statsmodels.tsa.arima.model import ARIMA

            arima = ARIMA(y_train, order=self.order, seasonal_order=self.seasonal_order)
            self._model = arima.fit()
            logger.info(f"ARIMA{self.order} fitted")
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        preds = self._model.predict(n_periods=horizon, **kwargs)
        return np.asarray(preds)


class ProphetModel:
    """
    Facebook Prophet — handles trend + seasonality + holidays automatically.
    Requires y_train as a 1-D array and dates_train as a DatetimeIndex or Series.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model = None
        self._freq: Optional[str] = None

    def fit(
        self, y_train: np.ndarray, dates_train: pd.DatetimeIndex | pd.Series | None = None
    ) -> "ProphetModel":
        from prophet import Prophet

        if dates_train is None:
            raise ValueError("ProphetModel requires dates_train")

        df = pd.DataFrame({"ds": pd.to_datetime(dates_train), "y": y_train})
        self._freq = pd.infer_freq(df["ds"])

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        self._model.fit(df)
        logger.info("Prophet fitted")
        return self

    def predict(self, horizon: int, **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        future = self._model.make_future_dataframe(
            periods=horizon, freq=self._freq or "h", include_history=False
        )
        forecast = self._model.predict(future)
        return forecast["yhat"].values
