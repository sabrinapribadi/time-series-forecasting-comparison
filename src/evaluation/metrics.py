"""
Evaluation metrics for time series forecasting.

All functions accept 1-D numpy arrays (y_true, y_pred) and return a scalar float.
MASE additionally requires the naive baseline error (in-sample mean absolute diff).
"""

from __future__ import annotations

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Lower is better."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. Lower is better."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, min_abs: float = 0.1) -> float:
    """
    Mean Absolute Percentage Error (%). Lower is better.
    Skips entries where |y_true| < min_abs (near-zero or negative values make MAPE
    meaningless — common in ETT where OT can go negative in winter months).
    Returns NaN if no valid entries remain.
    """
    mask = np.abs(y_true) >= min_abs
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%). Lower is better.
    Range [0, 200]. Symmetric around the zero-error axis unlike MAPE.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Mean Absolute Scaled Error. Lower is better; <1 beats naive seasonal baseline.

    Args:
        y_true:      Ground truth values.
        y_pred:      Model predictions.
        y_train:     In-sample training series used to compute the naive baseline.
        seasonality: Period m for seasonal naive (m=1 → random walk naive).
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) + 1e-8
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²). Higher is better; 1.0 is perfect."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    return float(1 - ss_res / ss_tot)


def compute_mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Directional Accuracy (%). Higher is better; 50% = random.
    Measures how often the forecast direction (up/down) matches actual.
    From IOH congestion metrics library.
    """
    if len(y_true) < 2:
        return float("nan")
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir) * 100)


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Forecast Bias — mean signed error (positive = over-forecast, negative = under-forecast).
    From IOH congestion metrics library. Range: (-∞, +∞), ideal = 0.
    """
    return float(np.mean(y_pred - y_true))


def compute_maape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Arctangent Absolute Percentage Error (degrees). Lower is better.
    Uses arctan instead of direct ratio — handles near-zero actuals gracefully.
    From IOH congestion metrics library.
    """
    return float(np.mean(np.arctan(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))) * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
) -> dict[str, float]:
    """
    Compute all metrics and return as a dictionary.

    Args:
        y_true:      Ground truth test values.
        y_pred:      Model predictions.
        y_train:     Training series (required for MASE; skipped if None).
        seasonality: Passed to compute_mase.
    """
    results: dict[str, float] = {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
        "SMAPE": compute_smape(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
        "MDA": compute_mda(y_true, y_pred),
        "Bias": compute_bias(y_true, y_pred),
        "MAAPE": compute_maape(y_true, y_pred),
    }
    if y_train is not None:
        results["MASE"] = compute_mase(y_true, y_pred, y_train, seasonality)
    return results
