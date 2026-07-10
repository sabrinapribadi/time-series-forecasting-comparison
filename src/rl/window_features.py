"""
Context feature extraction from a time-series lookback window.

Eight features capture the local regime of the series just before a forecast
window, giving the LinUCB bandit signal about which model is likely best.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

FEATURE_NAMES: list[str] = [
    "level",       # normalised mean — series level relative to global scale
    "volatility",  # normalised std  — local spread / noise
    "trend",       # linear slope normalised by scale — rising / falling regime
    "autocorr",    # lag-1 Pearson correlation — persistence / AR structure
    "max_jump",    # max single-step change / scale — shock / spike presence
    "range_ratio", # (max - min) / scale — width of the local range
    "skewness",    # distributional skew — asymmetric tail exposure
    "bias",        # always 1.0 — intercept for LinUCB linear model
]

N_FEATURES = len(FEATURE_NAMES)


def extract_window_features(window: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Extract 8 context features from a 1-D time window.

    Args:
        window: 1-D array of OT values (the lookback window).
        scale:  Global std of the test series used for normalisation.

    Returns:
        Float32 array of shape (8,).
    """
    eps = 1e-8
    w = np.asarray(window, dtype=np.float64)
    n = len(w)
    s = float(scale) + eps

    level = float(np.mean(w)) / s
    volatility = float(np.std(w)) / s
    range_ratio = float(np.ptp(w)) / s

    if n >= 2:
        xs = np.arange(n, dtype=np.float64)
        slope = float(np.polyfit(xs, w, 1)[0])
        trend = slope / s

        r = float(np.corrcoef(w[:-1], w[1:])[0, 1])
        autocorr = r if not np.isnan(r) else 0.0

        max_jump = float(np.max(np.abs(np.diff(w)))) / s
        skewness = float(scipy_stats.skew(w))
    else:
        trend = autocorr = max_jump = skewness = 0.0

    return np.array(
        [level, volatility, trend, autocorr, max_jump, range_ratio, skewness, 1.0],
        dtype=np.float32,
    )
