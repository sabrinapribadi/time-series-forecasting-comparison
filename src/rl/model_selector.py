"""
Contextual bandit model selector.

Trains a LinUCB bandit offline on pre-computed forecast results.
At each 24-hour window, the bandit observes features of the preceding window
and selects the forecasting model predicted to have the lowest RMSE.

Training uses full-feedback offline updates: all 11 model arms are updated
at every window, since the pre-computed forecasts give us each arm's reward
for every context — no simulator required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.rl.bandit import LinUCBBandit
from src.rl.window_features import FEATURE_NAMES, N_FEATURES, extract_window_features

# Models present in data/forecasts/ETT/ (one JSON per model)
_CANDIDATE_MODELS = [
    "arima", "catboost", "holt_winters", "lightgbm", "lstm",
    "nbeats", "prophet", "random_forest", "tft", "transformer", "xgboost",
]

# Multivariate models have 3,480 test points; univariate have 3,485.
# The univariate series starts 5 rows earlier: univariate[5:] == multivariate[:].
# We align everything to the shorter 3,480-point common window.
_MULTIV_LEN = 3480
_UNIV_OFFSET = 5   # univariate[5:] aligns with multivariate[:]


class BanditModelSelector:
    """
    Train and evaluate a LinUCB contextual bandit on pre-computed ETT forecasts.

    Usage::

        selector = BanditModelSelector(FORECAST_DIR, alpha=0.5)
        selector.train()
        results = selector.get_results()   # serialisable dict for caching
    """

    WINDOW_SIZE = 24  # one 24-hour forecast horizon per bandit step

    def __init__(
        self,
        forecast_dir: str | Path,
        alpha: float = 0.5,
    ) -> None:
        self.forecast_dir = Path(forecast_dir)
        self.alpha = alpha
        self._trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full offline training loop."""
        forecasts = self._load_forecasts()
        self.arm_names: list[str] = sorted(forecasts.keys())
        n_arms = len(self.arm_names)

        # Use the common 3,480-point test window (multivariate alignment)
        y_true: np.ndarray = forecasts[self.arm_names[0]]["y_true"]
        n = len(y_true)                              # 3480
        n_windows = n // self.WINDOW_SIZE            # 145

        # Per-window RMSE matrix: shape (n_windows, n_arms)
        rmse_matrix = np.zeros((n_windows, n_arms), dtype=np.float64)
        for j, arm in enumerate(self.arm_names):
            yp = forecasts[arm]["y_pred"]
            for w in range(n_windows):
                s = w * self.WINDOW_SIZE
                e = s + self.WINDOW_SIZE
                rmse_matrix[w, j] = float(np.sqrt(np.mean((y_true[s:e] - yp[s:e]) ** 2)))

        # Reward: 1.0 = best arm in this window, 0.0 = worst arm
        min_r = rmse_matrix.min(axis=1, keepdims=True)
        max_r = rmse_matrix.max(axis=1, keepdims=True)
        reward_matrix = 1.0 - (rmse_matrix - min_r) / (max_r - min_r + 1e-8)

        # Context features: extract from the 24h lookback before each window
        scale = float(np.std(y_true))
        contexts = np.zeros((n_windows, N_FEATURES), dtype=np.float32)
        for w in range(n_windows):
            lb_end = w * self.WINDOW_SIZE
            lb_start = max(0, lb_end - self.WINDOW_SIZE)
            lookback = y_true[lb_start:lb_end] if lb_end > 0 else y_true[:self.WINDOW_SIZE]
            contexts[w] = extract_window_features(lookback, scale=scale)

        # Sequential offline training with rolling evaluation
        bandit = LinUCBBandit(n_arms, N_FEATURES, alpha=self.alpha)
        selections = np.zeros(n_windows, dtype=int)
        selected_rmses = np.zeros(n_windows, dtype=np.float64)

        for w in range(n_windows):
            ctx = contexts[w]
            # Record what the bandit would pick NOW (before updating on window w)
            arm, _ = bandit.select_arm(ctx)
            selections[w] = arm
            selected_rmses[w] = rmse_matrix[w, arm]
            # Full-feedback update: all arms observe their reward
            for a in range(n_arms):
                bandit.update(a, ctx, float(reward_matrix[w, a]))

        self._bandit = bandit
        self._rmse_matrix = rmse_matrix
        self._contexts = contexts
        self._selections = selections
        self._selected_rmses = selected_rmses
        self._oracle_rmses = rmse_matrix.min(axis=1)
        self._y_true = y_true
        self._scale = scale
        self._n_windows = n_windows
        self._n_arms = n_arms
        self._trained = True

    def get_results(self) -> dict:
        """
        Return a fully serialisable results dict for Streamlit caching.
        Includes: KPI metrics, per-window arrays, cumulative regret, learned weights.
        """
        if not self._trained:
            self.train()

        avg_per_arm = self._rmse_matrix.mean(axis=0)
        static_best_arm = int(np.argmin(avg_per_arm))

        # Baselines
        random_rmse_per_window = self._rmse_matrix.mean(axis=1)

        # Cumulative regret vs oracle
        regret_bandit = np.cumsum(self._selected_rmses - self._oracle_rmses)
        regret_static = np.cumsum(self._rmse_matrix[:, static_best_arm] - self._oracle_rmses)
        regret_random = np.cumsum(random_rmse_per_window - self._oracle_rmses)

        # Arm selection counts
        counts = np.bincount(self._selections, minlength=self._n_arms)

        # Learned weights (theta per arm — one vector per arm)
        weights = self._bandit.get_weights()

        # Feature statistics for dashboard slider calibration
        feat_min = self._contexts.min(axis=0).tolist()
        feat_max = self._contexts.max(axis=0).tolist()
        feat_mean = self._contexts.mean(axis=0).tolist()

        return {
            # Metadata
            "arm_names": self.arm_names,
            "feature_names": FEATURE_NAMES,
            "n_windows": self._n_windows,
            "alpha": self.alpha,
            "scale": float(self._scale),
            # KPI averages
            "bandit_avg_rmse": float(np.mean(self._selected_rmses)),
            "oracle_avg_rmse": float(np.mean(self._oracle_rmses)),
            "static_best_name": self.arm_names[static_best_arm],
            "static_best_avg_rmse": float(avg_per_arm[static_best_arm]),
            "random_avg_rmse": float(np.mean(random_rmse_per_window)),
            # Per-arm summary
            "arm_avg_rmses": avg_per_arm.tolist(),
            "arm_selection_counts": counts.tolist(),
            # Per-window series
            "per_window_bandit_rmse": self._selected_rmses.tolist(),
            "per_window_oracle_rmse": self._oracle_rmses.tolist(),
            "per_window_static_rmse": self._rmse_matrix[:, static_best_arm].tolist(),
            "per_window_random_rmse": random_rmse_per_window.tolist(),
            "per_window_selections": self._selections.tolist(),
            "rmse_matrix": self._rmse_matrix.tolist(),
            # Cumulative regret
            "regret_bandit": regret_bandit.tolist(),
            "regret_static": regret_static.tolist(),
            "regret_random": regret_random.tolist(),
            # Learned parameters for what-if predictor
            "weights": [w.tolist() for w in weights],
            # Feature statistics for slider calibration
            "feat_min": feat_min,
            "feat_max": feat_max,
            "feat_mean": feat_mean,
        }

    def predict_from_features(self, context: np.ndarray) -> tuple[str, list[float]]:
        """
        Given a pre-built context vector, return (model_name, linear_scores_per_arm).

        Uses the learned theta weights only (no UCB bonus), so the score is a
        pure expected-reward estimate — appropriate for the what-if predictor.
        """
        if not self._trained:
            self.train()
        weights = self._bandit.get_weights()
        scores = [float(theta @ context.astype(np.float64)) for theta in weights]
        best = int(np.argmax(scores))
        return self.arm_names[best], scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_forecasts(self) -> dict[str, dict]:
        """
        Load forecast JSONs and align all models to the common 3,480-point window.

        Multivariate models (catboost, lightgbm, random_forest, xgboost) have
        3,480 test points.  Univariate models have 3,485; their first 5 values
        correspond to 5 rows before the multivariate window starts, so we trim
        y_true and y_pred to align: univariate[5:] == multivariate[:].
        """
        forecasts: dict[str, dict] = {}
        for model in _CANDIDATE_MODELS:
            path = self.forecast_dir / f"{model}_h1.json"
            if not path.exists():
                continue
            raw = json.loads(path.read_text())
            y_true = np.array(raw["y_true"], dtype=np.float64)
            y_pred = np.array(raw["y_pred"], dtype=np.float64)

            # Align to the common 3,480-point window
            if len(y_true) > _MULTIV_LEN:
                offset = len(y_true) - _MULTIV_LEN
                y_true = y_true[offset:]
                y_pred = y_pred[offset:]

            forecasts[model] = {"y_true": y_true, "y_pred": y_pred}

        if not forecasts:
            raise FileNotFoundError(
                f"No forecast JSON files found in {self.forecast_dir}. "
                "Run scripts/run_inference.py first."
            )
        return forecasts
