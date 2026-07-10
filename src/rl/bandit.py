"""
LinUCB contextual bandit for model selection.

Reference: Li et al., 2010 — "A Contextual-Bandit Approach to Personalized News Article Recommendation"
"""

from __future__ import annotations

import numpy as np


class LinUCBBandit:
    """
    Linear Upper Confidence Bound bandit.

    Each arm a has a parameter vector theta_a estimated by ridge regression on
    (context, reward) pairs. The UCB score adds an exploration bonus proportional
    to the uncertainty of the current context under arm a's model:

        score_a(x) = theta_a^T x  +  alpha * sqrt(x^T A_a^{-1} x)

    With full-feedback offline updates (all arms updated each round), the bandit
    converges to the best arm quickly while still adapting to context.
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        # A[a] = I + sum of outer(x, x) — ridge-regularised feature covariance
        self.A: list[np.ndarray] = [np.eye(n_features, dtype=np.float64) for _ in range(n_arms)]
        # b[a] = sum of reward * x
        self.b: list[np.ndarray] = [np.zeros(n_features, dtype=np.float64) for _ in range(n_arms)]

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        ctx = context.astype(np.float64)
        self.A[arm] += np.outer(ctx, ctx)
        self.b[arm] += reward * ctx

    def select_arm(self, context: np.ndarray) -> tuple[int, np.ndarray]:
        """Return (best_arm_index, ucb_scores_per_arm)."""
        ctx = context.astype(np.float64)
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
                # A_a^{-1} x  via solve (numerically better than explicit inverse)
                inv_A_ctx = np.linalg.solve(self.A[a], ctx)
                conf = float(np.sqrt(max(ctx @ inv_A_ctx, 0.0)))
                scores[a] = float(theta @ ctx) + self.alpha * conf
            except np.linalg.LinAlgError:
                scores[a] = 0.0
        return int(np.argmax(scores)), scores

    def get_weights(self) -> list[np.ndarray]:
        """Return the learned theta vector (expected reward estimator) per arm."""
        weights = []
        for a in range(self.n_arms):
            try:
                weights.append(np.linalg.solve(self.A[a], self.b[a]))
            except np.linalg.LinAlgError:
                weights.append(np.zeros(self.n_features))
        return weights
