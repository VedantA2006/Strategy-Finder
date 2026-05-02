"""
ml_optimizer.py — RandomForest-based parameter suggestion engine.

Learns from (params_vector, score) history to suggest promising
parameter combinations for the next generation.

Always enforces rr_ratio >= sl_mult × 1.5 + 0.1.
"""

from __future__ import annotations

import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class StrategyOptimizer:
    """
    Maintains a history of strategy parameter vectors and their scores.
    Once enough data is collected (≥ 50 samples), fits a RandomForest
    and uses it to predict scores for randomly sampled parameter vectors,
    returning the top-N most promising ones.
    """

    def __init__(self):
        self.X: list[list[float]] = []  # param vectors
        self.y: list[float] = []        # scores
        self.model: RandomForestRegressor | None = None
        self._fitted = False

    def record(self, params_vector: list[float], score_val: float) -> None:
        """Record a (params, score) observation."""
        self.X.append(params_vector)
        self.y.append(score_val)

    def fit(self) -> bool:
        """
        Fit the model if we have enough data.
        Returns True if model was fitted successfully.
        """
        if len(self.X) < 50:
            return False

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)
        self.model.fit(X_arr, y_arr)
        self._fitted = True
        return True

    def suggest(self, n: int = 10) -> list[list[float]]:
        """
        Generate 500 random valid param vectors, predict their scores,
        and return the top-n most promising ones.

        Falls back to pure random if model isn't trained yet.
        """
        candidates = [self.random_params() for _ in range(500)]

        if not self._fitted:
            if not self.fit():
                # Not enough data — return random params
                return candidates[:n]

        X_candidates = np.array(candidates)
        predictions = self.model.predict(X_candidates)

        # Sort by predicted score descending, return top-n
        top_indices = np.argsort(predictions)[::-1][:n]
        return [candidates[i] for i in top_indices]

    @staticmethod
    def random_params() -> list[float]:
        """
        Generate a random valid [sl_mult, rr_ratio, cooldown, atr_gate] vector.
        Always enforces rr_ratio >= sl_mult × 1.5 + 0.1.
        """
        sl_mult = round(random.uniform(1.0, 4.0), 1)
        min_rr = round(sl_mult * 1.5 + 0.1, 1)
        rr_ratio = round(random.uniform(min_rr, min_rr + 4.0), 1)
        cooldown = float(random.randint(2, 10))
        atr_gate = round(random.uniform(0.0005, 0.003), 4)
        return [sl_mult, rr_ratio, cooldown, atr_gate]
