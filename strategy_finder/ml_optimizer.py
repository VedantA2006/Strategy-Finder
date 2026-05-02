"""
ml_optimizer.py — Advanced GP-based Bayesian optimizer.

Uses GaussianProcessRegressor with Upper Confidence Bound (UCB) acquisition
to balance exploration and exploitation across the 12-feature parameter space.
Maintains separate models per asset.
"""

from __future__ import annotations

import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C


class StrategyOptimizer:
    """
    Maintains a history of strategy parameter vectors and their scores per asset.
    Fits a GP surrogate model and uses UCB to suggest new candidates.
    """

    def __init__(self):
        # Maps asset -> list of vectors
        self.X: dict[str, list[list[float]]] = {}
        self.y: dict[str, list[float]] = {}
        self.models: dict[str, GaussianProcessRegressor] = {}
        self.last_fit_count: dict[str, int] = {}

    def record(self, asset: str, params_vector: list[float], score_val: float) -> None:
        """Record a (params, score) observation for an asset."""
        if asset not in self.X:
            self.X[asset] = []
            self.y[asset] = []
            self.last_fit_count[asset] = 0
            
        self.X[asset].append(params_vector)
        self.y[asset].append(score_val)

    def fit(self, asset: str) -> bool:
        """
        Fit the GP model for an asset if we have enough data (>= 30 samples)
        and at least 25 new observations since the last fit.
        """
        if asset not in self.X or len(self.X[asset]) < 30:
            return False
            
        new_obs = len(self.X[asset]) - self.last_fit_count.get(asset, 0)
        if asset in self.models and new_obs < 25:
            return True # Already fitted, skip refit

        # Matern kernel is good for non-smooth parameter spaces
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1,  # noise level
            n_restarts_optimizer=3,
            random_state=42
        )
        
        X_arr = np.array(self.X[asset])
        y_arr = np.array(self.y[asset])
        
        # Standardize targets for better GP convergence
        y_mean = np.mean(y_arr)
        y_std = np.std(y_arr) if np.std(y_arr) > 0 else 1.0
        y_norm = (y_arr - y_mean) / y_std
        
        try:
            model.fit(X_arr, y_norm)
            self.models[asset] = model
            self.last_fit_count[asset] = len(self.X[asset])
            return True
        except Exception as e:
            print(f"GP Fit failed for {asset}: {e}")
            return False

    def suggest(self, asset: str, n: int = 10) -> list[list[float]]:
        """
        Generate 500 random valid param vectors, predict their UCB scores,
        and return the top-n most promising ones.
        """
        candidates = [self.random_params() for _ in range(500)]

        if not self.fit(asset) or asset not in self.models:
            return candidates[:n]

        X_candidates = np.array(candidates)
        model = self.models[asset]
        
        # Predict mean and std
        mean, std = model.predict(X_candidates, return_std=True)
        
        # UCB Acquisition: balance high expected reward + high uncertainty
        ucb_scores = mean + 2.0 * std

        top_indices = np.argsort(ucb_scores)[::-1][:n]
        return [candidates[i] for i in top_indices]

    @staticmethod
    def random_params() -> list[float]:
        """
        Generate a random valid parameter vector for suggestions.
        Format must match the 12-feature vector in strategy.py.
        """
        sl_mult = round(random.uniform(1.0, 4.0), 1)
        min_rr = round(sl_mult * 1.5 + 0.1, 1)
        rr_ratio = round(random.uniform(min_rr, min_rr + 4.0), 1)
        cooldown = float(random.randint(2, 10))
        atr_gate = round(random.uniform(0.0005, 0.003), 4)
        
        # Randomize the structural features for exploration
        complexity = float(random.randint(2, 6))
        n_and = float(random.randint(1, 4))
        n_or = float(random.randint(0, 2))
        unique_inds = float(random.randint(2, 8))
        tfs = float(random.randint(1, 4))
        
        has_vol = float(random.choice([0, 1]))
        has_regime = float(random.choice([0, 1]))
        has_sess = float(random.choice([0, 1]))
        
        return [
            sl_mult, rr_ratio, cooldown, atr_gate,
            complexity, n_and, n_or, unique_inds, tfs,
            has_vol, has_regime, has_sess
        ]
