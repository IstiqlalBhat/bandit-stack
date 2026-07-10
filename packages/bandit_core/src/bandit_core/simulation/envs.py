"""Synthetic environments with known optimal arms.

All environments share one interface so the harness never branches:
`round()` returns the arm features for this round (None for non-contextual
environments), `pull(arm, arm_features)` returns a sampled reward, and
`regret(arm, arm_features)` returns the expected-reward gap versus the
optimal arm for that round.
"""

from __future__ import annotations

import numpy as np


class BernoulliEnv:
    def __init__(self, probs: list[float], seed: int | None = None) -> None:
        self.probs = np.asarray(probs, dtype=float)
        if np.any((self.probs < 0) | (self.probs > 1)):
            raise ValueError("probs must be in [0, 1]")
        self.n_arms = len(probs)
        self.rng = np.random.default_rng(seed)

    def round(self) -> None:
        return None

    def pull(self, arm: int, arm_features=None) -> float:
        return float(self.rng.random() < self.probs[arm])

    def regret(self, arm: int, arm_features=None) -> float:
        return float(self.probs.max() - self.probs[arm])


class GaussianEnv:
    def __init__(self, means: list[float], sigma: float = 1.0, seed: int | None = None) -> None:
        self.means = np.asarray(means, dtype=float)
        self.sigma = sigma
        self.n_arms = len(means)
        self.rng = np.random.default_rng(seed)

    def round(self) -> None:
        return None

    def pull(self, arm: int, arm_features=None) -> float:
        return float(self.rng.normal(self.means[arm], self.sigma))

    def regret(self, arm: int, arm_features=None) -> float:
        return float(self.means.max() - self.means[arm])


class LinearContextualEnv:
    """Reward = x_arm @ theta + Gaussian noise, with fresh random unit-norm
    arm features each round — so the best arm changes round to round and a
    non-contextual policy cannot win."""

    def __init__(
        self,
        theta: list[float] | np.ndarray,
        n_arms: int,
        noise: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.dim = len(self.theta)
        self.n_arms = n_arms
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def round(self) -> np.ndarray:
        x = self.rng.normal(size=(self.n_arms, self.dim))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def pull(self, arm: int, arm_features: np.ndarray = None) -> float:
        mean = float(arm_features[arm] @ self.theta)
        return mean + float(self.rng.normal(0.0, self.noise))

    def regret(self, arm: int, arm_features: np.ndarray = None) -> float:
        expected = arm_features @ self.theta
        return float(expected.max() - expected[arm])
