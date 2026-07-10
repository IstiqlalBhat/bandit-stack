"""Linear Thompson sampling for contextual bandits.

Bayesian linear regression over a shared weight vector: each arm is described
by a d-dimensional feature vector (request features x arm features, encoded by
the caller); reward is modeled as x @ theta + noise. The posterior covariance
is maintained via Sherman-Morrison so updates are O(d^2).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bandit_core.policies.base import Decision, smoothed_propensity


class LinTS:
    def __init__(
        self,
        dim: int,
        v: float = 1.0,
        lambda_reg: float = 1.0,
        propensity_samples: int = 200,
        seed: int | None = None,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if v <= 0 or lambda_reg <= 0:
            raise ValueError("v and lambda_reg must be positive")
        self.dim = dim
        self.v = v
        self.a_inv = np.eye(dim) / lambda_reg
        self.b = np.zeros(dim)
        self.propensity_samples = propensity_samples
        self.rng = np.random.default_rng(seed)

    def _stack(self, arm_features: Sequence[np.ndarray]) -> np.ndarray:
        x = np.asarray(arm_features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"arm_features must be (n_arms, {self.dim}), got {x.shape}")
        if x.shape[0] < 2:
            raise ValueError("LinTS.choose needs at least 2 arms")
        return x

    def _sample_thetas(self, k: int) -> np.ndarray:
        mu = self.a_inv @ self.b
        cov = self.v**2 * (self.a_inv + self.a_inv.T) / 2.0
        return self.rng.multivariate_normal(mu, cov, size=k)

    def choose(self, arm_features: Sequence[np.ndarray] | None = None) -> Decision:
        if arm_features is None:
            raise ValueError("LinTS is contextual: arm_features is required")
        x = self._stack(arm_features)
        theta = self._sample_thetas(1)[0]
        arm = int(np.argmax(x @ theta))
        return Decision(arm=arm, propensity=self._propensity(arm, x))

    def _propensity(self, arm: int, x: np.ndarray) -> float:
        k = self.propensity_samples
        thetas = self._sample_thetas(k)
        scores = x @ thetas.T  # (n_arms, k)
        wins = int(np.sum(np.argmax(scores, axis=0) == arm))
        return smoothed_propensity(wins, k, x.shape[0])

    def update(
        self,
        arm: int,
        reward: float,
        arm_features: Sequence[np.ndarray] | None = None,
    ) -> None:
        if arm_features is None:
            raise ValueError("LinTS is contextual: arm_features is required")
        x = np.asarray(arm_features[arm], dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"chosen arm features must be ({self.dim},), got {x.shape}")
        ax = self.a_inv @ x
        self.a_inv -= np.outer(ax, ax) / (1.0 + x @ ax)
        self.b += reward * x

    def state_dict(self) -> dict:
        return {
            "policy": "lin_ts",
            "dim": self.dim,
            "v": self.v,
            "a_inv": self.a_inv.tolist(),
            "b": self.b.tolist(),
        }
