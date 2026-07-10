"""Epsilon-greedy baseline with exact propensities."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bandit_core.policies.base import Decision

_TIE_TOL = 1e-12


class EpsilonGreedy:
    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be >= 2")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=float)
        self.means = np.zeros(n_arms, dtype=float)
        self.rng = np.random.default_rng(seed)

    def _greedy_set(self) -> np.ndarray:
        return np.flatnonzero(self.means >= self.means.max() - _TIE_TOL)

    def choose(self, arm_features: Sequence[np.ndarray] | None = None) -> Decision:
        greedy = self._greedy_set()
        if self.rng.random() < self.epsilon:
            arm = int(self.rng.integers(self.n_arms))
        else:
            arm = int(self.rng.choice(greedy))
        return Decision(arm=arm, propensity=self._propensity(arm, greedy))

    def _propensity(self, arm: int, greedy: np.ndarray) -> float:
        p = self.epsilon / self.n_arms
        if arm in greedy:
            p += (1.0 - self.epsilon) / len(greedy)
        return float(p)

    def update(
        self,
        arm: int,
        reward: float,
        arm_features: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.counts[arm] += 1.0
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]

    def state_dict(self) -> dict:
        return {
            "policy": "epsilon_greedy",
            "epsilon": self.epsilon,
            "counts": self.counts.tolist(),
            "means": self.means.tolist(),
        }
