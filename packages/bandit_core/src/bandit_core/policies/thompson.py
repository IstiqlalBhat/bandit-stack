"""Thompson sampling policies for non-contextual bandits."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bandit_core.policies.base import Decision, smoothed_propensity


class BetaBernoulliTS:
    """Thompson sampling with per-arm Beta posteriors.

    Rewards must lie in [0, 1]; fractional rewards use the standard relaxation
    alpha += r, beta += 1 - r.
    """

    def __init__(
        self,
        n_arms: int,
        prior: tuple[float, float] = (1.0, 1.0),
        propensity_samples: int = 200,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be >= 2")
        if prior[0] <= 0 or prior[1] <= 0:
            raise ValueError("Beta prior parameters must be positive")
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, prior[0], dtype=float)
        self.beta = np.full(n_arms, prior[1], dtype=float)
        self.propensity_samples = propensity_samples
        self.rng = np.random.default_rng(seed)

    def choose(self, arm_features: Sequence[np.ndarray] | None = None) -> Decision:
        theta = self.rng.beta(self.alpha, self.beta)
        arm = int(np.argmax(theta))
        return Decision(arm=arm, propensity=self._propensity(arm))

    def _propensity(self, arm: int) -> float:
        k = self.propensity_samples
        samples = self.rng.beta(
            self.alpha[:, None], self.beta[:, None], size=(self.n_arms, k)
        )
        wins = int(np.sum(np.argmax(samples, axis=0) == arm))
        return smoothed_propensity(wins, k, self.n_arms)

    def update(
        self,
        arm: int,
        reward: float,
        arm_features: Sequence[np.ndarray] | None = None,
    ) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"BetaBernoulliTS requires reward in [0, 1], got {reward}")
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def state_dict(self) -> dict:
        return {
            "policy": "beta_bernoulli_ts",
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
        }


class GaussianTS:
    """Thompson sampling with Normal posteriors (known observation variance).

    Conjugate update for unknown mean: prior N(prior_mean, prior_var) per arm,
    observations N(mu_arm, obs_var).
    """

    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        obs_var: float = 1.0,
        propensity_samples: int = 200,
        seed: int | None = None,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be >= 2")
        if prior_var <= 0 or obs_var <= 0:
            raise ValueError("variances must be positive")
        self.n_arms = n_arms
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.obs_var = obs_var
        self.counts = np.zeros(n_arms, dtype=float)
        self.reward_sums = np.zeros(n_arms, dtype=float)
        self.propensity_samples = propensity_samples
        self.rng = np.random.default_rng(seed)

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        precision = 1.0 / self.prior_var + self.counts / self.obs_var
        var = 1.0 / precision
        mean = var * (self.prior_mean / self.prior_var + self.reward_sums / self.obs_var)
        return mean, var

    def choose(self, arm_features: Sequence[np.ndarray] | None = None) -> Decision:
        mean, var = self._posterior()
        theta = self.rng.normal(mean, np.sqrt(var))
        arm = int(np.argmax(theta))
        return Decision(arm=arm, propensity=self._propensity(arm, mean, var))

    def _propensity(self, arm: int, mean: np.ndarray, var: np.ndarray) -> float:
        k = self.propensity_samples
        samples = self.rng.normal(
            mean[:, None], np.sqrt(var)[:, None], size=(self.n_arms, k)
        )
        wins = int(np.sum(np.argmax(samples, axis=0) == arm))
        return smoothed_propensity(wins, k, self.n_arms)

    def update(
        self,
        arm: int,
        reward: float,
        arm_features: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.counts[arm] += 1.0
        self.reward_sums[arm] += reward

    def state_dict(self) -> dict:
        return {
            "policy": "gaussian_ts",
            "counts": self.counts.tolist(),
            "reward_sums": self.reward_sums.tolist(),
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "obs_var": self.obs_var,
        }
