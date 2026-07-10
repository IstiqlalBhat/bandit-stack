"""Off-policy estimators for non-contextual target policies.

A target policy is expressed as a probability vector over arms. Estimators
answer: "what average reward would this policy have earned on the logged
traffic?" — the counterfactual question behind shadow mode.

- ips: inverse propensity scoring. Unbiased, high variance when the behavior
  policy rarely played the target's preferred arms.
- snips: self-normalized IPS. Slightly biased, much lower variance; the
  default for reports.
- doubly_robust: direct method (per-arm empirical reward means) plus an IPS
  correction on its residuals. Unbiased if either component is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoggedData:
    arms: np.ndarray  # (n,) int — arm chosen by the behavior policy
    propensities: np.ndarray  # (n,) float — behavior probability of that arm
    rewards: np.ndarray  # (n,) float

    def __post_init__(self) -> None:
        n = len(self.arms)
        if len(self.propensities) != n or len(self.rewards) != n:
            raise ValueError("arms, propensities, rewards must have equal length")
        if n == 0:
            raise ValueError("logged data is empty")


def _validate(log: LoggedData, target_probs: np.ndarray) -> np.ndarray:
    target_probs = np.asarray(target_probs, dtype=float)
    if np.any(log.propensities <= 0.0):
        raise ValueError("propensities must be strictly positive")
    if np.any(target_probs < 0.0) or not np.isclose(target_probs.sum(), 1.0, atol=1e-6):
        raise ValueError("target_probs must be a probability distribution")
    if np.any(log.arms >= len(target_probs)) or np.any(log.arms < 0):
        raise ValueError("logged arm index outside target_probs range")
    return target_probs


def _weights(log: LoggedData, target_probs: np.ndarray) -> np.ndarray:
    return target_probs[log.arms] / log.propensities


def ips(log: LoggedData, target_probs: np.ndarray) -> float:
    target_probs = _validate(log, target_probs)
    return float(np.mean(_weights(log, target_probs) * log.rewards))


def snips(log: LoggedData, target_probs: np.ndarray) -> float:
    target_probs = _validate(log, target_probs)
    w = _weights(log, target_probs)
    total = w.sum()
    if total == 0.0:
        raise ValueError("no overlap between logged arms and target policy support")
    return float((w * log.rewards).sum() / total)


def doubly_robust(log: LoggedData, target_probs: np.ndarray) -> float:
    target_probs = _validate(log, target_probs)
    n_arms = len(target_probs)

    # Direct-method component: per-arm empirical mean reward from the log.
    # Arms never logged get the global mean as a fallback.
    q = np.full(n_arms, log.rewards.mean())
    for arm in range(n_arms):
        mask = log.arms == arm
        if mask.any():
            q[arm] = log.rewards[mask].mean()

    direct = float(target_probs @ q)
    correction = float(np.mean(_weights(log, target_probs) * (log.rewards - q[log.arms])))
    return direct + correction
