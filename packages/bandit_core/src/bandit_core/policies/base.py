"""Policy interface shared by all bandit policies.

Every decision carries a propensity — the probability the policy assigned to
the arm it chose. Propensities are what make logged decisions usable for
off-policy evaluation later, so no policy may return a decision without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Decision:
    arm: int
    propensity: float


@runtime_checkable
class BanditPolicy(Protocol):
    def choose(self, arm_features: Sequence[np.ndarray] | None = None) -> Decision: ...

    def update(
        self,
        arm: int,
        reward: float,
        arm_features: Sequence[np.ndarray] | None = None,
    ) -> None: ...

    def state_dict(self) -> dict: ...


def smoothed_propensity(win_count: int, n_samples: int, n_arms: int) -> float:
    """Laplace-smoothed Monte Carlo propensity estimate.

    Smoothing keeps estimates strictly inside (0, 1) so inverse-propensity
    weights stay finite even when the MC sample never picks the chosen arm.
    """
    return (win_count + 1) / (n_samples + n_arms)
