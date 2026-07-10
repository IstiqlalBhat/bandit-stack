"""Simulation runner: play a policy against an environment, track regret."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bandit_core.policies.base import BanditPolicy


@dataclass
class SimResult:
    cumulative_regret: np.ndarray  # (horizon,)
    arms: np.ndarray  # (horizon,)
    propensities: np.ndarray  # (horizon,)


def simulate(policy: BanditPolicy, env, horizon: int) -> SimResult:
    instantaneous = np.empty(horizon)
    arms = np.empty(horizon, dtype=int)
    propensities = np.empty(horizon)
    for t in range(horizon):
        arm_features = env.round()
        decision = policy.choose(arm_features)
        reward = env.pull(decision.arm, arm_features)
        policy.update(decision.arm, reward, arm_features)
        instantaneous[t] = env.regret(decision.arm, arm_features)
        arms[t] = decision.arm
        propensities[t] = decision.propensity
    return SimResult(
        cumulative_regret=np.cumsum(instantaneous),
        arms=arms,
        propensities=propensities,
    )


def compare(
    policy_factories: dict[str, Callable[[int], BanditPolicy]],
    env_factory: Callable[[int], object],
    horizon: int,
    n_seeds: int = 5,
) -> dict[str, np.ndarray]:
    """Run each policy over n_seeds independent environment draws.

    Factories take a seed so every (policy, seed) pair is reproducible and
    all policies face identically-seeded environments.

    Returns {policy_name: (n_seeds, horizon) cumulative regret}.
    """
    results: dict[str, np.ndarray] = {}
    for name, make_policy in policy_factories.items():
        runs = np.empty((n_seeds, horizon))
        for s in range(n_seeds):
            policy = make_policy(s)
            env = env_factory(s)
            runs[s] = simulate(policy, env, horizon).cumulative_regret
        results[name] = runs
    return results
