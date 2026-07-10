import numpy as np
import pytest

from bandit_core import EpsilonGreedy


def test_propensity_is_exact_with_unique_greedy_arm():
    policy = EpsilonGreedy(n_arms=4, epsilon=0.2, seed=0)
    policy.counts[:] = 1.0
    policy.means[:] = [0.1, 0.9, 0.2, 0.3]
    seen = {}
    for _ in range(500):
        d = policy.choose()
        seen[d.arm] = d.propensity
    assert seen[1] == pytest.approx(0.8 + 0.05)
    for arm in (0, 2, 3):
        if arm in seen:
            assert seen[arm] == pytest.approx(0.05)


def test_empirical_frequencies_match_propensities():
    policy = EpsilonGreedy(n_arms=4, epsilon=0.2, seed=1)
    policy.counts[:] = 1.0
    policy.means[:] = [0.1, 0.9, 0.2, 0.3]
    picks = np.array([policy.choose().arm for _ in range(20_000)])
    freq_greedy = np.mean(picks == 1)
    assert freq_greedy == pytest.approx(0.85, abs=0.01)


def test_ties_split_propensity_across_greedy_set():
    policy = EpsilonGreedy(n_arms=4, epsilon=0.0, seed=2)
    policy.means[:] = [0.5, 0.5, 0.1, 0.1]
    d = policy.choose()
    assert d.arm in (0, 1)
    assert d.propensity == pytest.approx(0.5)


def test_incremental_mean_matches_batch_mean():
    policy = EpsilonGreedy(n_arms=2, epsilon=0.1, seed=3)
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0]
    for r in rewards:
        policy.update(0, r)
    assert policy.means[0] == pytest.approx(np.mean(rewards))
    assert policy.counts[0] == len(rewards)
