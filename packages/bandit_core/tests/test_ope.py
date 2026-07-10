"""Off-policy evaluation: estimate a target policy's value from logged
decisions made by a different (behavior) policy.

Ground truth setup: behavior = uniform random over 3 arms (propensity exactly
1/3), environment = Bernoulli with known per-arm probabilities. The true value
of any target policy is then target_probs @ arm_probs, so every estimator can
be checked against an exact answer.
"""

import numpy as np
import pytest

from bandit_core.ope import LoggedData, doubly_robust, ips, snips

ARM_PROBS = np.array([0.2, 0.5, 0.8])


def make_log(n: int, seed: int) -> LoggedData:
    rng = np.random.default_rng(seed)
    arms = rng.integers(0, 3, size=n)
    rewards = (rng.random(n) < ARM_PROBS[arms]).astype(float)
    propensities = np.full(n, 1.0 / 3.0)
    return LoggedData(arms=arms, propensities=propensities, rewards=rewards)


class TestIPS:
    def test_recovers_value_of_fixed_arm_policy(self):
        log = make_log(20_000, seed=0)
        estimate = ips(log, target_probs=np.array([0.0, 0.0, 1.0]))
        assert estimate == pytest.approx(0.8, abs=0.03)

    def test_recovers_value_of_uniform_policy(self):
        log = make_log(20_000, seed=1)
        estimate = ips(log, target_probs=np.array([1 / 3, 1 / 3, 1 / 3]))
        assert estimate == pytest.approx(0.5, abs=0.03)

    def test_rejects_nonpositive_propensity(self):
        log = LoggedData(
            arms=np.array([0]), propensities=np.array([0.0]), rewards=np.array([1.0])
        )
        with pytest.raises(ValueError):
            ips(log, target_probs=np.array([1.0, 0.0]))

    def test_rejects_target_probs_not_summing_to_one(self):
        log = make_log(100, seed=2)
        with pytest.raises(ValueError):
            ips(log, target_probs=np.array([0.5, 0.2, 0.1]))


class TestSNIPS:
    def test_recovers_value_of_fixed_arm_policy(self):
        log = make_log(20_000, seed=3)
        estimate = snips(log, target_probs=np.array([0.0, 0.0, 1.0]))
        assert estimate == pytest.approx(0.8, abs=0.03)

    def test_invariant_to_reward_shift_direction(self):
        # SNIPS normalizes by total importance weight, so a constant reward
        # yields exactly that constant regardless of propensity noise.
        log = make_log(5_000, seed=4)
        shifted = LoggedData(
            arms=log.arms,
            propensities=log.propensities,
            rewards=np.ones_like(log.rewards),
        )
        estimate = snips(shifted, target_probs=np.array([0.0, 1.0, 0.0]))
        assert estimate == pytest.approx(1.0, abs=1e-12)


class TestDoublyRobust:
    def test_recovers_value_of_fixed_arm_policy(self):
        log = make_log(20_000, seed=5)
        estimate = doubly_robust(log, target_probs=np.array([0.0, 0.0, 1.0]))
        assert estimate == pytest.approx(0.8, abs=0.02)

    def test_recovers_value_of_mixed_policy(self):
        log = make_log(20_000, seed=6)
        target = np.array([0.2, 0.3, 0.5])
        truth = float(target @ ARM_PROBS)
        estimate = doubly_robust(log, target_probs=target)
        assert estimate == pytest.approx(truth, abs=0.02)

    def test_no_higher_variance_than_ips(self):
        target = np.array([0.0, 0.0, 1.0])
        ips_estimates = []
        dr_estimates = []
        for seed in range(30):
            log = make_log(2_000, seed=100 + seed)
            ips_estimates.append(ips(log, target_probs=target))
            dr_estimates.append(doubly_robust(log, target_probs=target))
        assert np.std(dr_estimates) <= np.std(ips_estimates) * 1.1
