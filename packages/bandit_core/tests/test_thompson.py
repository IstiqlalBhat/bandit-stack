import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bandit_core import BetaBernoulliTS, GaussianTS


class TestBetaBernoulliTS:
    def test_posterior_counts_track_rewards(self):
        policy = BetaBernoulliTS(n_arms=3, seed=0)
        policy.update(0, 1.0)
        policy.update(0, 1.0)
        policy.update(0, 0.0)
        policy.update(2, 0.5)
        assert policy.alpha.tolist() == [3.0, 1.0, 1.5]
        assert policy.beta.tolist() == [2.0, 1.0, 1.5]

    def test_rejects_out_of_range_reward(self):
        policy = BetaBernoulliTS(n_arms=2, seed=0)
        with pytest.raises(ValueError):
            policy.update(0, 1.5)
        with pytest.raises(ValueError):
            policy.update(0, -0.1)

    def test_dominant_arm_gets_high_propensity(self):
        policy = BetaBernoulliTS(n_arms=2, propensity_samples=500, seed=0)
        policy.alpha[:] = [100.0, 1.0]
        policy.beta[:] = [1.0, 100.0]
        decision = policy.choose()
        assert decision.arm == 0
        assert decision.propensity > 0.9

    def test_propensity_within_open_unit_interval(self):
        policy = BetaBernoulliTS(n_arms=4, propensity_samples=50, seed=1)
        for _ in range(50):
            d = policy.choose()
            assert 0.0 < d.propensity < 1.0
            policy.update(d.arm, float(policy.rng.random() < 0.5))

    def test_seeded_runs_are_deterministic(self):
        a = BetaBernoulliTS(n_arms=3, seed=42)
        b = BetaBernoulliTS(n_arms=3, seed=42)
        for _ in range(20):
            da, db = a.choose(), b.choose()
            assert da == db
            a.update(da.arm, 1.0)
            b.update(db.arm, 1.0)

    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50))
    def test_posterior_state_stays_valid(self, rewards):
        policy = BetaBernoulliTS(n_arms=2, seed=0)
        for r in rewards:
            policy.update(0, r)
        assert np.all(policy.alpha > 0)
        assert np.all(policy.beta > 0)
        assert policy.alpha[0] + policy.beta[0] == pytest.approx(2.0 + len(rewards))


class TestGaussianTS:
    def test_posterior_mean_converges_to_empirical_mean(self):
        policy = GaussianTS(n_arms=2, prior_var=100.0, obs_var=1.0, seed=0)
        rewards = [3.0, 3.5, 2.5, 3.0, 3.2, 2.8]
        for r in rewards:
            policy.update(0, r)
        mean, var = policy._posterior()
        assert mean[0] == pytest.approx(np.mean(rewards), abs=0.05)
        assert var[0] < var[1]

    def test_dominant_arm_gets_high_propensity(self):
        policy = GaussianTS(n_arms=2, propensity_samples=500, seed=0)
        for _ in range(200):
            policy.update(0, 10.0)
            policy.update(1, 0.0)
        decision = policy.choose()
        assert decision.arm == 0
        assert decision.propensity > 0.9

    def test_state_dict_round_trips_values(self):
        policy = GaussianTS(n_arms=2, seed=0)
        policy.update(1, 2.0)
        state = policy.state_dict()
        assert state["counts"] == [0.0, 1.0]
        assert state["reward_sums"] == [0.0, 2.0]
