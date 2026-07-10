"""Integration tests: the M1 exit criteria.

Thompson sampling must beat epsilon-greedy on a Bernoulli bandit, its regret
must be sublinear, and LinTS must beat a context-blind baseline on a
contextual environment. Seeds are fixed so these are deterministic.
"""

import numpy as np

from bandit_core import BetaBernoulliTS, EpsilonGreedy, LinTS
from bandit_core.simulation import BernoulliEnv, LinearContextualEnv, compare

HORIZON = 2000
N_SEEDS = 4
PROBS = [0.3, 0.5, 0.7]


def _bernoulli_results():
    return compare(
        policy_factories={
            "ts": lambda s: BetaBernoulliTS(
                n_arms=3, propensity_samples=32, seed=1000 + s
            ),
            "eps_greedy_0.1": lambda s: EpsilonGreedy(n_arms=3, epsilon=0.1, seed=1000 + s),
        },
        env_factory=lambda s: BernoulliEnv(PROBS, seed=2000 + s),
        horizon=HORIZON,
        n_seeds=N_SEEDS,
    )


def test_thompson_beats_epsilon_greedy_on_bernoulli():
    results = _bernoulli_results()
    ts_final = results["ts"][:, -1].mean()
    eg_final = results["eps_greedy_0.1"][:, -1].mean()
    assert ts_final < eg_final, f"TS regret {ts_final:.1f} vs EG {eg_final:.1f}"


def test_thompson_regret_is_sublinear():
    results = _bernoulli_results()
    curve = results["ts"].mean(axis=0)
    # Linear regret doubles from T/2 to T; require clearly slower growth.
    ratio = curve[-1] / curve[HORIZON // 2 - 1]
    assert ratio < 1.6, f"regret growth ratio {ratio:.2f} suggests linear regret"


def test_lin_ts_beats_context_blind_baseline():
    theta = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    results = compare(
        policy_factories={
            "lin_ts": lambda s: LinTS(dim=5, propensity_samples=32, seed=1000 + s),
            "eps_greedy_0.1": lambda s: EpsilonGreedy(n_arms=4, epsilon=0.1, seed=1000 + s),
        },
        env_factory=lambda s: LinearContextualEnv(theta, n_arms=4, noise=0.1, seed=2000 + s),
        horizon=1500,
        n_seeds=N_SEEDS,
    )
    lin_final = results["lin_ts"][:, -1].mean()
    eg_final = results["eps_greedy_0.1"][:, -1].mean()
    assert lin_final < 0.5 * eg_final, (
        f"LinTS regret {lin_final:.1f} not clearly better than blind EG {eg_final:.1f}"
    )
