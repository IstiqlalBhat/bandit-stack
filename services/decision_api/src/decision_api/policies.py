"""Build bandit policies from route config and restore posterior state from
snapshots. Non-contextual policies only for now; LinTS joins when the LLM
proxy (M3) starts supplying request features."""

from __future__ import annotations

import numpy as np

from bandit_core import BanditPolicy, BetaBernoulliTS, EpsilonGreedy, GaussianTS

POLICY_TYPES = ("beta_ts", "gaussian_ts", "epsilon_greedy")
_NUMBER = (int, float)
POLICY_PARAMS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "beta_ts": {
        "prior_alpha": _NUMBER,
        "prior_beta": _NUMBER,
        "propensity_samples": int,
    },
    "gaussian_ts": {
        "prior_mean": _NUMBER,
        "prior_var": _NUMBER,
        "obs_var": _NUMBER,
        "propensity_samples": int,
    },
    "epsilon_greedy": {"epsilon": _NUMBER},
}


def _validate_params(policy_type: str, params: dict) -> None:
    spec = POLICY_PARAMS.get(policy_type, {})
    unknown = set(params) - set(spec)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown parameters for {policy_type!r}: {names}")
    for name, value in params.items():
        expected = spec[name]
        # bool is an int subclass; a boolean here is always a caller mistake
        if isinstance(value, bool) or not isinstance(value, expected):
            kind = "an integer" if expected is int else "a number"
            raise ValueError(
                f"parameter {name!r} for {policy_type!r} must be {kind}, got {value!r}"
            )


def build_policy(policy_config: dict, n_arms: int, seed: int | None) -> BanditPolicy:
    policy_type = policy_config.get("type")
    params = policy_config.get("params") or {}
    _validate_params(policy_type, params)
    if policy_type == "beta_ts":
        return BetaBernoulliTS(
            n_arms=n_arms,
            prior=(params.get("prior_alpha", 1.0), params.get("prior_beta", 1.0)),
            propensity_samples=params.get("propensity_samples", 200),
            seed=seed,
        )
    if policy_type == "gaussian_ts":
        return GaussianTS(
            n_arms=n_arms,
            prior_mean=params.get("prior_mean", 0.0),
            prior_var=params.get("prior_var", 1.0),
            obs_var=params.get("obs_var", 1.0),
            propensity_samples=params.get("propensity_samples", 200),
            seed=seed,
        )
    if policy_type == "epsilon_greedy":
        return EpsilonGreedy(
            n_arms=n_arms,
            epsilon=params.get("epsilon", 0.1),
            seed=seed,
        )
    raise ValueError(f"unknown policy type: {policy_type!r}")


def apply_state(policy: BanditPolicy, state: dict) -> None:
    """Overwrite a freshly-built policy's posterior with snapshot state."""
    kind = state.get("policy")
    if kind == "beta_bernoulli_ts":
        policy.alpha = np.asarray(state["alpha"], dtype=float)
        policy.beta = np.asarray(state["beta"], dtype=float)
    elif kind == "gaussian_ts":
        policy.counts = np.asarray(state["counts"], dtype=float)
        policy.reward_sums = np.asarray(state["reward_sums"], dtype=float)
    elif kind == "epsilon_greedy":
        policy.counts = np.asarray(state["counts"], dtype=float)
        policy.means = np.asarray(state["means"], dtype=float)
    else:
        raise ValueError(f"unknown snapshot kind: {kind!r}")
