"""M1 regret report: policy comparison plots on synthetic environments.

Usage: uv run python apps/reports/regret_report.py
Writes docs/plots/m1-regret-curves.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bandit_core import BetaBernoulliTS, EpsilonGreedy, LinTS
from bandit_core.simulation import BernoulliEnv, LinearContextualEnv, compare
from bandit_core.simulation.plots import plot_regret

HORIZON = 3000
N_SEEDS = 8


def main() -> None:
    probs = [0.3, 0.5, 0.7]
    bernoulli = compare(
        policy_factories={
            "Thompson sampling": lambda s: BetaBernoulliTS(
                n_arms=3, propensity_samples=32, seed=1000 + s
            ),
            "ε-greedy (ε=0.1)": lambda s: EpsilonGreedy(n_arms=3, epsilon=0.1, seed=1000 + s),
            "ε-greedy (ε=0.01)": lambda s: EpsilonGreedy(n_arms=3, epsilon=0.01, seed=1000 + s),
        },
        env_factory=lambda s: BernoulliEnv(probs, seed=2000 + s),
        horizon=HORIZON,
        n_seeds=N_SEEDS,
    )

    theta = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    contextual = compare(
        policy_factories={
            "LinTS": lambda s: LinTS(dim=5, propensity_samples=32, seed=1000 + s),
            "ε-greedy (context-blind)": lambda s: EpsilonGreedy(
                n_arms=4, epsilon=0.1, seed=1000 + s
            ),
        },
        env_factory=lambda s: LinearContextualEnv(theta, n_arms=4, noise=0.1, seed=2000 + s),
        horizon=HORIZON,
        n_seeds=N_SEEDS,
    )

    out = Path(__file__).resolve().parents[2] / "docs" / "plots" / "m1-regret-curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_regret(
        {
            f"Bernoulli bandit, probs={probs}": bernoulli,
            "Linear contextual bandit (d=5, 4 arms)": contextual,
        },
        path=str(out),
        suptitle=f"Cumulative regret over {HORIZON} rounds ({N_SEEDS} seeds, mean ± 1 std)",
    )

    for env_name, results in {"bernoulli": bernoulli, "contextual": contextual}.items():
        print(f"\n{env_name} — final cumulative regret (mean over {N_SEEDS} seeds):")
        for policy_name, runs in results.items():
            print(f"  {policy_name:28s} {runs[:, -1].mean():8.1f} ± {runs[:, -1].std():.1f}")
    print(f"\nplot saved to {out}")


if __name__ == "__main__":
    main()
