"""Regret-curve plotting. matplotlib is an optional dependency of the
workspace (dev group), not of bandit_core itself — import stays local."""

from __future__ import annotations

import numpy as np


def plot_regret(
    results_by_env: dict[str, dict[str, np.ndarray]],
    path: str,
    suptitle: str = "Cumulative regret (mean ± 1 std)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results_by_env)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, (env_name, results) in zip(axes[0], results_by_env.items()):
        for policy_name, runs in results.items():
            t = np.arange(1, runs.shape[1] + 1)
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            ax.plot(t, mean, label=policy_name)
            # cumulative regret is nonnegative; keep the band honest
            ax.fill_between(t, np.maximum(mean - std, 0.0), mean + std, alpha=0.2)
        ax.set_title(env_name)
        ax.set_xlabel("round")
        ax.set_ylabel("cumulative regret")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
