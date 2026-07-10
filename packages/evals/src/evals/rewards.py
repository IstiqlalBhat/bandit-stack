"""Composite reward: quality-per-dollar in one number.

`usd_per_quality_point` is the customer's exchange rate — how many dollars of
request cost cancel out one full quality point. Smaller values mean cost
matters more. The result is clamped to [0, 1] so it can feed Beta-Bernoulli
Thompson sampling directly.
"""

from __future__ import annotations


def composite_reward(quality: float, cost_usd: float, usd_per_quality_point: float) -> float:
    if usd_per_quality_point <= 0:
        raise ValueError("usd_per_quality_point must be positive")
    if not 0.0 <= quality <= 1.0:
        raise ValueError(f"quality must be in [0, 1], got {quality}")
    if cost_usd < 0:
        raise ValueError("cost_usd must be nonnegative")
    return min(1.0, max(0.0, quality - cost_usd / usd_per_quality_point))
