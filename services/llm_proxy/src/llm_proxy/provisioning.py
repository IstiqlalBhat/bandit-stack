"""Create or verify the proxy's immutable decision route."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from llm_proxy.config import DecisionAPIConfig, RewardConfig


@dataclass(frozen=True)
class ProvisionedRoute:
    route_id: str
    created: bool


class ProvisioningError(RuntimeError):
    pass


def _expected_route(
    cfg: DecisionAPIConfig, ordered_arms: list[str], reward: RewardConfig
) -> dict:
    return {
        "name": cfg.route_name,
        "arms": ordered_arms,
        "policy": cfg.policy.model_dump(),
        "reward": reward.model_dump(),
        "seed": cfg.seed,
    }


def _validate_existing(existing: dict, expected: dict) -> None:
    if existing.get("arms") != expected["arms"]:
        raise ProvisioningError(
            f"route {expected['name']!r} has incompatible ordered arms; "
            "choose a new versioned route_name"
        )
    if existing.get("policy") != expected["policy"]:
        raise ProvisioningError(
            f"route {expected['name']!r} has incompatible policy; "
            "choose a new versioned route_name"
        )
    if existing.get("reward") != expected["reward"]:
        raise ProvisioningError(
            f"route {expected['name']!r} has incompatible reward tradeoff; "
            "choose a new versioned route_name"
        )
    if existing.get("seed") != expected["seed"]:
        raise ProvisioningError(
            f"route {expected['name']!r} has incompatible seed; "
            "choose a new versioned route_name"
        )


async def provision_route(
    client: httpx.AsyncClient,
    cfg: DecisionAPIConfig,
    ordered_arms: list[str],
    reward: RewardConfig,
) -> ProvisionedRoute:
    expected = _expected_route(cfg, ordered_arms, reward)
    response = await client.get(f"/routes/by-name/{cfg.route_name}")
    if response.status_code == 404:
        created = await client.post("/routes", json=expected)
        if created.status_code == 409:
            winner = await client.get(f"/routes/by-name/{cfg.route_name}")
            winner.raise_for_status()
            route = winner.json()
            _validate_existing(route, expected)
            return ProvisionedRoute(route_id=route["id"], created=False)
        created.raise_for_status()
        route = created.json()
        return ProvisionedRoute(route_id=route["id"], created=True)
    response.raise_for_status()
    existing = response.json()
    _validate_existing(existing, expected)
    return ProvisionedRoute(route_id=existing["id"], created=False)
