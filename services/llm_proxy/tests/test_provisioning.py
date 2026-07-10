import asyncio
import json

import httpx
import pytest

from llm_proxy.config import DecisionAPIConfig, RewardConfig
from llm_proxy.provisioning import provision_route


ARMS = ["mock-mini", "mock-opus"]


def route_json(**overrides):
    route = {
        "id": "route-1",
        "name": "pilot-v1",
        "arms": ARMS,
        "policy": {
            "type": "beta_ts",
            "params": {"propensity_samples": 64},
        },
        "reward": {"usd_per_quality_point": 0.01},
        "seed": 7,
    }
    route.update(overrides)
    return route


def test_exact_existing_route_is_reused_without_post():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json=route_json())

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
        policy={"type": "beta_ts", "params": {"propensity_samples": 64}},
        seed=7,
    )
    reward = RewardConfig(usd_per_quality_point=0.01)
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, reward)

    result = asyncio.run(run())
    assert result.route_id == "route-1"
    assert result.created is False
    assert calls == [("GET", "/routes/by-name/pilot-v1")]


def test_missing_route_is_created_with_complete_identity():
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404, json={"detail": "missing"})
        posted.append(json.loads(request.content))
        return httpx.Response(201, json={"id": "route-2", **posted[-1]})

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v2",
        seed=11,
    )
    reward = RewardConfig(usd_per_quality_point=0.002)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, reward)

    result = asyncio.run(run())
    assert result.route_id == "route-2"
    assert result.created is True
    assert posted == [
        {
            "name": "pilot-v2",
            "arms": ARMS,
            "policy": {
                "type": "beta_ts",
                "params": {"propensity_samples": 64},
            },
            "reward": {"usd_per_quality_point": 0.002},
            "seed": 11,
        }
    ]


def test_existing_route_with_reordered_arms_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=route_json(id="stale", arms=list(reversed(ARMS)))
        )

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
        seed=7,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, RewardConfig())

    with pytest.raises(RuntimeError, match="ordered arms.*versioned route_name"):
        asyncio.run(run())


def test_existing_route_with_different_policy_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=route_json(policy={"type": "epsilon_greedy", "params": {"epsilon": 0.1}}),
        )

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
        seed=7,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, RewardConfig())

    with pytest.raises(RuntimeError, match="policy.*versioned route_name"):
        asyncio.run(run())


def test_existing_route_with_different_reward_tradeoff_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=route_json(reward={"usd_per_quality_point": 0.0001}),
        )

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
        seed=7,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, RewardConfig())

    with pytest.raises(RuntimeError, match="reward.*versioned route_name"):
        asyncio.run(run())


def test_existing_route_with_different_seed_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=route_json(seed=99))

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
        seed=7,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, RewardConfig())

    with pytest.raises(RuntimeError, match="seed.*versioned route_name"):
        asyncio.run(run())


def test_create_race_refetches_and_validates_winner():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if calls == ["GET"]:
            return httpx.Response(404, json={"detail": "missing"})
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "winner created it"})
        return httpx.Response(200, json=route_json())

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
        seed=7,
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            return await provision_route(client, cfg, ARMS, RewardConfig())

    result = asyncio.run(run())
    assert result.route_id == "route-1"
    assert result.created is False
    assert calls == ["GET", "POST", "GET"]
