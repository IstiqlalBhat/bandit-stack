import asyncio

import httpx

from llm_proxy.config import DecisionAPIConfig, RewardConfig
from llm_proxy.shadow import ShadowRouter


def test_deleted_cached_route_is_reprovisioned_and_decision_retried_once():
    lookups = 0
    decision_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal lookups
        if request.method == "GET":
            lookups += 1
            route_id = "old" if lookups == 1 else "new"
            return httpx.Response(
                200,
                json={
                    "id": route_id,
                    "name": "pilot-v1",
                    "arms": ["mini", "premium"],
                    "policy": {
                        "type": "beta_ts",
                        "params": {"propensity_samples": 64},
                    },
                    "reward": {"usd_per_quality_point": 0.01},
                    "seed": None,
                },
            )
        decision_paths.append(request.url.path)
        if request.url.path == "/routes/old/decide":
            return httpx.Response(404, json={"detail": "route deleted"})
        return httpx.Response(
            200,
            json={
                "decision_id": "decision-new",
                "arm_index": 0,
                "arm_name": "mini",
                "propensity": 0.6,
            },
        )

    cfg = DecisionAPIConfig(
        base_url="http://decision.test",
        api_key_env="DECISION_API_TEST_KEY",
        route_name="pilot-v1",
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url=cfg.base_url
        ) as client:
            router = ShadowRouter(
                client, cfg, ["mini", "premium"], RewardConfig()
            )
            await router.prepare()
            return await router.decide({"kind": "test"}), router.ready

    pick, ready = asyncio.run(run())
    assert pick is not None
    assert pick.decision_id == "decision-new"
    assert ready is True
    assert decision_paths == ["/routes/old/decide", "/routes/new/decide"]
