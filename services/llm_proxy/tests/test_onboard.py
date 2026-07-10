"""Operator onboarding CLI: provision/verify the decision route before first
traffic, with distinct exit codes per failure class so ops scripts can branch:
0 = ok (created or reused), 2 = incompatible route identity, 3 = decision API
unreachable or rejecting."""

import json

import httpx
import pytest

from llm_proxy.onboard import run_onboarding

ARMS = ["mock-mini", "mock-opus"]
EXPECTED_POLICY = {"type": "beta_ts", "params": {"propensity_samples": 64}}
EXPECTED_REWARD = {"usd_per_quality_point": 0.01}


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ONB_DECISION_KEY", "test-key")
    cfg = {
        "mode": "shadow",
        "default_model": "mock-mini",
        "models": [
            {"name": "mock-mini", "base_url": "http://upstream.test/v1",
             "api_key_env": None, "input_usd_per_mtok": 0.15, "output_usd_per_mtok": 0.60},
            {"name": "mock-opus", "base_url": "http://upstream.test/v1",
             "api_key_env": None, "input_usd_per_mtok": 5.0, "output_usd_per_mtok": 25.0},
        ],
        "decision_api": {
            "base_url": "http://decision.test",
            "api_key_env": "ONB_DECISION_KEY",
            "route_name": "onboard-route",
        },
        "auth": {
            "client_api_key_env": "ONB_CLIENT_KEY",
            "admin_api_key_env": "ONB_ADMIN_KEY",
        },
        "database_url": f"sqlite:///{tmp_path}/proxy.db",
    }
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def route_body(**overrides) -> dict:
    body = {
        "id": "r1",
        "name": "onboard-route",
        "arms": ARMS,
        "policy": EXPECTED_POLICY,
        "reward": EXPECTED_REWARD,
        "seed": None,
    }
    body.update(overrides)
    return body


class TestOnboarding:
    def test_creates_missing_route_with_bearer_auth(self, config_path):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("authorization"))
            if request.method == "GET":
                return httpx.Response(404, json={"detail": "not found"})
            assert request.method == "POST"
            assert json.loads(request.content)["arms"] == ARMS
            return httpx.Response(201, json=route_body())

        code, message = run_onboarding(config_path, transport=httpx.MockTransport(handler))
        assert code == 0
        assert "created" in message
        assert "r1" in message
        assert seen == ["Bearer test-key", "Bearer test-key"]

    def test_reuses_matching_route(self, config_path):
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json=route_body()))
        code, message = run_onboarding(config_path, transport=transport)
        assert code == 0
        assert "reused" in message

    def test_incompatible_route_exits_2(self, config_path):
        transport = httpx.MockTransport(
            lambda r: httpx.Response(200, json=route_body(arms=list(reversed(ARMS))))
        )
        code, message = run_onboarding(config_path, transport=transport)
        assert code == 2
        assert "incompatible" in message.lower()

    def test_unreachable_decision_api_exits_3(self, config_path):
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        code, message = run_onboarding(config_path, transport=httpx.MockTransport(down))
        assert code == 3
        assert "unreachable" in message.lower()

    def test_rejected_credentials_exit_3(self, config_path):
        transport = httpx.MockTransport(
            lambda r: httpx.Response(401, json={"detail": "invalid bearer token"})
        )
        code, message = run_onboarding(config_path, transport=transport)
        assert code == 3
        assert "401" in message
