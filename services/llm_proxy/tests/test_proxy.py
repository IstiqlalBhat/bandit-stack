"""Proxy behavior tests with mocked upstream + decision-api transports.

Contract under test: an OpenAI-format request always gets served by the
configured default model (shadow mode), the shadow bandit pick is logged but
never affects serving, costs are tracked (exact when the upstream reports
usage, estimated otherwise), streams pass through byte-identical, and any
decision-api failure degrades to plain proxying — never a broken request.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_proxy.app import create_app
from llm_proxy.config import DecisionAPIConfig, ModelSpec, ProxyConfig
from llm_proxy.pricing import estimate_tokens

STREAM_PIECES = ["Hel", "lo ", "world"]
REQUEST = {"model": "gpt-4o", "messages": [{"role": "user", "content": "ping"}]}
USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
MINI_RATES = (0.15, 0.60)  # usd per mtok in, out
AUTH_HEADERS = {"authorization": "Bearer client-test-key"}
ADMIN_HEADERS = {"authorization": "Bearer admin-test-key"}


class AuthenticatedTestClient(TestClient):
    def __init__(self, app, **kwargs):
        headers = {**AUTH_HEADERS, **(kwargs.pop("headers", {}) or {})}
        super().__init__(app, headers=headers, **kwargs)

    def request(self, method, url, **kwargs):
        if str(url).startswith("/admin/"):
            kwargs["headers"] = {
                **ADMIN_HEADERS,
                **(kwargs.get("headers", {}) or {}),
            }
        return super().request(method, url, **kwargs)


def make_config(tmp_path, **overrides) -> ProxyConfig:
    base = dict(
        mode="shadow",
        default_model="mock-mini",
        models=[
            ModelSpec(
                name="mock-mini",
                base_url="http://upstream.test/v1",
                api_key_env="MOCK_UPSTREAM_KEY",
                input_usd_per_mtok=MINI_RATES[0],
                output_usd_per_mtok=MINI_RATES[1],
            ),
            ModelSpec(
                name="mock-opus",
                base_url="http://upstream.test/v1",
                api_key_env=None,
                input_usd_per_mtok=5.0,
                output_usd_per_mtok=25.0,
            ),
        ],
        decision_api=DecisionAPIConfig(
            base_url="http://decision.test",
            api_key_env="DECISION_API_TEST_KEY",
            route_name="llm-proxy-shadow",
        ),
        database_url=f"sqlite:///{tmp_path}/proxy.db",
        auth={
            "client_api_key_env": "PROXY_CLIENT_TEST_KEY",
            "admin_api_key_env": "PROXY_ADMIN_TEST_KEY",
        },
    )
    base.update(overrides)
    return ProxyConfig(**base)


def upstream_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    if request.headers.get("x-fail") == "1":
        return httpx.Response(500, json={"error": {"message": "upstream boom"}})
    if body.get("stream"):
        events = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            for piece in STREAM_PIECES
        ]
        events.append(
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": body["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        if (body.get("stream_options") or {}).get("include_usage"):
            events.append(
                {"id": "c1", "object": "chat.completion.chunk", "model": body["model"], "choices": [], "usage": USAGE}
            )
        payload = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
        payload += b"data: [DONE]\n\n"
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
            ],
            "usage": USAGE,
        },
    )


def decision_handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET" and request.url.path == "/routes/by-name/llm-proxy-shadow":
        return httpx.Response(
            200,
            json={
                "id": "r1",
                "name": "llm-proxy-shadow",
                "arms": ["mock-mini", "mock-opus"],
                "policy": {
                    "type": "beta_ts",
                    "params": {"propensity_samples": 64},
                },
                "reward": {"usd_per_quality_point": 0.01},
                "seed": None,
            },
        )
    if request.method == "POST" and request.url.path == "/routes/r1/decide":
        return httpx.Response(
            200,
            json={"decision_id": "d1", "arm_index": 1, "arm_name": "mock-opus", "propensity": 0.5},
        )
    return httpx.Response(404, json={"detail": "unexpected call"})


def make_client(tmp_path, decision) -> TestClient:
    app = create_app(
        make_config(tmp_path),
        upstream_transport=httpx.MockTransport(upstream_handler),
        decision_transport=decision,
    )
    return AuthenticatedTestClient(app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_UPSTREAM_KEY", "sk-mock")
    with make_client(tmp_path, httpx.MockTransport(decision_handler)) as c:
        yield c


@pytest.fixture
def failopen_client(tmp_path):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("decision api down")

    with make_client(tmp_path, httpx.MockTransport(down)) as c:
        yield c


class TestNonStreaming:
    def test_passthrough_with_model_override(self, client):
        resp = client.post("/v1/chat/completions", json=REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "pong"
        # upstream echoes the model it was asked for => proves the override
        assert body["model"] == "mock-mini"

    def test_request_logged_with_exact_usage_cost(self, client):
        client.post("/v1/chat/completions", json=REQUEST)
        logs = client.get("/admin/requests").json()
        assert len(logs) == 1
        entry = logs[0]
        assert entry["served_model"] == "mock-mini"
        assert entry["client_requested_model"] == "gpt-4o"
        assert entry["prompt_tokens"] == USAGE["prompt_tokens"]
        assert entry["completion_tokens"] == USAGE["completion_tokens"]
        expected = (12 * MINI_RATES[0] + 5 * MINI_RATES[1]) / 1e6
        assert entry["cost_usd"] == pytest.approx(expected)
        assert entry["cost_source"] == "usage"
        assert entry["status_code"] == 200
        assert entry["latency_ms"] >= 0

    def test_shadow_decision_recorded_but_not_served(self, client):
        resp = client.post("/v1/chat/completions", json=REQUEST)
        assert resp.json()["model"] == "mock-mini"  # served default...
        entry = client.get("/admin/requests").json()[0]
        assert entry["shadow_model"] == "mock-opus"  # ...while bandit picked opus
        assert entry["decision_id"] == "d1"
        assert entry["propensity"] == 0.5

    def test_upstream_error_passes_through(self, client):
        resp = client.post("/v1/chat/completions", json=REQUEST, headers={"x-fail": "1"})
        assert resp.status_code == 500
        assert "boom" in resp.text
        entry = client.get("/admin/requests").json()[0]
        assert entry["status_code"] == 500
        assert entry["cost_usd"] is None
        assert entry["error"]

    def test_upstream_gets_server_key_not_client_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MOCK_UPSTREAM_KEY", "sk-mock")
        seen = {}

        def capturing(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return upstream_handler(request)

        app = create_app(
            make_config(tmp_path),
            upstream_transport=httpx.MockTransport(capturing),
            decision_transport=httpx.MockTransport(decision_handler),
        )
        with TestClient(app) as c:
            c.post(
                "/v1/chat/completions",
                json=REQUEST,
                headers=AUTH_HEADERS,
            )
        assert seen["auth"] == "Bearer sk-mock"


class TestStreaming:
    def test_sse_passthrough(self, client):
        with client.stream(
            "POST", "/v1/chat/completions", json={**REQUEST, "stream": True}
        ) as resp:
            assert resp.status_code == 200
            raw = b"".join(resp.iter_bytes())
        text = raw.decode()
        assert text.endswith("data: [DONE]\n\n")
        events = [
            json.loads(line[len("data: "):])
            for line in text.strip().split("\n\n")
            if line != "data: [DONE]"
        ]
        content = "".join(
            e["choices"][0]["delta"].get("content", "") for e in events if e["choices"]
        )
        assert content == "Hello world"

    def test_stream_with_usage_reports_exact_cost(self, client):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={**REQUEST, "stream": True, "stream_options": {"include_usage": True}},
        ) as resp:
            b"".join(resp.iter_bytes())
        entry = client.get("/admin/requests").json()[0]
        assert entry["stream"] is True
        assert entry["cost_source"] == "usage"
        assert entry["completion_tokens"] == USAGE["completion_tokens"]

    def test_stream_without_usage_estimates_cost(self, client):
        with client.stream(
            "POST", "/v1/chat/completions", json={**REQUEST, "stream": True}
        ) as resp:
            b"".join(resp.iter_bytes())
        entry = client.get("/admin/requests").json()[0]
        assert entry["cost_source"] == "estimated"
        assert entry["completion_tokens"] == estimate_tokens("Hello world")
        assert entry["cost_usd"] > 0


class TestFailOpen:
    def test_serves_normally_when_decision_api_down(self, failopen_client):
        resp = failopen_client.post("/v1/chat/completions", json=REQUEST)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "pong"
        entry = failopen_client.get("/admin/requests").json()[0]
        assert entry["shadow_model"] is None
        assert entry["decision_id"] is None


class TestReadiness:
    def test_reports_compatible_provisioned_route(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"ready": True, "route_name": "llm-proxy-shadow"}


class TestAdminSummary:
    def test_totals_and_shadow_counts(self, client):
        for _ in range(3):
            client.post("/v1/chat/completions", json=REQUEST)
        summary = client.get("/admin/summary").json()
        assert summary["n_requests"] == 3
        assert summary["total_cost_usd"] == pytest.approx(
            3 * (12 * MINI_RATES[0] + 5 * MINI_RATES[1]) / 1e6
        )
        assert summary["served"]["mock-mini"] == 3
        assert summary["shadow"]["mock-opus"] == 3
        assert summary["shadow_missing"] == 0


class TestConfig:
    def test_retention_days_must_be_positive(self, tmp_path):
        assert make_config(tmp_path, retention_days=30).retention_days == 30
        with pytest.raises(ValueError, match="retention_days"):
            make_config(tmp_path, retention_days=0)

    def test_rate_limit_accepts_positive_rate_and_burst(self, tmp_path):
        config = make_config(
            tmp_path,
            rate_limit={"requests_per_minute": 120, "burst": 3},
        )
        assert config.rate_limit.requests_per_minute == 120
        assert config.rate_limit.burst == 3

    def test_client_and_admin_auth_configuration_is_required(self, tmp_path):
        raw = make_config(tmp_path).model_dump()
        raw.pop("auth", None)
        with pytest.raises(ValueError, match="auth"):
            ProxyConfig.model_validate(raw)

    def test_decision_api_internal_key_reference_is_required(self, tmp_path):
        raw = make_config(tmp_path).model_dump()
        raw["decision_api"].pop("api_key_env", None)
        with pytest.raises(ValueError, match="api_key_env"):
            ProxyConfig.model_validate(raw)

    def test_default_model_must_be_a_configured_model(self, tmp_path):
        with pytest.raises(ValueError):
            ProxyConfig(
                mode="shadow",
                default_model="ghost",
                models=make_config(tmp_path).models,
                decision_api=DecisionAPIConfig(
                    base_url="http://decision.test",
                    api_key_env="DECISION_API_TEST_KEY",
                    route_name="r",
                ),
                database_url=f"sqlite:///{tmp_path}/x.db",
            )

    def test_unknown_fields_are_rejected_instead_of_silently_ignored(self, tmp_path):
        raw = make_config(tmp_path).model_dump()
        raw["inline_api_key"] = "must-not-be-accepted"
        with pytest.raises(ValueError, match="inline_api_key"):
            ProxyConfig.model_validate(raw)

    def test_route_name_must_be_url_safe(self):
        with pytest.raises(ValueError, match="route_name"):
            DecisionAPIConfig(
                base_url="http://decision.test", route_name="customer/route"
            )


class TestPricing:
    def test_estimate_tokens_is_ceil_chars_over_4(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcde") == 2
