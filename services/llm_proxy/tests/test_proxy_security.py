import httpx
from fastapi.testclient import TestClient

from llm_proxy.app import create_app
from test_proxy import REQUEST, decision_handler, make_config, upstream_handler


CLIENT = {"authorization": "Bearer client-test-key"}
ADMIN = {"authorization": "Bearer admin-test-key"}


def test_client_and_admin_roles_are_separate_and_rejection_never_calls_upstream(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROXY_CLIENT_TEST_KEY", "client-test-key")
    monkeypatch.setenv("PROXY_ADMIN_TEST_KEY", "admin-test-key")
    monkeypatch.setenv("DECISION_API_TEST_KEY", "internal-test-key")
    upstream_calls = 0

    def counted_upstream(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return upstream_handler(request)

    app = create_app(
        make_config(tmp_path),
        upstream_transport=httpx.MockTransport(counted_upstream),
        decision_transport=httpx.MockTransport(decision_handler),
    )
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=REQUEST).status_code == 401
        assert client.post(
            "/v1/chat/completions",
            json=REQUEST,
            headers={"authorization": "Bearer wrong-key"},
        ).status_code == 401
        assert client.post(
            "/v1/chat/completions", json=REQUEST, headers=ADMIN
        ).status_code == 401
        assert client.post(
            "/v1/chat/completions", json=REQUEST, headers=CLIENT
        ).status_code == 200

        assert client.get("/admin/summary", headers=CLIENT).status_code == 401
        assert client.get("/admin/summary", headers=ADMIN).status_code == 200
        assert client.post(
            "/feedback",
            json={"request_id": "missing", "quality": 1.0},
            headers=ADMIN,
        ).status_code == 401

    assert upstream_calls == 1


def test_proxy_authenticates_every_decision_api_call(tmp_path, monkeypatch):
    monkeypatch.setenv("PROXY_CLIENT_TEST_KEY", "client-test-key")
    monkeypatch.setenv("PROXY_ADMIN_TEST_KEY", "admin-test-key")
    monkeypatch.setenv("DECISION_API_TEST_KEY", "internal-test-key")
    seen = []

    def protected_decision(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        if request.headers.get("authorization") != "Bearer internal-test-key":
            return httpx.Response(401, json={"detail": "invalid bearer token"})
        return decision_handler(request)

    app = create_app(
        make_config(tmp_path),
        upstream_transport=httpx.MockTransport(upstream_handler),
        decision_transport=httpx.MockTransport(protected_decision),
    )
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        assert client.post(
            "/v1/chat/completions", json=REQUEST, headers=CLIENT
        ).status_code == 200

    assert seen
    assert set(seen) == {"Bearer internal-test-key"}


def test_chat_burst_returns_openai_shaped_429_before_upstream(tmp_path, monkeypatch):
    monkeypatch.setenv("PROXY_CLIENT_TEST_KEY", "client-test-key")
    monkeypatch.setenv("PROXY_ADMIN_TEST_KEY", "admin-test-key")
    monkeypatch.setenv("DECISION_API_TEST_KEY", "internal-test-key")
    upstream_calls = 0

    def counted_upstream(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return upstream_handler(request)

    app = create_app(
        make_config(
            tmp_path,
            rate_limit={"requests_per_minute": 60, "burst": 2},
        ),
        upstream_transport=httpx.MockTransport(counted_upstream),
        decision_transport=httpx.MockTransport(decision_handler),
    )
    with TestClient(app) as client:
        assert client.post(
            "/v1/chat/completions", json=REQUEST, headers=CLIENT
        ).status_code == 200
        assert client.post(
            "/v1/chat/completions", json=REQUEST, headers=CLIENT
        ).status_code == 200
        limited = client.post(
            "/v1/chat/completions", json=REQUEST, headers=CLIENT
        )

    assert limited.status_code == 429
    assert limited.headers["retry-after"]
    assert limited.json() == {
        "error": {
            "message": "rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }
    assert upstream_calls == 2
