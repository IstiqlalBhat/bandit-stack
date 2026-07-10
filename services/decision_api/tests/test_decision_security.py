import pytest
from fastapi.testclient import TestClient

from decision_api.app import create_app
from decision_api.security import required_env_key


ROUTE = {
    "name": "secured-route",
    "arms": ["mini", "premium"],
    "policy": {"type": "beta_ts", "params": {"propensity_samples": 32}},
}


def test_internal_bearer_key_protects_api_while_health_stays_public(tmp_path):
    app = create_app(
        f"sqlite:///{tmp_path}/decision.db", api_key="internal-test-key"
    )
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.post("/routes", json=ROUTE).status_code == 401
        assert client.post(
            "/routes",
            json=ROUTE,
            headers={"authorization": "Bearer wrong-key"},
        ).status_code == 401
        created = client.post(
            "/routes",
            json=ROUTE,
            headers={"authorization": "Bearer internal-test-key"},
        )
        assert created.status_code == 201


def test_every_decision_data_operation_requires_the_internal_key(tmp_path):
    app = create_app(
        f"sqlite:///{tmp_path}/decision.db", api_key="internal-test-key"
    )
    authorized = {"authorization": "Bearer internal-test-key"}
    with TestClient(app) as client:
        created = client.post("/routes", json=ROUTE, headers=authorized).json()
        route_id = created["id"]

        attempts = [
            client.get(f"/routes/by-name/{ROUTE['name']}"),
            client.post(f"/routes/{route_id}/decide", json={}),
            client.post(
                "/rewards", json={"decision_id": "not-visible", "value": 1.0}
            ),
            client.get(f"/routes/{route_id}/state"),
            client.post(
                f"/routes/{route_id}/ope", json={"target_probs": [0.5, 0.5]}
            ),
        ]

        assert [response.status_code for response in attempts] == [401] * len(attempts)


def test_required_key_fails_closed_when_environment_variable_is_missing(monkeypatch):
    monkeypatch.delenv("DECISION_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DECISION_API_KEY must be set"):
        required_env_key("DECISION_API_KEY")
