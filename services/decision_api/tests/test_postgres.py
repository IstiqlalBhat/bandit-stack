import os
import uuid

import pytest
from fastapi.testclient import TestClient

from decision_api.app import create_app


POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
INTERNAL_HEADERS = {"authorization": "Bearer internal-test-key"}


@pytest.mark.skipif(POSTGRES_URL is None, reason="set TEST_POSTGRES_URL for PostgreSQL")
def test_postgresql_round_trip_and_restart():
    route_name = f"postgres-{uuid.uuid4().hex}"
    with TestClient(
        create_app(POSTGRES_URL, api_key="internal-test-key"),
        headers=INTERNAL_HEADERS,
    ) as client:
        created = client.post(
            "/routes",
            json={
                "name": route_name,
                "arms": ["cheap", "premium"],
                "policy": {
                    "type": "beta_ts",
                    "params": {"propensity_samples": 32},
                },
                "seed": 19,
            },
        )
        assert created.status_code == 201
        route_id = created.json()["id"]
        decision = client.post(
            f"/routes/{route_id}/decide",
            json={"context": {"nested": {"tier": "pilot"}}},
        )
        assert decision.status_code == 200
        assert 0.0 < decision.json()["propensity"] < 1.0
        reward = client.post(
            "/rewards",
            json={"decision_id": decision.json()["decision_id"], "value": 0.75},
        )
        assert reward.status_code == 200

    with TestClient(
        create_app(POSTGRES_URL, api_key="internal-test-key"),
        headers=INTERNAL_HEADERS,
    ) as restarted:
        route = restarted.get(f"/routes/by-name/{route_name}")
        assert route.status_code == 200
        state = restarted.get(f"/routes/{route_id}/state").json()
        assert state["n_decisions"] == 1
        assert state["n_rewards"] == 1
        assert state["policy_version"] == 1
