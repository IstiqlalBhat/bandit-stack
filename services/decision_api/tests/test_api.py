"""API contract + behavior tests for the decision service.

The learning tests drive the service exactly like a customer would: POST
/decide, observe a synthetic outcome, POST /rewards — and assert the policy
converges, survives a process restart, and yields sane OPE estimates.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from decision_api.app import create_app

ARMS = ["haiku", "sonnet", "opus"]
TRUE_PROBS = {"haiku": 0.2, "sonnet": 0.5, "opus": 0.9}
INTERNAL_HEADERS = {"authorization": "Bearer internal-test-key"}

BETA_ROUTE = {
    "name": "llm-router",
    "arms": ARMS,
    "policy": {"type": "beta_ts", "params": {"propensity_samples": 64}},
    "seed": 7,
}


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/test.db"


@pytest.fixture
def client(db_url):
    with TestClient(
        create_app(db_url, api_key="internal-test-key"),
        headers=INTERNAL_HEADERS,
    ) as c:
        yield c


def run_rounds(client: TestClient, route_id: str, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    decisions = []
    for _ in range(n):
        d = client.post(f"/routes/{route_id}/decide", json={}).json()
        reward = float(rng.random() < TRUE_PROBS[d["arm_name"]])
        r = client.post("/rewards", json={"decision_id": d["decision_id"], "value": reward})
        assert r.status_code == 200
        decisions.append(d)
    return decisions


class TestRoutes:
    def test_create_route(self, client):
        resp = client.post("/routes", json=BETA_ROUTE)
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"]
        assert body["name"] == "llm-router"
        assert body["arms"] == ARMS

    def test_rejects_unknown_policy_type(self, client):
        bad = {**BETA_ROUTE, "policy": {"type": "dqn"}}
        assert client.post("/routes", json=bad).status_code == 422

    def test_rejects_fewer_than_two_arms(self, client):
        bad = {**BETA_ROUTE, "arms": ["only-one"]}
        assert client.post("/routes", json=bad).status_code == 422

    def test_rejects_duplicate_arms(self, client):
        bad = {**BETA_ROUTE, "arms": ["same", "same"]}
        assert client.post("/routes", json=bad).status_code == 422

    def test_rejects_blank_arm_names(self, client):
        bad = {**BETA_ROUTE, "arms": ["valid", "  "]}
        assert client.post("/routes", json=bad).status_code == 422

    def test_route_name_must_be_url_safe(self, client):
        bad = {**BETA_ROUTE, "name": "customer/route"}
        assert client.post("/routes", json=bad).status_code == 422

    def test_unknown_route_fields_are_rejected(self, client):
        bad = {**BETA_ROUTE, "inline_api_key": "must-not-be-accepted"}
        assert client.post("/routes", json=bad).status_code == 422

    def test_rejects_unknown_policy_parameters(self, client):
        bad = {
            **BETA_ROUTE,
            "policy": {"type": "beta_ts", "params": {"propensit_samples": 64}},
        }
        assert client.post("/routes", json=bad).status_code == 422

    def test_rejects_invalid_policy_parameter_types_at_creation(self, client):
        bad = {
            **BETA_ROUTE,
            "policy": {"type": "beta_ts", "params": {"propensity_samples": "many"}},
        }
        assert client.post("/routes", json=bad).status_code == 422

    def test_lookup_by_name(self, client):
        created = client.post("/routes", json=BETA_ROUTE).json()
        resp = client.get(f"/routes/by-name/{BETA_ROUTE['name']}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == created["id"]
        assert body["arms"] == ARMS

    def test_lookup_by_name_404(self, client):
        assert client.get("/routes/by-name/never-created").status_code == 404

    def test_reward_tradeoff_round_trips_as_route_identity(self, client):
        route = {
            **BETA_ROUTE,
            "name": "llm-router-v2",
            "reward": {"usd_per_quality_point": 0.0025},
        }
        created = client.post("/routes", json=route)
        assert created.status_code == 201
        assert created.json()["reward"] == {"usd_per_quality_point": 0.0025}
        fetched = client.get("/routes/by-name/llm-router-v2")
        assert fetched.json()["reward"] == {"usd_per_quality_point": 0.0025}


class TestDecide:
    def test_returns_valid_decision(self, client):
        route_id = client.post("/routes", json=BETA_ROUTE).json()["id"]
        resp = client.post(f"/routes/{route_id}/decide", json={})
        assert resp.status_code == 200
        d = resp.json()
        assert d["arm_index"] in range(3)
        assert d["arm_name"] == ARMS[d["arm_index"]]
        assert 0.0 < d["propensity"] < 1.0

    def test_decisions_get_unique_ids(self, client):
        route_id = client.post("/routes", json=BETA_ROUTE).json()["id"]
        ids = {client.post(f"/routes/{route_id}/decide", json={}).json()["decision_id"] for _ in range(5)}
        assert len(ids) == 5

    def test_unknown_route_404(self, client):
        assert client.post("/routes/nope/decide", json={}).status_code == 404


class TestRewards:
    def test_unknown_decision_404(self, client):
        client.post("/routes", json=BETA_ROUTE)
        resp = client.post("/rewards", json={"decision_id": "nope", "value": 1.0})
        assert resp.status_code == 404

    def test_out_of_range_reward_for_beta_ts_422(self, client):
        route_id = client.post("/routes", json=BETA_ROUTE).json()["id"]
        d = client.post(f"/routes/{route_id}/decide", json={}).json()
        resp = client.post("/rewards", json={"decision_id": d["decision_id"], "value": 2.0})
        assert resp.status_code == 422


class TestLearning:
    def test_policy_concentrates_on_best_arm(self, client):
        route_id = client.post("/routes", json=BETA_ROUTE).json()["id"]
        decisions = run_rounds(client, route_id, n=400, seed=0)

        last = [d["arm_name"] for d in decisions[-100:]]
        assert last.count("opus") / len(last) > 0.6

        state = client.get(f"/routes/{route_id}/state").json()
        assert state["n_decisions"] == 400
        assert state["n_rewards"] == 400
        alpha = np.array(state["state"]["alpha"])
        beta = np.array(state["state"]["beta"])
        posterior_means = alpha / (alpha + beta)
        assert int(np.argmax(posterior_means)) == ARMS.index("opus")

    def test_state_survives_restart(self, client, db_url):
        route_id = client.post("/routes", json=BETA_ROUTE).json()["id"]
        run_rounds(client, route_id, n=50, seed=1)
        state_before = client.get(f"/routes/{route_id}/state").json()

        with TestClient(
            create_app(db_url, api_key="internal-test-key"),
            headers=INTERNAL_HEADERS,
        ) as fresh:
            state_after = fresh.get(f"/routes/{route_id}/state").json()
            assert state_after["state"] == state_before["state"]
            assert state_after["policy_version"] == state_before["policy_version"]
            # and the restored policy still serves decisions
            assert fresh.post(f"/routes/{route_id}/decide", json={}).status_code == 200


class TestOPE:
    def make_uniform_route(self, client) -> str:
        route = {
            "name": "uniform-logger",
            "arms": ARMS,
            # epsilon=1.0 => pure uniform behavior with exact propensity 1/3
            "policy": {"type": "epsilon_greedy", "params": {"epsilon": 1.0}},
            "seed": 11,
        }
        return client.post("/routes", json=route).json()["id"]

    def test_estimates_fixed_arm_and_uniform_targets(self, client):
        route_id = self.make_uniform_route(client)
        run_rounds(client, route_id, n=600, seed=2)

        resp = client.post(
            f"/routes/{route_id}/ope", json={"target_probs": [0.0, 0.0, 1.0]}
        )
        assert resp.status_code == 200
        report = resp.json()
        assert report["n"] == 600
        assert report["snips"] == pytest.approx(TRUE_PROBS["opus"], abs=0.1)

        uniform = client.post(
            f"/routes/{route_id}/ope", json={"target_probs": [1 / 3, 1 / 3, 1 / 3]}
        ).json()
        truth = float(np.mean(list(TRUE_PROBS.values())))
        assert uniform["snips"] == pytest.approx(truth, abs=0.1)

    def test_rejects_invalid_target_probs(self, client):
        route_id = self.make_uniform_route(client)
        run_rounds(client, route_id, n=10, seed=3)
        resp = client.post(
            f"/routes/{route_id}/ope", json={"target_probs": [0.5, 0.5, 0.5]}
        )
        assert resp.status_code == 422

    def test_no_rewarded_decisions_yet_409(self, client):
        route_id = self.make_uniform_route(client)
        resp = client.post(
            f"/routes/{route_id}/ope", json={"target_probs": [0.0, 0.0, 1.0]}
        )
        assert resp.status_code == 409
