"""M4 behavior: assisted routing, explicit feedback -> composite reward,
LLM-judge sampling, and the causal attribution rule.

Attribution rule under test: a reward may only be posted to the decision
engine when the arm the bandit picked is the arm that was actually served
(always true in assisted mode; true in shadow mode only when the pick
coincides with the default). Anything else would label one model's quality
with another model's arm and corrupt the policy.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_proxy.app import create_app
from llm_proxy.config import JudgeConfig
from test_proxy import (
    AUTH_HEADERS,
    AuthenticatedTestClient,
    REQUEST,
    decision_handler,
    make_config,
    upstream_handler,
)

U = 0.01  # usd_per_quality_point used throughout
MINI_COST = (12 * 0.15 + 5 * 0.60) / 1e6
OPUS_COST = (12 * 5.0 + 5 * 25.0) / 1e6


def build_client(
    tmp_path,
    *,
    mode="assisted",
    judge=None,
    judge_reply="SCORE: 0.5",
    decision_fails_rewards=False,
    reward_sink=None,
):
    def decision(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/rewards":
            if decision_fails_rewards:
                raise httpx.ConnectError("rewards down")
            if reward_sink is not None:
                reward_sink.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True, "policy_version": 1})
        return decision_handler(request)

    def judge_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": judge_reply}}]},
        )

    app = create_app(
        make_config(tmp_path, mode=mode, reward={"usd_per_quality_point": U}, judge=judge),
        upstream_transport=httpx.MockTransport(upstream_handler),
        decision_transport=httpx.MockTransport(decision),
        judge_transport=httpx.MockTransport(judge_handler),
    )
    return AuthenticatedTestClient(app)


class TestAssistedServing:
    def test_serves_the_bandit_pick(self, tmp_path):
        sink = []
        with build_client(tmp_path, reward_sink=sink) as client:
            resp = client.post("/v1/chat/completions", json=REQUEST)
            assert resp.status_code == 200
            # decision_handler picks mock-opus; upstream echoes the model it served
            assert resp.json()["model"] == "mock-opus"
            entry = client.get("/admin/requests").json()[0]
            assert entry["served_model"] == "mock-opus"
            assert entry["shadow_model"] == "mock-opus"
            assert entry["cost_usd"] == pytest.approx(OPUS_COST)

    def test_fails_open_to_default_when_engine_down(self, tmp_path):
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("decision api down")

        app = create_app(
            make_config(tmp_path, mode="assisted"),
            upstream_transport=httpx.MockTransport(upstream_handler),
            decision_transport=httpx.MockTransport(down),
        )
        with AuthenticatedTestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=REQUEST)
            assert resp.status_code == 200
            assert resp.json()["model"] == "mock-mini"


class TestRequestIdHeader:
    def test_header_matches_log_for_both_paths(self, tmp_path):
        with build_client(tmp_path) as client:
            resp = client.post("/v1/chat/completions", json=REQUEST)
            rid = resp.headers["x-proxy-request-id"]
            assert client.get("/admin/requests").json()[0]["request_id"] == rid

            with client.stream(
                "POST", "/v1/chat/completions", json={**REQUEST, "stream": True}
            ) as sresp:
                srid = sresp.headers["x-proxy-request-id"]
                b"".join(sresp.iter_bytes())
            assert client.get("/admin/requests").json()[0]["request_id"] == srid
            assert srid != rid


class TestFeedback:
    def test_posts_composite_reward_when_pick_was_served(self, tmp_path):
        sink = []
        with build_client(tmp_path, reward_sink=sink) as client:
            rid = client.post("/v1/chat/completions", json=REQUEST).headers["x-proxy-request-id"]
            resp = client.post("/feedback", json={"request_id": rid, "quality": 1.0})
            assert resp.status_code == 200
            body = resp.json()
            expected = 1.0 - OPUS_COST / U
            assert body["reward_posted"] is True
            assert body["reward"] == pytest.approx(expected)
            assert len(sink) == 1
            assert sink[0]["decision_id"] == "d1"
            assert sink[0]["component"] == "composite"
            assert sink[0]["value"] == pytest.approx(expected)
            entry = client.get("/admin/requests").json()[0]
            assert entry["quality"] == 1.0
            assert entry["quality_source"] == "explicit"
            assert entry["reward_value"] == pytest.approx(expected)
            assert entry["reward_posted"] is True

    def test_records_quality_but_skips_reward_on_shadow_mismatch(self, tmp_path):
        # shadow mode: pick is mock-opus but mock-mini is served -> no reward
        sink = []
        with build_client(tmp_path, mode="shadow", reward_sink=sink) as client:
            rid = client.post("/v1/chat/completions", json=REQUEST).headers["x-proxy-request-id"]
            resp = client.post("/feedback", json={"request_id": rid, "quality": 1.0})
            assert resp.status_code == 200
            assert resp.json()["reward_posted"] is False
            assert sink == []
            entry = client.get("/admin/requests").json()[0]
            assert entry["quality"] == 1.0
            assert entry["reward_posted"] is False

    def test_unknown_request_404(self, tmp_path):
        with build_client(tmp_path) as client:
            assert (
                client.post("/feedback", json={"request_id": "nope", "quality": 1.0}).status_code
                == 404
            )

    def test_duplicate_feedback_409(self, tmp_path):
        sink = []
        with build_client(tmp_path, reward_sink=sink) as client:
            rid = client.post("/v1/chat/completions", json=REQUEST).headers["x-proxy-request-id"]
            assert client.post("/feedback", json={"request_id": rid, "quality": 1.0}).status_code == 200
            assert client.post("/feedback", json={"request_id": rid, "quality": 0.0}).status_code == 409
            assert len(sink) == 1

    def test_quality_out_of_range_422(self, tmp_path):
        with build_client(tmp_path) as client:
            rid = client.post("/v1/chat/completions", json=REQUEST).headers["x-proxy-request-id"]
            assert client.post("/feedback", json={"request_id": rid, "quality": 1.5}).status_code == 422

    def test_reward_post_failure_is_fail_open(self, tmp_path):
        with build_client(tmp_path, decision_fails_rewards=True) as client:
            rid = client.post("/v1/chat/completions", json=REQUEST).headers["x-proxy-request-id"]
            resp = client.post("/feedback", json={"request_id": rid, "quality": 1.0})
            assert resp.status_code == 200
            assert resp.json()["reward_posted"] is False
            entry = client.get("/admin/requests").json()[0]
            assert entry["quality"] == 1.0
            assert entry["reward_posted"] is False


def judge_cfg(sample_rate: float) -> JudgeConfig:
    return JudgeConfig(
        base_url="http://judge.test/v1", model="judge-1", sample_rate=sample_rate
    )


class TestJudgeSampling:
    def test_judge_scores_and_posts_reward(self, tmp_path):
        sink = []
        with build_client(tmp_path, judge=judge_cfg(1.0), reward_sink=sink) as client:
            client.post("/v1/chat/completions", json=REQUEST)
            entry = client.get("/admin/requests").json()[0]
            expected = 0.5 - OPUS_COST / U
            assert entry["quality"] == pytest.approx(0.5)
            assert entry["quality_source"] == "judge"
            assert entry["reward_value"] == pytest.approx(expected)
            assert entry["reward_posted"] is True
            assert len(sink) == 1
            assert sink[0]["value"] == pytest.approx(expected)

    def test_judge_scores_streamed_responses(self, tmp_path):
        with build_client(tmp_path, judge=judge_cfg(1.0)) as client:
            with client.stream(
                "POST", "/v1/chat/completions", json={**REQUEST, "stream": True}
            ) as resp:
                b"".join(resp.iter_bytes())
            entry = client.get("/admin/requests").json()[0]
            assert entry["quality"] == pytest.approx(0.5)
            assert entry["quality_source"] == "judge"

    def test_sample_rate_zero_never_judges(self, tmp_path):
        sink = []
        with build_client(tmp_path, judge=judge_cfg(0.0), reward_sink=sink) as client:
            client.post("/v1/chat/completions", json=REQUEST)
            entry = client.get("/admin/requests").json()[0]
            assert entry["quality"] is None
            assert sink == []

    def test_unparseable_judge_reply_posts_nothing(self, tmp_path):
        sink = []
        with build_client(
            tmp_path, judge=judge_cfg(1.0), judge_reply="wonderful!", reward_sink=sink
        ) as client:
            client.post("/v1/chat/completions", json=REQUEST)
            entry = client.get("/admin/requests").json()[0]
            assert entry["quality"] is None
            assert entry["reward_posted"] is False
            assert sink == []

    def test_feedback_after_judge_reward_is_409(self, tmp_path):
        with build_client(tmp_path, judge=judge_cfg(1.0)) as client:
            rid = client.post("/v1/chat/completions", json=REQUEST).headers["x-proxy-request-id"]
            assert client.post("/feedback", json={"request_id": rid, "quality": 1.0}).status_code == 409


class TestReadyzRecovery:
    def test_readyz_recovers_when_decision_api_comes_back(self, tmp_path):
        # startup provisioning fails (decision API not up yet); the readiness
        # probe itself must retry, so ordering of service startup never
        # permanently bricks readiness
        calls = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise httpx.ConnectError("decision api still booting")
            return decision_handler(request)

        app = create_app(
            make_config(tmp_path),
            upstream_transport=httpx.MockTransport(upstream_handler),
            decision_transport=httpx.MockTransport(flaky),
        )
        with TestClient(app) as client:
            assert client.get("/readyz").status_code == 503  # retried, still down
            assert client.get("/readyz").status_code == 200  # engine is back
            assert client.get("/readyz").json()["ready"] is True
