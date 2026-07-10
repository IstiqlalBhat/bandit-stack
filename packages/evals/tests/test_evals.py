import asyncio
import json

import httpx
import pytest

from evals import JudgeClient, composite_reward, score_json_validity


class TestCompositeReward:
    def test_quality_minus_cost_in_quality_units(self):
        # 0.9 quality, $0.000185 cost, $0.001 offsets one quality point
        assert composite_reward(0.9, 0.000185, usd_per_quality_point=0.001) == pytest.approx(
            0.9 - 0.185
        )

    def test_clamps_to_unit_interval(self):
        assert composite_reward(0.1, 1.0, usd_per_quality_point=0.001) == 0.0
        assert composite_reward(1.0, 0.0, usd_per_quality_point=0.001) == 1.0

    def test_rejects_nonpositive_lambda(self):
        with pytest.raises(ValueError):
            composite_reward(0.5, 0.1, usd_per_quality_point=0.0)

    def test_rejects_quality_outside_unit_interval(self):
        with pytest.raises(ValueError):
            composite_reward(1.5, 0.0, usd_per_quality_point=0.01)


class TestJsonValidity:
    def test_valid_json_scores_one(self):
        assert score_json_validity('{"a": 1}') == 1.0
        assert score_json_validity("[1, 2]") == 1.0

    def test_invalid_json_scores_zero(self):
        assert score_json_validity("not json {") == 0.0

    def test_json_inside_code_fence_counts(self):
        assert score_json_validity('```json\n{"a": 1}\n```') == 1.0


def make_judge(reply: str | Exception, seen: list | None = None, **kwargs) -> JudgeClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(reply, Exception):
            raise reply
        body = json.loads(request.content)
        assert body["model"] == "judge-1"
        if seen is not None:
            seen.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply}}
                ]
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://judge.test/v1"
    )
    return JudgeClient(client=client, model="judge-1", **kwargs)


def run(coro):
    return asyncio.run(coro)


class TestJudgeClient:
    def test_parses_score_line(self):
        judge = make_judge("SCORE: 0.8")
        assert run(judge.score("What is 2+2?", "4")) == pytest.approx(0.8)

    def test_clamps_out_of_range_scores(self):
        judge = make_judge("SCORE: 1.4")
        assert run(judge.score("q", "r")) == 1.0

    def test_score_embedded_in_prose(self):
        judge = make_judge("The response is decent.\nSCORE: 0.65\n")
        assert run(judge.score("q", "r")) == pytest.approx(0.65)

    def test_garbage_returns_none(self):
        judge = make_judge("a delightful answer!")
        assert run(judge.score("q", "r")) is None

    def test_transport_error_returns_none(self):
        judge = make_judge(httpx.ConnectError("judge down"))
        assert run(judge.score("q", "r")) is None

    def test_uses_max_completion_tokens_not_max_tokens(self):
        # reasoning models (gpt-5.x) reject max_tokens outright
        seen: list = []
        judge = make_judge("SCORE: 0.7", seen)
        run(judge.score("q", "r"))
        assert "max_tokens" not in seen[0]
        assert seen[0]["max_completion_tokens"] == 512

    def test_extra_body_merges_into_request(self):
        seen: list = []
        judge = make_judge("SCORE: 0.7", seen, extra_body={"reasoning_effort": "minimal"})
        run(judge.score("q", "r"))
        assert seen[0]["reasoning_effort"] == "minimal"
