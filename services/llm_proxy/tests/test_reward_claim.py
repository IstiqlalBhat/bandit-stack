"""Single-reward invariant under real concurrency.

A slow LLM judge (background task) and an immediate explicit feedback can race
to post a reward for the same request. Posting must be guarded by an atomic
claim on the request row — one UPDATE ... WHERE reward_posted = false — so
exactly one of them ever reaches the decision engine.
"""

from llm_proxy.db import (
    RequestLog,
    claim_reward_slot,
    make_session_factory,
    release_reward_slot,
)


def make_factory(tmp_path):
    factory = make_session_factory(f"sqlite:///{tmp_path}/claims.db")
    with factory() as session:
        session.add(
            RequestLog(
                request_id="r-1",
                served_model="mock-mini",
                stream=False,
                status_code=200,
                latency_ms=1.0,
            )
        )
        session.commit()
    return factory


def test_only_the_first_claim_wins(tmp_path):
    factory = make_factory(tmp_path)
    assert claim_reward_slot(factory, "r-1") is True
    assert claim_reward_slot(factory, "r-1") is False
    assert claim_reward_slot(factory, "r-1") is False


def test_release_reopens_the_slot(tmp_path):
    factory = make_factory(tmp_path)
    assert claim_reward_slot(factory, "r-1") is True
    release_reward_slot(factory, "r-1")
    assert claim_reward_slot(factory, "r-1") is True


def test_claiming_unknown_request_fails(tmp_path):
    factory = make_factory(tmp_path)
    assert claim_reward_slot(factory, "ghost") is False
