import pytest

from llm_proxy.rate_limit import TokenBucket


def test_token_bucket_refills_at_configured_rate():
    now = 100.0

    def clock() -> float:
        return now

    bucket = TokenBucket(requests_per_minute=60, burst=1, clock=clock)
    assert bucket.consume() is None
    assert bucket.consume() == pytest.approx(1.0)

    now += 1.0
    assert bucket.consume() is None
