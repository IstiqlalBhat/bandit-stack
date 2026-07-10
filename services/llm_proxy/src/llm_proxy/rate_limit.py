"""Process-local token bucket for single-worker pilot deployments."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        requests_per_minute: int,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate_per_second = requests_per_minute / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._updated_at = clock()
        self._lock = threading.Lock()

    def consume(self) -> float | None:
        """Consume one token, or return seconds until a token is available."""
        with self._lock:
            now = self._clock()
            elapsed = max(0.0, now - self._updated_at)
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate_per_second,
            )
            self._updated_at = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return None
            return (1.0 - self._tokens) / self._rate_per_second
