"""LLM-as-judge over any OpenAI-compatible endpoint.

Best-effort by design: any transport error, bad status, or unparseable reply
returns None — a missing quality sample, never a broken pipeline.
"""

from __future__ import annotations

import re

import httpx

RUBRIC = (
    "You are a strict evaluator. Given a task and a response, rate how well "
    "the response completes the task. Consider correctness first, then "
    "completeness and clarity. Reply with exactly one line: 'SCORE: <x>' "
    "where <x> is a number from 0.0 (useless) to 1.0 (perfect)."
)

_SCORE = re.compile(r"SCORE:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


class JudgeClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str,
        rubric: str = RUBRIC,
        max_tokens: int = 16,
    ) -> None:
        self._client = client
        self.model = model
        self.rubric = rubric
        self.max_tokens = max_tokens

    async def score(self, task: str, response: str) -> float | None:
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [
                        {"role": "system", "content": self.rubric},
                        {
                            "role": "user",
                            "content": f"Task:\n{task}\n\nResponse:\n{response}",
                        },
                    ],
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return None
        match = _SCORE.search(content or "")
        if not match:
            return None
        return min(1.0, max(0.0, float(match.group(1))))
