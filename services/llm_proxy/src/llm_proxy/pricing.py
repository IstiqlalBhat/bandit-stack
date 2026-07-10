"""Cost accounting. Exact when the upstream reports token usage; otherwise a
chars/4 estimate, always labeled as such in the request log."""

from __future__ import annotations

from llm_proxy.config import ModelSpec


def estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content") or "")) for m in messages)


def cost_usd(model: ModelSpec, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * model.input_usd_per_mtok
        + completion_tokens * model.output_usd_per_mtok
    ) / 1e6
