"""Deterministic heuristic scorers. Each returns a quality in [0, 1]."""

from __future__ import annotations

import json
import re

_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)


def score_json_validity(text: str) -> float:
    """1.0 if the text (optionally inside a code fence) parses as JSON."""
    candidate = text.strip()
    fenced = _CODE_FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        json.loads(candidate)
        return 1.0
    except (json.JSONDecodeError, ValueError):
        return 0.0
