"""Mock OpenAI-compatible upstream for live e2e verification.

Non-stream returns "pong" with fixed usage; stream returns "Hello world" in
three chunks (usage chunk only when stream_options asks); X-Fail: 1 header
forces a 500. Echoes back the requested model so the proxy's model override
is observable.

Task mode: a last user message like "What is 3+4?" gets a numeric answer that
is correct with a per-model probability — simulating real quality differences
between cheap and premium models (seed via MOCK_SEED for reproducibility).

Run: uv run uvicorn mock_upstream:app --app-dir apps/reports --port 9341
"""

import json
import os
import random
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()

STREAM_PIECES = ["Hel", "lo ", "world"]
USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}

TASK_RE = re.compile(r"What is (\d+)\s*\+\s*(\d+)\?")
MODEL_ACCURACY = {"mock-mini": 0.60, "mock-sonnet": 0.92, "mock-opus": 0.95}
_rng = random.Random(int(os.environ["MOCK_SEED"]) if os.environ.get("MOCK_SEED") else None)


def task_answer(model: str, content: str) -> str | None:
    match = TASK_RE.search(content)
    if not match:
        return None
    total = int(match.group(1)) + int(match.group(2))
    correct = _rng.random() < MODEL_ACCURACY.get(model, 0.5)
    return f"The answer is {total if correct else total + 1}."


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if request.headers.get("x-fail") == "1":
        return JSONResponse({"error": {"message": "upstream boom"}}, status_code=500)
    model = body.get("model")
    if body.get("stream"):
        events = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            for piece in STREAM_PIECES
        ]
        events.append(
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        if (body.get("stream_options") or {}).get("include_usage"):
            events.append(
                {"id": "c1", "object": "chat.completion.chunk", "model": model, "choices": [], "usage": USAGE}
            )
        payload = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
        payload += b"data: [DONE]\n\n"
        return Response(content=payload, media_type="text/event-stream")
    last_user = next(
        (str(m.get("content") or "") for m in reversed(body.get("messages") or []) if m.get("role") == "user"),
        "",
    )
    if model == "mock-judge":
        content = "SCORE: 0.8"
    else:
        content = task_answer(model, last_user) or "pong"
    return JSONResponse(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": USAGE,
        }
    )
