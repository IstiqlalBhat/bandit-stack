"""OpenAI-compatible proxy with shadow and assisted modes.

Shadow: log what the bandit WOULD pick, serve the configured default.
Assisted: serve the bandit's pick (fail-open to the default).

Quality closes the loop: explicit feedback (`POST /feedback`) or a sampled
LLM judge scores a response, the score is composed with the request's cost
into one reward, and that reward is posted to the decision engine.

Causal attribution rule: a reward is only posted when the arm the bandit
picked is the arm that was actually served — otherwise one model's quality
would be attributed to another model's arm and corrupt the policy. At most
one reward is ever posted per request (first of feedback/judge wins).

The proxy must never be the reason a request fails: decision-engine errors
degrade to plain proxying and upstream errors pass through as-is.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from starlette.background import BackgroundTask

from evals import JudgeClient, composite_reward
from llm_proxy.config import ModelSpec, ProxyConfig
from llm_proxy.db import (
    RequestLog,
    claim_reward_slot,
    make_session_factory,
    prune_history,
    release_reward_slot,
    utcnow,
)
from llm_proxy.pricing import cost_usd, estimate_prompt_tokens, estimate_tokens
from llm_proxy.rate_limit import TokenBucket
from llm_proxy.security import bearer_auth, required_key
from llm_proxy.shadow import ShadowRouter


class FeedbackRequest(BaseModel):
    request_id: str
    quality: float = Field(ge=0.0, le=1.0)


def parse_sse(raw: bytes) -> tuple[str, dict | None]:
    """Extract concatenated delta content and the usage block (if any) from a
    buffered OpenAI SSE stream."""
    content_parts: list[str] = []
    usage = None
    for block in raw.decode(errors="replace").split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        if payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in event.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                content_parts.append(piece)
        if event.get("usage"):
            usage = event["usage"]
    return "".join(content_parts), usage


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def create_app(
    config: ProxyConfig,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    decision_transport: httpx.AsyncBaseTransport | None = None,
    judge_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    session_factory = make_session_factory(config.database_url)
    client_api_key = required_key(
        config.auth.client_api_key, config.auth.client_api_key_env
    )
    admin_api_key = required_key(
        config.auth.admin_api_key, config.auth.admin_api_key_env
    )
    decision_api_key = required_key(
        config.decision_api.api_key, config.decision_api.api_key_env
    )
    require_client = bearer_auth(client_api_key)
    require_admin = bearer_auth(admin_api_key)
    chat_limiter = TokenBucket(
        config.rate_limit.requests_per_minute, config.rate_limit.burst
    )

    class RateLimitExceeded(Exception):
        def __init__(self, retry_after: float) -> None:
            self.retry_after = retry_after

    def require_chat_access(request: Request) -> None:
        require_client(request)
        retry_after = chat_limiter.consume()
        if retry_after is not None:
            raise RateLimitExceeded(retry_after)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        prune_history(
            session_factory,
            utcnow() - timedelta(days=config.retention_days),
        )
        app.state.upstream = httpx.AsyncClient(
            transport=upstream_transport, timeout=httpx.Timeout(120.0, connect=10.0)
        )
        decision_client = httpx.AsyncClient(
            transport=decision_transport,
            base_url=config.decision_api.base_url,
            timeout=config.decision_api.timeout_s,
            headers={"authorization": f"Bearer {decision_api_key}"},
        )
        app.state.decision = decision_client
        app.state.shadow = ShadowRouter(
            decision_client,
            config.decision_api,
            [m.name for m in config.models],
            config.reward,
        )
        await app.state.shadow.prepare()
        judge_client = None
        app.state.judge = None
        if config.judge is not None:
            headers = {}
            if config.judge.api_key:
                headers["authorization"] = f"Bearer {config.judge.api_key}"
            judge_client = httpx.AsyncClient(
                transport=judge_transport,
                base_url=config.judge.base_url,
                timeout=httpx.Timeout(30.0),
                headers=headers,
            )
            app.state.judge = JudgeClient(
                judge_client, config.judge.model, extra_body=config.judge.params
            )
        yield
        await app.state.upstream.aclose()
        await decision_client.aclose()
        if judge_client is not None:
            await judge_client.aclose()

    app = FastAPI(title="llm-proxy", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            {
                "error": {
                    "message": "rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
            status_code=429,
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))},
        )

    # ---- persistence helpers -------------------------------------------------
    def write_log(entry: dict) -> None:
        with session_factory() as session:
            session.add(RequestLog(**entry))
            session.commit()

    def update_row(request_id: str, **fields) -> None:
        with session_factory() as session:
            session.execute(
                update(RequestLog).where(RequestLog.request_id == request_id).values(**fields)
            )
            session.commit()

    def get_row(request_id: str) -> RequestLog | None:
        with session_factory() as session:
            return session.execute(
                select(RequestLog).where(RequestLog.request_id == request_id).limit(1)
            ).scalar_one_or_none()

    # ---- reward plumbing -----------------------------------------------------
    async def post_reward(decision_id: str, value: float) -> bool:
        try:
            resp = await app.state.decision.post(
                "/rewards",
                json={"decision_id": decision_id, "value": value, "component": "composite"},
            )
            return resp.status_code == 200
        except Exception:
            return False

    def reward_eligible(entry_or_row) -> bool:
        get = (
            entry_or_row.get
            if isinstance(entry_or_row, dict)
            else lambda k: getattr(entry_or_row, k)
        )
        return bool(get("decision_id")) and get("shadow_model") == get("served_model")

    async def maybe_judge(entry: dict, task: str, response_text: str) -> None:
        judge: JudgeClient | None = app.state.judge
        if judge is None or config.judge is None:
            return
        if entry["status_code"] != 200 or not reward_eligible(entry):
            return
        if random.random() >= config.judge.sample_rate:
            return
        quality = await judge.score(task, response_text)
        if quality is None:
            return
        value = composite_reward(
            quality, entry["cost_usd"] or 0.0, config.reward.usd_per_quality_point
        )
        # atomic claim: explicit feedback may have raced us while we scored
        if not claim_reward_slot(session_factory, entry["request_id"]):
            return
        posted = await post_reward(entry["decision_id"], value)
        if not posted:
            release_reward_slot(session_factory, entry["request_id"])
        update_row(
            entry["request_id"],
            quality=quality,
            quality_source="judge",
            reward_value=value,
        )

    def forward_headers(request: Request, model: ModelSpec) -> dict:
        headers = {
            k: v for k, v in request.headers.items() if k.lower().startswith("x-")
        }
        if model.api_key:
            headers["authorization"] = f"Bearer {model.api_key}"
        return headers

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        shadow = app.state.shadow
        if not shadow.ready:
            # self-heal: startup ordering or an engine outage must not brick
            # readiness — each probe re-attempts provisioning
            await shadow.prepare()
        body = {
            "ready": shadow.ready,
            "route_name": config.decision_api.route_name,
        }
        if shadow.ready:
            return body
        return JSONResponse(body, status_code=503)

    # ---- the request path ------------------------------------------------
    @app.post(
        "/v1/chat/completions", dependencies=[Depends(require_chat_access)]
    )
    async def chat_completions(request: Request):
        body = await request.json()
        client_model = body.get("model")
        messages = body.get("messages") or []
        is_stream = bool(body.get("stream"))

        pick = await request.app.state.shadow.decide(
            {
                "client_model": client_model,
                "n_messages": len(messages),
                "prompt_chars": sum(len(str(m.get("content") or "")) for m in messages),
                "stream": is_stream,
            }
        )
        served = config.model_by_name(config.default_model)
        if config.mode == "assisted" and pick is not None:
            try:
                served = config.model_by_name(pick.model)
            except KeyError:
                pass  # engine named an arm we no longer serve — keep default
        body["model"] = served.name

        rid = uuid.uuid4().hex
        entry: dict = {
            "request_id": rid,
            "client_requested_model": client_model,
            "served_model": served.name,
            "shadow_model": pick.model if pick else None,
            "decision_id": pick.decision_id if pick else None,
            "propensity": pick.propensity if pick else None,
            "stream": is_stream,
            "status_code": 0,
            "latency_ms": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost_usd": None,
            "cost_source": None,
            "error": None,
        }
        rid_header = {"x-proxy-request-id": rid}
        task_text = last_user_text(messages)
        url = f"{served.base_url}/chat/completions"
        headers = forward_headers(request, served)
        upstream: httpx.AsyncClient = request.app.state.upstream
        t0 = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - t0) * 1000

        def fill_costs(usage: dict | None, response_text: str) -> None:
            if usage and usage.get("prompt_tokens") is not None:
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                source = "usage"
            else:
                pt = estimate_prompt_tokens(messages)
                ct = estimate_tokens(response_text)
                source = "estimated"
            entry.update(
                prompt_tokens=pt,
                completion_tokens=ct,
                cost_usd=cost_usd(served, pt, ct),
                cost_source=source,
            )

        if is_stream:
            req = upstream.build_request("POST", url, json=body, headers=headers)
            try:
                resp = await upstream.send(req, stream=True)
            except httpx.TransportError as exc:
                entry.update(
                    status_code=502, latency_ms=elapsed_ms(), error=f"upstream unreachable: {exc}"
                )
                return JSONResponse(
                    {"error": {"message": "upstream unreachable"}},
                    status_code=502,
                    headers=rid_header,
                    background=BackgroundTask(write_log, entry),
                )
            if resp.status_code != 200:
                content = await resp.aread()
                await resp.aclose()
                entry.update(
                    status_code=resp.status_code,
                    latency_ms=elapsed_ms(),
                    error=content.decode(errors="replace")[:2000],
                )
                return Response(
                    content=content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                    headers=rid_header,
                    background=BackgroundTask(write_log, entry),
                )

            buffer = bytearray()

            async def passthrough():
                try:
                    async for chunk in resp.aiter_bytes():
                        buffer.extend(chunk)
                        yield chunk
                finally:
                    await resp.aclose()

            async def finalize_stream() -> None:
                content, usage = parse_sse(bytes(buffer))
                entry["status_code"] = 200
                entry["latency_ms"] = elapsed_ms()
                fill_costs(usage, content)
                write_log(entry)
                await maybe_judge(entry, task_text, content)

            return StreamingResponse(
                passthrough(),
                media_type=resp.headers.get("content-type", "text/event-stream"),
                headers=rid_header,
                background=BackgroundTask(finalize_stream),
            )

        try:
            resp = await upstream.post(url, json=body, headers=headers)
        except httpx.TransportError as exc:
            entry.update(
                status_code=502, latency_ms=elapsed_ms(), error=f"upstream unreachable: {exc}"
            )
            return JSONResponse(
                {"error": {"message": "upstream unreachable"}},
                status_code=502,
                headers=rid_header,
                background=BackgroundTask(write_log, entry),
            )
        entry["status_code"] = resp.status_code
        entry["latency_ms"] = elapsed_ms()
        if resp.status_code != 200:
            entry["error"] = resp.text[:2000]
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
                headers=rid_header,
                background=BackgroundTask(write_log, entry),
            )
        data = resp.json()
        response_text = "".join(
            str((c.get("message") or {}).get("content") or "")
            for c in data.get("choices") or []
        )
        fill_costs(data.get("usage"), response_text)

        async def finalize_json() -> None:
            write_log(entry)
            await maybe_judge(entry, task_text, response_text)

        return JSONResponse(
            data, headers=rid_header, background=BackgroundTask(finalize_json)
        )

    # ---- feedback → reward -------------------------------------------------
    @app.post("/feedback", dependencies=[Depends(require_client)])
    async def feedback(body: FeedbackRequest) -> dict:
        row = None
        for _ in range(4):  # the log row lands in a background task; tolerate a beat
            row = get_row(body.request_id)
            if row is not None:
                break
            await asyncio.sleep(0.05)
        if row is None:
            raise HTTPException(status_code=404, detail=f"request {body.request_id} not found")
        if row.reward_posted:
            raise HTTPException(status_code=409, detail="reward already posted for this request")

        value = None
        posted = False
        if reward_eligible(row):
            value = composite_reward(
                body.quality, row.cost_usd or 0.0, config.reward.usd_per_quality_point
            )
            # atomic claim: the background judge may be scoring this request now
            if not claim_reward_slot(session_factory, body.request_id):
                raise HTTPException(
                    status_code=409, detail="reward already posted for this request"
                )
            posted = await post_reward(row.decision_id, value)
            if not posted:
                release_reward_slot(session_factory, body.request_id)
        update_row(
            body.request_id,
            quality=body.quality,
            quality_source="explicit",
            reward_value=value,
        )
        return {
            "ok": True,
            "quality": body.quality,
            "reward": value,
            "reward_posted": posted,
        }

    # ---- admin -------------------------------------------------------------
    @app.get("/admin/requests", dependencies=[Depends(require_admin)])
    def admin_requests(limit: int = 50) -> list[dict]:
        with session_factory() as session:
            rows = (
                session.execute(
                    select(RequestLog).order_by(RequestLog.id.desc()).limit(min(limit, 1000))
                )
                .scalars()
                .all()
            )
            return [r.to_dict() for r in rows]

    @app.get("/admin/summary", dependencies=[Depends(require_admin)])
    def admin_summary() -> dict:
        with session_factory() as session:
            n = session.scalar(select(func.count()).select_from(RequestLog))
            total = session.scalar(
                select(func.coalesce(func.sum(RequestLog.cost_usd), 0.0))
            )
            served = dict(
                session.execute(
                    select(RequestLog.served_model, func.count()).group_by(
                        RequestLog.served_model
                    )
                ).all()
            )
            shadow = dict(
                session.execute(
                    select(RequestLog.shadow_model, func.count())
                    .where(RequestLog.shadow_model.is_not(None))
                    .group_by(RequestLog.shadow_model)
                ).all()
            )
            missing = session.scalar(
                select(func.count())
                .select_from(RequestLog)
                .where(RequestLog.shadow_model.is_(None))
            )
            rewards_posted = session.scalar(
                select(func.count())
                .select_from(RequestLog)
                .where(RequestLog.reward_posted.is_(True))
            )
            return {
                "n_requests": n,
                "total_cost_usd": total,
                "served": served,
                "shadow": shadow,
                "shadow_missing": missing,
                "rewards_posted": rewards_posted,
            }

    return app
