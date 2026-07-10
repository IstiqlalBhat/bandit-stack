from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import update

from llm_proxy.app import create_app
from llm_proxy.db import RequestLog, make_session_factory
from test_proxy import (
    REQUEST,
    AuthenticatedTestClient,
    decision_handler,
    make_config,
    upstream_handler,
)


def make_app(config):
    return create_app(
        config,
        upstream_transport=httpx.MockTransport(upstream_handler),
        decision_transport=httpx.MockTransport(decision_handler),
    )


def test_startup_prunes_expired_request_logs_and_keeps_recent_rows(tmp_path):
    config = make_config(tmp_path, retention_days=30)
    with AuthenticatedTestClient(make_app(config)) as client:
        old_id = client.post(
            "/v1/chat/completions", json=REQUEST
        ).headers["x-proxy-request-id"]
        recent_id = client.post(
            "/v1/chat/completions", json=REQUEST
        ).headers["x-proxy-request-id"]

    session_factory = make_session_factory(config.database_url)
    expired_at = datetime.now(timezone.utc) - timedelta(days=31)
    with session_factory() as session:
        session.execute(
            update(RequestLog)
            .where(RequestLog.request_id == old_id)
            .values(created_at=expired_at)
        )
        session.commit()

    with AuthenticatedTestClient(make_app(config)) as restarted:
        request_ids = [
            row["request_id"] for row in restarted.get("/admin/requests").json()
        ]

    assert request_ids == [recent_id]
