"""Single-tenant Bearer authentication for the decision service."""

from __future__ import annotations

import secrets
import os
from collections.abc import Callable

from fastapi import HTTPException, Request


def required_env_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def bearer_auth(expected_key: str | None) -> Callable[[Request], None]:
    def require_bearer(request: Request) -> None:
        if expected_key is None:
            return
        authorization = request.headers.get("authorization", "")
        scheme, separator, supplied = authorization.partition(" ")
        valid = (
            separator == " "
            and scheme.lower() == "bearer"
            and bool(supplied)
            and secrets.compare_digest(supplied, expected_key)
        )
        if not valid:
            raise HTTPException(
                status_code=401,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_bearer
