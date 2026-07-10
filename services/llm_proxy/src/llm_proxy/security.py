"""Bearer authentication helpers that never expose configured key values."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from fastapi import HTTPException, Request


def required_key(value: str | None, env_name: str) -> str:
    if not value:
        raise RuntimeError(f"{env_name} must be set")
    return value


def bearer_auth(expected_key: str) -> Callable[[Request], None]:
    def require_bearer(request: Request) -> None:
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
