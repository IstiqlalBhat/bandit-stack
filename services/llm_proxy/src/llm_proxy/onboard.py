"""Operator onboarding: provision or verify the proxy's decision route before
sending the first traffic.

    uv run python -m llm_proxy.onboard configs/llm_proxy.json

Exit codes: 0 = route created or reused; 2 = a route with this name exists but
its identity (arms/policy/reward/seed) is incompatible — pick a new versioned
route_name; 3 = decision API unreachable or rejecting (check base_url and the
key in $<api_key_env>).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from llm_proxy.config import load_config
from llm_proxy.provisioning import ProvisioningError, provision_route

EXIT_OK = 0
EXIT_INCOMPATIBLE = 2
EXIT_UNREACHABLE = 3


def run_onboarding(
    config_path: str, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[int, str]:
    config = load_config(config_path)
    cfg = config.decision_api
    arms = [m.name for m in config.models]
    headers = {}
    if cfg.api_key:
        headers["authorization"] = f"Bearer {cfg.api_key}"

    async def go():
        async with httpx.AsyncClient(
            transport=transport, base_url=cfg.base_url, timeout=5.0, headers=headers
        ) as client:
            return await provision_route(client, cfg, arms, config.reward)

    try:
        route = asyncio.run(go())
    except ProvisioningError as exc:
        return EXIT_INCOMPATIBLE, f"INCOMPATIBLE: {exc}"
    except httpx.HTTPStatusError as exc:
        return (
            EXIT_UNREACHABLE,
            f"UNREACHABLE: decision API at {cfg.base_url} answered "
            f"{exc.response.status_code} — check the key in ${cfg.api_key_env}",
        )
    except httpx.TransportError as exc:
        return EXIT_UNREACHABLE, f"UNREACHABLE: {cfg.base_url}: {exc}"

    verb = "created" if route.created else "reused"
    return EXIT_OK, (
        f"OK: route {cfg.route_name!r} {verb} (id {route.route_id}), "
        f"arms {arms}, mode {config.mode!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the proxy config JSON")
    args = parser.parse_args(argv)
    code, message = run_onboarding(args.config)
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
